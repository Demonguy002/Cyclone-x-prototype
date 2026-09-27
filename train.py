
# ============================================================
# CYCLONE-X FINAL MODEL — LEAKAGE-FREE MULTI-SOURCE FUSION
# Adaptive Multi-Source Vision Fusion
#
# Historical training sources:
#   1) HURSAT-B1  -> satellite vision
#   2) IBTrACS    -> historical track / prior intensity context
#   3) ERA5       -> environmental context when available
#
# TARGET:
#   0 = Depression / Weak System
#   1 = Cyclonic Storm
#   2 = Severe Cyclone
#
# LEAKAGE CONTROL:
#   - Current WMO_WIND is ONLY the target, never an input.
#   - Current WMO_PRES is NEVER an input.
#   - All intensity-change/trend features use T-1, T-2, T-3 only.
#   - Test storms are completely unseen during training.
#   - Scalers are fitted on TRAIN ONLY.
#   - No target-derived satellite/ERA5 feature is used.
#
# OUTPUT:
#   E:\Cyclone-x-prototype\model1\
#
# The script creates the complete Model-1 artifact package in one run.
# ============================================================

import json
import math
import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Concatenate,
    Multiply,
    Add,
    Reshape,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(r"E:\Cyclone-x-prototype")

HURSAT_DIR = BASE / "datas" / "HURSAT_NI"
IBTRACS_FILE = BASE / "datas" / "IBTrACS.NI.v04r01.csv"

# Your ERA5 pilot/full data can be arranged as:
#   datas/single_levels/<SID>/*.nc
# or any nested structure containing <SID> in the path.
ERA5_DIR = BASE / "datas" / "single_levels"

MODEL_DIR = BASE / "model1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2010
END_YEAR = 2016

IMAGE_SIZE = 64
MAX_SAMPLES_PER_STORM = 120
MATCH_HOURS = 4.0

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

RANDOM_STATE = 42

EPOCHS = 20
BATCH_SIZE = 16

# If False, samples without ERA5 are retained with an ERA5-validity
# mask of 0. The fusion gate can then reduce the ERA5 contribution.
# This is intentional: adaptive missing-source handling is part of
# the CYCLONE-X design.
REQUIRE_ERA5 = False

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
# Keep CPU training predictable on a laptop; TensorFlow can still use all available CPU cores.
try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, (os.cpu_count() or 4) // 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

CLASS_NAMES = [
    "Depression / Weak System",
    "Cyclonic Storm",
    "Severe Cyclone",
]

# ============================================================
# HELPERS
# ============================================================

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def target_from_wind(wind):
    """Target only. This value is NEVER passed into the model."""
    if pd.isna(wind):
        return np.nan
    wind = float(wind)
    if wind < 34:
        return 0
    if wind < 64:
        return 1
    return 2


def extract_hursat_timestamp(filename):
    # Typical HURSAT timestamp:
    # ...YYYY.MM.DD.HHMM...
    m = re.search(r"\.(\d{4})\.(\d{2})\.(\d{2})\.(\d{4})\.", filename)
    if not m:
        # More tolerant fallback
        m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{4})", filename)
    if not m:
        return None

    try:
        return pd.Timestamp(
            year=int(m.group(1)),
            month=int(m.group(2)),
            day=int(m.group(3)),
            hour=int(m.group(4)[:2]),
            minute=int(m.group(4)[2:]),
        )
    except Exception:
        return None


def normalize_satellite(arr):
    """Convert one storm-centered satellite field into a normalized 64x64 image."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    # HURSAT files can carry a time/channel dimension. Reduce ONLY
    # non-spatial dimensions; never use ARCHER/intensity variables.
    if arr.ndim > 2:
        # For a typical (time, lat, lon, channel) field:
        # select the first time slice, then aggregate remaining channels.
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            # Spatial dimensions are expected to be the two largest dimensions.
            # Move the smallest/non-spatial dimension to the end if possible.
            shapes = list(arr.shape)
            spatial_axes = sorted(range(3), key=lambda i: shapes[i], reverse=True)[:2]
            channel_axis = [i for i in range(3) if i not in spatial_axes][0]
            arr = np.moveaxis(arr, channel_axis, -1)
            arr = np.nanmedian(arr, axis=-1)

    if arr.ndim != 2:
        return None

    finite = np.isfinite(arr)
    if finite.sum() < 100:
        return None

    fill = float(np.nanmedian(arr[finite]))
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)

    lo, hi = np.percentile(arr, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr))
        hi = float(np.max(arr))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)

    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    out = np.asarray(img, dtype=np.float32) / 255.0

    # A single HURSAT IR observation is intentionally kept as one physical
    # channel. The CNN consumes a 1-channel satellite image.
    return out[..., None]


def _hursat_2d(da):
    """Extract a spatial 2-D field from a HURSAT DataArray without using metadata."""
    arr = np.asarray(da.values)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    # Typical (time, lat, lon)
    if arr.ndim == 3:
        return arr[0].astype(np.float32)

    # Typical local HURSAT fallback observed as (time, lat, lon, channel).
    if arr.ndim == 4:
        arr = arr[0]
        # Identify the two largest dimensions as lat/lon and aggregate the
        # remaining spectral/channel dimension.
        shapes = list(arr.shape)
        spatial_axes = sorted(range(3), key=lambda i: shapes[i], reverse=True)[:2]
        channel_axis = [i for i in range(3) if i not in spatial_axes][0]
        arr = np.moveaxis(arr, channel_axis, -1)
        return np.nanmedian(arr, axis=-1).astype(np.float32)

    return None


def find_satellite_array(ds):
    """
    Select only genuine satellite-observation variables.

    Preferred HURSAT-B1 field:
      IRWIN / infrared-window brightness temperature.

    Fallback:
      rad10 when IRWIN is absent.

    ARCHER center/intensity/pressure variables are NEVER accepted.
    """
    preferred = [
        "IRWIN", "irwin",
        "IRWIN_CDR", "irwin_cdr",
        "IRWIN_V", "irwin_v",
        "rad10", "Rad10",
    ]

    # Never use variables whose names indicate derived storm intensity,
    # pressure, center-fix, ring/spiral scores, etc.
    forbidden = (
        "archer", "wind", "pres", "pressure", "intensity",
        "ring_score", "spiral_score", "center", "vmax", "mws"
    )

    names = list(ds.data_vars)
    ordered = []

    for wanted in preferred:
        for name in names:
            if name.lower() == wanted.lower() and name not in ordered:
                ordered.append(name)

    # Do not silently choose arbitrary metadata/derived variables.
    for name in ordered:
        if any(token in name.lower() for token in forbidden):
            continue
        try:
            arr = _hursat_2d(ds[name])
            if arr is not None and np.isfinite(arr).sum() >= 100:
                return arr
        except Exception:
            continue

    return None


def source_quality(image):
    """Label-independent quality score from the satellite image only."""
    if image is None:
        return 0.0
    x = image[..., 0]
    finite_ratio = float(np.isfinite(x).mean())
    dynamic = float(np.std(x))
    dynamic_score = min(1.0, dynamic / 0.15)
    return float(np.clip(0.5 * finite_ratio + 0.5 * dynamic_score, 0, 1))


def find_era5_variables(ds):
    """
    Map common ERA5 names to a stable feature schema.
    Only environmental variables are used.
    """
    aliases = {
        "u10": [
            "u10", "10u", "u_component_of_wind_10m",
            "10m_u_component_of_wind",
        ],
        "v10": [
            "v10", "10v", "v_component_of_wind_10m",
            "10m_v_component_of_wind",
        ],
        "mslp": [
            "msl", "mean_sea_level_pressure",
            "mslp", "mean_sea_level_pressure_pa",
        ],
        "sst": [
            "sst", "sea_surface_temperature",
        ],
        "t2m": [
            "t2m", "2m_temperature",
            "2m_air_temperature",
        ],
        "d2m": [
            "d2m", "2m_dewpoint_temperature",
            "2m_dewpoint_temperature",
        ],
        "tp": [
            "tp", "total_precipitation",
        ],
    }

    result = {}

    lower_map = {str(k).lower(): k for k in ds.data_vars}

    for canonical, names in aliases.items():
        for name in names:
            key = lower_map.get(name.lower())
            if key is not None:
                result[canonical] = key
                break

    return result


def find_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
    for c in ds.variables:
        if c in candidates:
            return c
    return None


def extract_era5_features(ds, timestamp, lat, lon):
    """
    Extract ERA5 at the nearest time/grid point.
    Returns fixed-length environmental features plus validity flag.
    """
    names = find_era5_variables(ds)

    # Stable feature order:
    # U10, V10, wind speed, wind direction, MSLP, SST,
    # T2M, D2M, TP
    values = np.zeros(9, dtype=np.float32)

    time_name = find_coord_name(
        ds, ["time", "valid_time", "forecast_reference_time"]
    )
    lat_name = find_coord_name(
        ds, ["latitude", "lat"]
    )
    lon_name = find_coord_name(
        ds, ["longitude", "lon"]
    )

    if not time_name or not lat_name or not lon_name:
        return values, 0.0

    try:
        tsel = ds.sel(
            {time_name: timestamp},
            method="nearest",
        )

        lats = tsel[lat_name].values
        lons = tsel[lon_name].values

        # ERA5 longitude can be 0..360 while IBTrACS is -180..180.
        query_lon = float(lon)
        if np.nanmax(lons) > 180 and query_lon < 0:
            query_lon = query_lon % 360

        point = tsel.sel(
            {
                lat_name: float(lat),
                lon_name: query_lon,
            },
            method="nearest",
        )

        def get(canonical):
            name = names.get(canonical)
            if name is None:
                return np.nan
            try:
                v = np.asarray(point[name].values).squeeze()
                if v.size == 0:
                    return np.nan
                return safe_float(v.reshape(-1)[0], np.nan)
            except Exception:
                return np.nan

        u10 = get("u10")
        v10 = get("v10")
        mslp = get("mslp")
        sst = get("sst")
        t2m = get("t2m")
        d2m = get("d2m")
        tp = get("tp")

        wind_speed = (
            math.sqrt(u10 * u10 + v10 * v10)
            if np.isfinite(u10) and np.isfinite(v10)
            else np.nan
        )

        wind_dir = (
            (math.degrees(math.atan2(-u10, -v10)) + 360.0) % 360.0
            if np.isfinite(u10) and np.isfinite(v10)
            else np.nan
        )

        vals = [
            u10, v10, wind_speed, wind_dir,
            mslp, sst, t2m, d2m, tp
        ]

        valid_count = sum(np.isfinite(v) for v in vals)

        for i, v in enumerate(vals):
            values[i] = float(v) if np.isfinite(v) else 0.0

        # Source is considered available if at least 3 environmental
        # quantities are present at the target point/time.
        valid = 1.0 if valid_count >= 3 else 0.0
        return values, valid

    except Exception:
        return values, 0.0


def discover_era5_files():
    mapping = {}

    if not ERA5_DIR.exists():
        return mapping

    for p in ERA5_DIR.rglob("*.nc"):
        parts = [x for x in p.parts]
        sid = None

        # Prefer a parent directory that looks like an IBTrACS SID.
        for part in reversed(parts[:-1]):
            if re.match(r"^\d{7}N\d{4}$", part):
                sid = part
                break

        if sid is None:
            # Fallback: find any SID-like string in the path.
            m = re.search(r"(\d{7}N\d{4})", str(p))
            if m:
                sid = m.group(1)

        if sid:
            mapping.setdefault(sid, []).append(p)

    return mapping


def nearest_row(storm_df, timestamp):
    times = storm_df["ISO_TIME"].values.astype("datetime64[ns]")
    target = np.datetime64(timestamp.to_datetime64())

    diff = np.abs(times - target).astype("timedelta64[s]").astype(np.int64)
    idx = int(np.argmin(diff))

    hours = float(diff[idx]) / 3600.0
    return storm_df.iloc[idx], hours


def build_previous_features(ib):
    """
    All intensity history is explicitly lagged before any change/rolling
    operation. No current WMO_WIND/WMO_PRES is allowed into X.
    """
    ib = ib.sort_values(["SID", "ISO_TIME"]).copy()
    g = ib.groupby("SID", group_keys=False)

    # Historical lags
    for col in [
        "WMO_WIND",
        "WMO_PRES",
        "LAT",
        "LON",
        "STORM_SPEED",
        "STORM_DIR",
        "DIST2LAND",
    ]:
        for lag in [1, 2, 3]:
            ib[f"{col}_L{lag}"] = g[col].shift(lag)

    # Previous-only trends
    ib["WIND_CHANGE_1"] = ib["WMO_WIND_L1"] - ib["WMO_WIND_L2"]
    ib["WIND_CHANGE_2"] = ib["WMO_WIND_L2"] - ib["WMO_WIND_L3"]
    ib["WIND_ACCELERATION"] = (
        ib["WIND_CHANGE_1"] - ib["WIND_CHANGE_2"]
    )

    ib["PRESSURE_CHANGE_1"] = (
        ib["WMO_PRES_L1"] - ib["WMO_PRES_L2"]
    )
    ib["PRESSURE_CHANGE_2"] = (
        ib["WMO_PRES_L2"] - ib["WMO_PRES_L3"]
    )
    ib["PRESSURE_ACCELERATION"] = (
        ib["PRESSURE_CHANGE_1"] - ib["PRESSURE_CHANGE_2"]
    )

    ib["SPEED_CHANGE_1"] = (
        ib["STORM_SPEED_L1"] - ib["STORM_SPEED_L2"]
    )
    ib["LAND_CHANGE_1"] = (
        ib["DIST2LAND_L1"] - ib["DIST2LAND_L2"]
    )

    # Previous-only rolling statistics.
    # Shift first, then roll. This is intentionally past-only.
    for source, prefix in [
        ("WMO_WIND", "WIND"),
        ("WMO_PRES", "PRESSURE"),
        ("STORM_SPEED", "SPEED"),
        ("DIST2LAND", "LAND"),
    ]:
        past = g[source].shift(1)
        past_group = past.groupby(ib["SID"], sort=False)
        for window in [2, 3, 4]:
            rolled = past_group.rolling(
                window=window,
                min_periods=1
            )
            ib[f"{prefix}_ROLL_MEAN_{window}"] = (
                rolled.mean().reset_index(level=0, drop=True)
            )
            ib[f"{prefix}_ROLL_STD_{window}"] = (
                rolled.std(ddof=0).reset_index(level=0, drop=True)
            )

    # Current position is an observation available to the system.
    # It is not the target and does not contain WMO_WIND/WMO_PRES.
    ib["LAT_RAD"] = np.deg2rad(ib["LAT"])
    ib["LON_RAD"] = np.deg2rad(ib["LON"])

    ib["MONTH_SIN"] = np.sin(2 * np.pi * ib["ISO_TIME"].dt.month / 12)
    ib["MONTH_COS"] = np.cos(2 * np.pi * ib["ISO_TIME"].dt.month / 12)

    ib["HOUR_SIN"] = np.sin(2 * np.pi * ib["ISO_TIME"].dt.hour / 24)
    ib["HOUR_COS"] = np.cos(2 * np.pi * ib["ISO_TIME"].dt.hour / 24)

    ib["DOY_SIN"] = np.sin(
        2 * np.pi * ib["ISO_TIME"].dt.dayofyear / 365.25
    )
    ib["DOY_COS"] = np.cos(
        2 * np.pi * ib["ISO_TIME"].dt.dayofyear / 365.25
    )

    ib["STORM_AGE_HOURS"] = (
        g["ISO_TIME"].transform(
            lambda x: (
                x - x.iloc[0]
            ).dt.total_seconds() / 3600.0
        )
    )

    # Previous motion vector.
    prev_dir = np.deg2rad(ib["STORM_DIR_L1"])
    ib["PREV_MOTION_U"] = (
        ib["STORM_SPEED_L1"] * np.sin(prev_dir)
    )
    ib["PREV_MOTION_V"] = (
        ib["STORM_SPEED_L1"] * np.cos(prev_dir)
    )

    return ib


TRACK_FEATURES = [
    # Current observed geometry
    "LAT", "LON", "DIST2LAND",

    # Previous intensity context
    "WMO_WIND_L1", "WMO_WIND_L2", "WMO_WIND_L3",
    "WMO_PRES_L1", "WMO_PRES_L2", "WMO_PRES_L3",

    # Previous movement
    "STORM_SPEED_L1", "STORM_SPEED_L2", "STORM_SPEED_L3",
    "STORM_DIR_L1", "STORM_DIR_L2", "STORM_DIR_L3",

    # Previous geography
    "LAT_L1", "LAT_L2", "LAT_L3",
    "LON_L1", "LON_L2", "LON_L3",
    "DIST2LAND_L1", "DIST2LAND_L2", "DIST2LAND_L3",

    # Past-only intensity trends
    "WIND_CHANGE_1", "WIND_CHANGE_2", "WIND_ACCELERATION",
    "PRESSURE_CHANGE_1", "PRESSURE_CHANGE_2",
    "PRESSURE_ACCELERATION",

    # Past-only motion/environmental geography trends
    "SPEED_CHANGE_1", "LAND_CHANGE_1",

    # Past-only rolling statistics
    "WIND_ROLL_MEAN_2", "WIND_ROLL_STD_2",
    "WIND_ROLL_MEAN_3", "WIND_ROLL_STD_3",
    "WIND_ROLL_MEAN_4", "WIND_ROLL_STD_4",

    "PRESSURE_ROLL_MEAN_2", "PRESSURE_ROLL_STD_2",
    "PRESSURE_ROLL_MEAN_3", "PRESSURE_ROLL_STD_3",
    "PRESSURE_ROLL_MEAN_4", "PRESSURE_ROLL_STD_4",

    "SPEED_ROLL_MEAN_2", "SPEED_ROLL_STD_2",
    "SPEED_ROLL_MEAN_3", "SPEED_ROLL_STD_3",
    "SPEED_ROLL_MEAN_4", "SPEED_ROLL_STD_4",

    "LAND_ROLL_MEAN_2", "LAND_ROLL_STD_2",
    "LAND_ROLL_MEAN_3", "LAND_ROLL_STD_3",
    "LAND_ROLL_MEAN_4", "LAND_ROLL_STD_4",

    # Temporal/geographic encoding
    "LAT_RAD", "LON_RAD",
    "MONTH_SIN", "MONTH_COS",
    "HOUR_SIN", "HOUR_COS",
    "DOY_SIN", "DOY_COS",
    "STORM_AGE_HOURS",
    "PREV_MOTION_U", "PREV_MOTION_V",
]

ERA5_FEATURES = [
    "U10", "V10", "ERA5_WIND_SPEED", "ERA5_WIND_DIR",
    "MSLP", "SST", "T2M", "D2M", "TP"
]

# ============================================================
# LOAD IBTRACS
# ============================================================

print()
print("=" * 78)
print(" CYCLONE-X FINAL MODEL 1 — ADAPTIVE MULTI-SOURCE VISION FUSION")
print("=" * 78)
print(f"Training years : {START_YEAR}-{END_YEAR}")
print(f"HURSAT         : {HURSAT_DIR}")
print(f"IBTrACS        : {IBTRACS_FILE}")
print(f"ERA5           : {ERA5_DIR}")
print(f"Output         : {MODEL_DIR}")
print("=" * 78)

if not HURSAT_DIR.exists():
    raise FileNotFoundError(f"HURSAT directory not found:\n{HURSAT_DIR}")

if not IBTRACS_FILE.exists():
    raise FileNotFoundError(f"IBTrACS file not found:\n{IBTRACS_FILE}")

print("\n[1/10] Loading IBTrACS...")

ib = pd.read_csv(
    IBTRACS_FILE,
    skiprows=[1],
    low_memory=False
)

ib["ISO_TIME"] = pd.to_datetime(
    ib["ISO_TIME"], errors="coerce"
)

for col in [
    "SEASON", "LAT", "LON", "WMO_WIND", "WMO_PRES",
    "STORM_SPEED", "STORM_DIR", "DIST2LAND"
]:
    if col in ib.columns:
        ib[col] = pd.to_numeric(
            ib[col], errors="coerce"
        )

ib = ib[
    (ib["SEASON"] >= START_YEAR) &
    (ib["SEASON"] <= END_YEAR)
].copy()

ib = ib.dropna(
    subset=["SID", "ISO_TIME", "LAT", "LON", "WMO_WIND"]
).copy()

ib["TARGET"] = ib["WMO_WIND"].apply(target_from_wind)

ib = ib.dropna(subset=["TARGET"]).copy()
ib["TARGET"] = ib["TARGET"].astype(int)

ib = build_previous_features(ib)

print(f"IBTrACS records: {len(ib):,}")
print(f"Cyclone events : {ib['SID'].nunique()}")

# ============================================================
# DISCOVER ERA5
# ============================================================

print("\n[2/10] Indexing ERA5 files...")

era5_files = discover_era5_files()
print(f"ERA5 storm groups found: {len(era5_files)}")

# ============================================================
# DISCOVER HURSAT
# ============================================================

print("\n[3/10] Indexing HURSAT files...")

all_nc = sorted(HURSAT_DIR.rglob("*.nc"))
print(f"HURSAT NetCDF files found: {len(all_nc):,}")

storm_lookup = {
    sid: g.sort_values("ISO_TIME").reset_index(drop=True)
    for sid, g in ib.groupby("SID")
}

# Cache ERA5 datasets. Only the small selected point is loaded.
era5_cache = {}

def get_era5_for_sample(sid, timestamp, lat, lon):
    files = era5_files.get(sid, [])

    if not files:
        return np.zeros(len(ERA5_FEATURES), dtype=np.float32), 0.0

    # Try likely files first by timestamp/year/month appearing in filename.
    stamp = timestamp.strftime("%Y_%m")
    ordered = sorted(
        files,
        key=lambda p: (
            0 if stamp in p.name else 1,
            0 if timestamp.strftime("%Y") in p.name else 1
        )
    )

    for path in ordered:
        key = str(path)

        try:
            if key not in era5_cache:
                era5_cache[key] = xr.open_dataset(
                    path,
                    decode_times=True
                )

            vals, valid = extract_era5_features(
                era5_cache[key],
                timestamp,
                lat,
                lon
            )

            if valid > 0:
                return vals, valid

        except Exception:
            continue

    return np.zeros(len(ERA5_FEATURES), dtype=np.float32), 0.0

# ============================================================
# BUILD FUSED DATASET
# ============================================================

print("\n[4/10] Matching HURSAT + IBTrACS + ERA5...")
print("This is the main data-fusion stage.")

images = []
track_rows = []
era_rows = []
quality_rows = []
targets = []
groups = []
manifest = []

storm_counts = {}
matched = 0
failed = 0
era5_available = 0

for i, nc_file in enumerate(all_nc, start=1):
    try:
        sid = nc_file.parent.name

        if sid not in storm_lookup:
            continue

        timestamp = extract_hursat_timestamp(nc_file.name)

        if timestamp is None:
            continue

        storm_df = storm_lookup[sid]

        row, diff_hours = nearest_row(
            storm_df, timestamp
        )

        if diff_hours > MATCH_HOURS:
            continue

        target = int(row["TARGET"])

        count = storm_counts.get(sid, 0)

        if count >= MAX_SAMPLES_PER_STORM:
            continue

        with xr.open_dataset(
            nc_file,
            decode_times=False
        ) as ds:

            raw = find_satellite_array(ds)

            if raw is None:
                failed += 1
                continue

            image = normalize_satellite(raw)

            if image is None:
                failed += 1
                continue

        sat_quality = source_quality(image)

        track = []
        for feature in TRACK_FEATURES:
            value = safe_float(row.get(feature, np.nan), np.nan)
            track.append(
                0.0 if not np.isfinite(value) else float(value)
            )

        track = np.asarray(track, dtype=np.float32)

        era_vals, era_valid = get_era5_for_sample(
            sid,
            timestamp,
            safe_float(row["LAT"], 0.0),
            safe_float(row["LON"], 0.0),
        )

        if REQUIRE_ERA5 and era_valid < 0.5:
            continue

        images.append(image)
        track_rows.append(track)
        era_rows.append(era_vals)
        quality_rows.append([
            sat_quality,
            float(era_valid),
        ])
        targets.append(target)
        groups.append(sid)

        manifest.append({
            "sid": sid,
            "timestamp": timestamp.isoformat(),
            "hursat_file": str(nc_file),
            "ibtracs_time": pd.Timestamp(row["ISO_TIME"]).isoformat(),
            "match_hours": diff_hours,
            "target": target,
            "target_name": CLASS_NAMES[target],
            "satellite_source": "HURSAT-B1 IRWIN/rad10",
            "satellite_quality": sat_quality,
            "era5_available": int(era_valid),
        })

        storm_counts[sid] = count + 1
        matched += 1

        if era_valid:
            era5_available += 1

    except Exception as exc:
        failed += 1
        continue

    if i % 500 == 0:
        print(
            f"  scanned={i:,} | matched={matched:,} | "
            f"ERA5={era5_available:,}"
        )

if len(images) < 100:
    raise RuntimeError(
        f"Only {len(images)} fused samples were built. "
        "Check HURSAT/IBTrACS matching and ERA5 files."
    )

X_img = np.asarray(images, dtype=np.float32)
X_track = np.asarray(track_rows, dtype=np.float32)
X_era = np.asarray(era_rows, dtype=np.float32)
X_quality = np.asarray(quality_rows, dtype=np.float32)
y = np.asarray(targets, dtype=np.int64)
groups = np.asarray(groups)

manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv(
    MODEL_DIR / "data_manifest_model1.csv",
    index=False
)

print()
print(f"Fused samples       : {len(y):,}")
print(f"Unique storms       : {len(np.unique(groups))}")
print(f"ERA5-backed samples : {era5_available:,}")
print(f"ERA5 coverage       : {100*era5_available/len(y):.2f}%")
print(f"Failed files        : {failed:,}")

# ============================================================
# STORM-WISE SPLIT
# ============================================================

print("\n[5/10] Creating event-wise train/validation/test split...")

gss1 = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

trainval_idx, test_idx = next(
    gss1.split(X_img, y, groups=groups)
)

trainval_groups = groups[trainval_idx]

gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE + 1
)

train_rel, val_rel = next(
    gss2.split(
        X_img[trainval_idx],
        y[trainval_idx],
        groups=trainval_groups
    )
)

train_idx = trainval_idx[train_rel]
val_idx = trainval_idx[val_rel]

def subset(arr, idx):
    return arr[idx]

Ximg_tr = subset(X_img, train_idx)
Ximg_va = subset(X_img, val_idx)
Ximg_te = subset(X_img, test_idx)

Xtrk_tr = subset(X_track, train_idx)
Xtrk_va = subset(X_track, val_idx)
Xtrk_te = subset(X_track, test_idx)

Xera_tr = subset(X_era, train_idx)
Xera_va = subset(X_era, val_idx)
Xera_te = subset(X_era, test_idx)

Xq_tr = subset(X_quality, train_idx)
Xq_va = subset(X_quality, val_idx)
Xq_te = subset(X_quality, test_idx)

y_tr = subset(y, train_idx)
y_va = subset(y, val_idx)
y_te = subset(y, test_idx)

groups_tr = groups[train_idx]
groups_va = groups[val_idx]
groups_te = groups[test_idx]

# Hard leakage audit: no cyclone SID may appear in more than one split.
train_sids = set(groups_tr.tolist())
val_sids = set(groups_va.tolist())
test_sids = set(groups_te.tolist())

assert not (train_sids & val_sids), "LEAKAGE: train/validation storm overlap detected."
assert not (train_sids & test_sids), "LEAKAGE: train/test storm overlap detected."
assert not (val_sids & test_sids), "LEAKAGE: validation/test storm overlap detected."

# Feature-name audit: current target variables must never enter the numeric branch.
for forbidden_feature in ["WMO_WIND", "WMO_PRES"]:
    if any(forbidden_feature == f for f in TRACK_FEATURES):
        raise RuntimeError(f"LEAKAGE: forbidden current target feature detected: {forbidden_feature}")

print(f"Train samples : {len(y_tr):,} | storms={len(np.unique(groups_tr))}")
print(f"Val samples   : {len(y_va):,} | storms={len(np.unique(groups_va))}")
print(f"Test samples  : {len(y_te):,} | storms={len(np.unique(groups_te))}")
print("Class distribution:")
print("  train:", np.bincount(y_tr, minlength=3).tolist())
print("  val  :", np.bincount(y_va, minlength=3).tolist())
print("  test :", np.bincount(y_te, minlength=3).tolist())
if len(np.unique(y_te)) < 3:
    print("WARNING: The fixed unseen-storm test split does not contain all 3 classes.")

# Save split information.
split_df = manifest_df.copy()
split_df["split"] = "unused"
split_df.loc[train_idx, "split"] = "train"
split_df.loc[val_idx, "split"] = "validation"
split_df.loc[test_idx, "split"] = "test"
split_df.to_csv(
    MODEL_DIR / "data_manifest_model1.csv",
    index=False
)

pd.DataFrame({"SID": sorted(train_sids)}).to_csv(MODEL_DIR / "train_storms.csv", index=False)
pd.DataFrame({"SID": sorted(val_sids)}).to_csv(MODEL_DIR / "validation_storms.csv", index=False)
pd.DataFrame({"SID": sorted(test_sids)}).to_csv(MODEL_DIR / "test_storms.csv", index=False)

with open(MODEL_DIR / "leakage_audit.json", "w", encoding="utf-8") as f:
    json.dump({
        "status": "PASS",
        "train_validation_sid_overlap": sorted(train_sids & val_sids),
        "train_test_sid_overlap": sorted(train_sids & test_sids),
        "validation_test_sid_overlap": sorted(val_sids & test_sids),
        "current_WMO_WIND_input": False,
        "current_WMO_PRES_input": False,
        "current_intensity_trend_input": False,
        "ARCHER_intensity_or_pressure_input": False,
        "preprocessing_fit_on_train_only": True,
        "test_used_for_training": False
    }, f, indent=2)

# ============================================================
# SCALE NUMERIC SOURCES — TRAIN ONLY
# ============================================================

print("\n[6/10] Scaling numeric fusion sources using TRAIN ONLY...")

track_imputer = SimpleImputer(strategy="median")
era_imputer = SimpleImputer(strategy="median")
track_scaler = StandardScaler()
era_scaler = StandardScaler()

# Fit imputers and scalers on TRAIN ONLY. Validation/test never influence preprocessing.
Xtrk_tr_i = track_imputer.fit_transform(Xtrk_tr)
Xtrk_va_i = track_imputer.transform(Xtrk_va)
Xtrk_te_i = track_imputer.transform(Xtrk_te)

Xera_tr_i = era_imputer.fit_transform(Xera_tr)
Xera_va_i = era_imputer.transform(Xera_va)
Xera_te_i = era_imputer.transform(Xera_te)

Xtrk_tr_s = track_scaler.fit_transform(Xtrk_tr_i)
Xtrk_va_s = track_scaler.transform(Xtrk_va_i)
Xtrk_te_s = track_scaler.transform(Xtrk_te_i)

Xera_tr_s = era_scaler.fit_transform(Xera_tr_i)
Xera_va_s = era_scaler.transform(Xera_va_i)
Xera_te_s = era_scaler.transform(Xera_te_i)

joblib.dump(
    track_imputer,
    MODEL_DIR / "track_imputer_model1.joblib"
)
joblib.dump(
    era_imputer,
    MODEL_DIR / "era5_imputer_model1.joblib"
)
joblib.dump(
    track_scaler,
    MODEL_DIR / "track_scaler_model1.joblib"
)
joblib.dump(
    era_scaler,
    MODEL_DIR / "era5_scaler_model1.joblib"
)

# ============================================================
# BUILD ADAPTIVE MULTI-SOURCE FUSION MODEL
# ============================================================

print("\n[7/10] Building adaptive satellite + track + ERA5 fusion model...")

sat_input = Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
    name="satellite_image"
)

x = Conv2D(
    32, 3, padding="same",
    activation="relu",
    name="sat_conv1"
)(sat_input)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(
    64, 3, padding="same",
    activation="relu",
    name="sat_conv2"
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(
    96, 3, padding="same",
    activation="relu",
    name="sat_conv3"
)(x)
x = BatchNormalization()(x)
sat_embed = GlobalAveragePooling2D(
    name="satellite_embedding"
)(x)

sat_embed = Dense(
    64, activation="relu",
    name="satellite_projection"
)(sat_embed)
sat_embed = Dropout(0.20)(sat_embed)

track_input = Input(
    shape=(len(TRACK_FEATURES),),
    name="historical_track"
)

t = Dense(
    64, activation="relu",
    name="track_dense1"
)(track_input)
t = Dropout(0.15)(t)
track_embed = Dense(
    64, activation="relu",
    name="track_embedding"
)(t)

era_input = Input(
    shape=(len(ERA5_FEATURES),),
    name="era5_environment"
)

e = Dense(
    48, activation="relu",
    name="era_dense1"
)(era_input)
e = Dropout(0.15)(e)
era_embed = Dense(
    64, activation="relu",
    name="era_embedding"
)(e)

quality_input = Input(
    shape=(2,),
    name="source_quality"
)

# Reliability-aware gate:
# [satellite_quality, era5_available]
# plus source embeddings -> source weights.
gate_input = Concatenate(name="fusion_gate_input")(
    [sat_embed, track_embed, era_embed, quality_input]
)

gate = Dense(
    32,
    activation="relu",
    name="fusion_gate_dense"
)(gate_input)

gate = Dense(
    3,
    activation="softmax",
    name="adaptive_source_weights"
)(gate)

# Make each source a 64-d vector and apply its learned weight.
w_sat = gate[:, 0:1]
w_track = gate[:, 1:2]
w_era = gate[:, 2:3]

fused = Add(name="adaptive_fusion")([
    Multiply()([sat_embed, w_sat]),
    Multiply()([track_embed, w_track]),
    Multiply()([era_embed, w_era]),
])

fused = Dense(
    96, activation="relu",
    name="fusion_dense"
)(fused)
fused = Dropout(0.30)(fused)

fused = Dense(
    48, activation="relu",
    name="fusion_representation"
)(fused)

output = Dense(
    3,
    activation="softmax",
    name="cyclone_stage"
)(fused)

model = Model(
    inputs=[
        sat_input,
        track_input,
        era_input,
        quality_input
    ],
    outputs=output,
    name="CYCLONE_X_Adaptive_MultiSource_Fusion"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

with open(
    MODEL_DIR / "model_architecture_model1.json",
    "w",
    encoding="utf-8"
) as f:
    f.write(model.to_json())

with open(
    MODEL_DIR / "model_summary_model1.txt",
    "w",
    encoding="utf-8"
) as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_tr,
    minlength=3
)

total = float(len(y_tr))
class_weight = {}

for c in range(3):
    if class_counts[c] > 0:
        class_weight[c] = total / (
            3.0 * float(class_counts[c])
        )

print("\nClass counts:", class_counts.tolist())
print("Class weights:", class_weight)

# ============================================================
# TRAIN
# ============================================================

print("\n[8/10] Training Model 1...")

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_DIR / "cyclonex_model1_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
]

history = model.fit(
    {
        "satellite_image": Ximg_tr,
        "historical_track": Xtrk_tr_s,
        "era5_environment": Xera_tr_s,
        "source_quality": Xq_tr,
    },
    to_categorical(y_tr, 3),
    validation_data=(
        {
            "satellite_image": Ximg_va,
            "historical_track": Xtrk_va_s,
            "era5_environment": Xera_va_s,
            "source_quality": Xq_va,
        },
        to_categorical(y_va, 3)
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

model.save(
    MODEL_DIR / "cyclonex_model1_final.keras"
)

# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CYCLONE-X Model 1 — Training / Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(
    MODEL_DIR / "training_accuracy_model1.png",
    dpi=180
)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CYCLONE-X Model 1 — Training / Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    MODEL_DIR / "training_loss_model1.png",
    dpi=180
)
plt.close()

# ============================================================
# FINAL UNSEEN TEST
# ============================================================

print("\n[9/10] Evaluating completely unseen cyclone storms...")

probs = model.predict(
    {
        "satellite_image": Ximg_te,
        "historical_track": Xtrk_te_s,
        "era5_environment": Xera_te_s,
        "source_quality": Xq_te,
    },
    batch_size=BATCH_SIZE,
    verbose=0
)

pred = np.argmax(probs, axis=1)

accuracy = accuracy_score(y_te, pred)
precision = precision_score(
    y_te, pred,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_te, pred,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_te, pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    y_te,
    pred,
    labels=[0, 1, 2]
)

report_dict = classification_report(
    y_te,
    pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

print()
print("=" * 78)
print(" FINAL UNSEEN-STORM TEST")
print("=" * 78)
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1       : {f1 * 100:.2f}%")
print()
print(classification_report(
    y_te,
    pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    zero_division=0
))
print("Confusion matrix:")
print(cm)
print("=" * 78)

# ============================================================
# CONFUSION MATRIX PNG
# ============================================================

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("CYCLONE-X Model 1 — Unseen-Storm Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1, 2], ["Weak", "Cyclonic", "Severe"])
plt.yticks([0, 1, 2], ["Weak", "Cyclonic", "Severe"])

threshold = cm.max() / 2.0 if cm.size else 0

for i in range(3):
    for j in range(3):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.tight_layout()
plt.savefig(
    MODEL_DIR / "confusion_matrix_model1.png",
    dpi=200
)
plt.close()

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
).to_csv(
    MODEL_DIR / "confusion_matrix_model1.csv"
)

# ============================================================
# TEST PREDICTIONS
# ============================================================

test_pred_df = manifest_df.iloc[test_idx].copy()
test_pred_df["predicted_class"] = pred
test_pred_df["predicted_name"] = [
    CLASS_NAMES[int(x)] for x in pred
]

for c in range(3):
    test_pred_df[f"prob_class_{c}"] = probs[:, c]

test_pred_df.to_csv(
    MODEL_DIR / "test_predictions_model1.csv",
    index=False
)

# ============================================================
# GRAD-CAM EXPLAINABILITY
# ============================================================

print("[10/10] Creating explainability artifact...")

def make_gradcam(
    sample_index,
    save_path
):
    sample_img = Ximg_te[sample_index:sample_index+1]

    inputs = {
        "satellite_image": sample_img,
        "historical_track": Xtrk_te_s[sample_index:sample_index+1],
        "era5_environment": Xera_te_s[sample_index:sample_index+1],
        "source_quality": Xq_te[sample_index:sample_index+1],
    }

    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer("sat_conv3").output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        class_id = tf.argmax(predictions[0])
        class_score = predictions[:, class_id]

    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    conv = conv_outputs[0]
    weights = pooled_grads[0]

    cam = tf.reduce_sum(
        conv * weights[tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    cam = tf.maximum(cam, 0)
    cam = cam / (
        tf.reduce_max(cam) + 1e-8
    )

    heat = cam.numpy()
    heat_img = Image.fromarray(
        np.uint8(heat * 255)
    ).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    heat = np.asarray(heat_img) / 255.0
    original = sample_img[0, ..., 0]

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original, cmap="gray")
    ax1.set_title("HURSAT Satellite Input")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(original, cmap="gray")
    ax2.imshow(heat, alpha=0.45, cmap="jet")
    ax2.set_title(
        f"Grad-CAM — {CLASS_NAMES[int(class_id)]}"
    )
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200
    )
    plt.close()

if len(Ximg_te) > 0:
    make_gradcam(
        0,
        MODEL_DIR / "gradcam_test_example_model1.png"
    )

# ============================================================
# MODEL CARD / JSON
# ============================================================

source_coverage = {
    "HURSAT_satellite": True,
    "IBTrACS_track": True,
    "ERA5_environment": bool(era5_available > 0),
    "ERA5_coverage_percent": float(
        100.0 * era5_available / len(y)
    ),
}

leakage_controls = {
    "current_WMO_WIND_used_as_feature": False,
    "current_WMO_PRES_used_as_feature": False,
    "current_intensity_change_used_as_feature": False,
    "current_target_derived_feature_used": False,
    "satellite_archer_intensity_used": False,
    "storm_wise_test_split": True,
    "test_storms_seen_during_training": False,
    "imputers_fit_on_training_only": True,
    "scalers_fit_on_training_only": True,
    "past_trends_use_only_T_minus_1_T_minus_2_T_minus_3": True,
    "era5_environment_is_target_free": True,
}

feature_schema = {
    "satellite_branch": {
        "source": "HURSAT-B1",
        "representation": "64x64 single-channel normalized infrared image",
        "vision_model": "lightweight CNN",
        "notes": "Current satellite image is an input observation; target wind is never used to construct the image."
    },
    "track_branch": {
        "source": "IBTrACS",
        "feature_count": len(TRACK_FEATURES),
        "features": TRACK_FEATURES,
        "notes": "Current position is allowed; intensity history and trends are strictly lagged."
    },
    "environment_branch": {
        "source": "ERA5",
        "feature_count": len(ERA5_FEATURES),
        "features": ERA5_FEATURES,
        "missing_source_handling": "validity mask + adaptive source gate"
    },
    "fusion": {
        "type": "adaptive reliability-aware three-source fusion",
        "sources": ["satellite", "historical_track", "ERA5_environment"],
        "gate_inputs": [
            "satellite_quality",
            "era5_available",
            "source_embeddings"
        ]
    },
    "target": {
        "source": "IBTrACS WMO_WIND",
        "used_as_input": False,
        "classes": {
            "0": CLASS_NAMES[0],
            "1": CLASS_NAMES[1],
            "2": CLASS_NAMES[2]
        }
    }
}

with open(
    MODEL_DIR / "feature_schema_model1.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(feature_schema, f, indent=4)

with open(
    MODEL_DIR / "label_map_model1.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "0": CLASS_NAMES[0],
            "1": CLASS_NAMES[1],
            "2": CLASS_NAMES[2]
        },
        f,
        indent=4
    )

final_report = {
    "project": "CYCLONE-X",
    "model_name": "CYCLONE-X Final Leakage-Free Adaptive Multi-Source Vision Fusion",
    "training_period": f"{START_YEAR}-{END_YEAR}",
    "task": "Cyclone stage classification",
    "data_sources": source_coverage,
    "sample_count": int(len(y)),
    "storm_count": int(len(np.unique(groups))),
    "train_samples": int(len(y_tr)),
    "validation_samples": int(len(y_va)),
    "unseen_test_samples": int(len(y_te)),
    "train_storms": int(len(np.unique(groups_tr))),
    "validation_storms": int(len(np.unique(groups_va))),
    "unseen_test_storms": int(len(np.unique(groups_te))),
    "metrics": {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    },
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "leakage_control": leakage_controls,
    "architecture": {
        "satellite": "CNN",
        "historical_track": "MLP",
        "ERA5": "MLP",
        "fusion": "adaptive reliability-aware gated fusion",
        "output": "3-class softmax"
    },
    "leakage_audit": "PASS — event-wise SID separation and train-only preprocessing checks executed before training.",
    "limitations": [
        "This artifact trains current cyclone-stage classification; it does not by itself prove 6–48 h track forecasting.",
        "Forecasting requires a separate temporally supervised model using future track targets.",
        "ERA5 contribution depends on actual ERA5 coverage in the local dataset.",
        "HURSAT is historical storm-centric data; live operation requires an official live satellite source."
    ]
}

with open(
    MODEL_DIR / "cyclonex_model1_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        final_report,
        f,
        indent=4
    )

# Comparison CSV with the single final fusion model.
pd.DataFrame([{
    "model": "CYCLONE-X Adaptive Multi-Source Vision Fusion",
    "accuracy": accuracy,
    "precision_weighted": precision,
    "recall_weighted": recall,
    "f1_weighted": f1,
    "unseen_storm_test": True,
}]).to_csv(
    MODEL_DIR / "model_metrics_model1.csv",
    index=False
)

# ============================================================
# REQUIREMENTS SNAPSHOT
# ============================================================

requirements = """tensorflow
numpy
pandas
xarray
netCDF4
scikit-learn
joblib
matplotlib
pillow
"""

(MODEL_DIR / "requirements_model1.txt").write_text(
    requirements,
    encoding="utf-8"
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print(" CYCLONE-X MODEL 1 COMPLETE")
print("=" * 78)
print(f"UNSEEN-STORM ACCURACY : {accuracy * 100:.2f}%")
print(f"UNSEEN-STORM F1       : {f1 * 100:.2f}%")
print(f"Fused samples         : {len(y):,}")
print(f"Test storms held out  : {len(np.unique(groups_te))}")
print(f"ERA5 coverage         : {100*era5_available/len(y):.2f}%")
print()
print("Generated directory:")
print(MODEL_DIR)
print()
print("Important:")
print("  Current WMO_WIND/WMO_PRES are NOT model inputs.")
print("  Current intensity-change features are NOT model inputs.")
print("  Test cyclone SIDs are completely unseen.")
print("  HURSAT satellite + IBTrACS history + ERA5 environment are fused.")
print("  Missing ERA5 is exposed through the source-validity mask.")
print("  HURSAT ARCHER-derived intensity/pressure variables are excluded.")
print()
print("Key artifacts:")
for name in [
    "cyclonex_model1_final.keras",
    "cyclonex_model1_best.keras",
    "model_architecture_model1.json",
    "cyclonex_model1_report.json",
    "feature_schema_model1.json",
    "label_map_model1.json",
    "model_metrics_model1.csv",
    "confusion_matrix_model1.csv",
    "confusion_matrix_model1.png",
    "training_accuracy_model1.png",
    "training_loss_model1.png",
    "gradcam_test_example_model1.png",
    "test_predictions_model1.csv",
    "data_manifest_model1.csv",
    "track_imputer_model1.joblib",
    "era5_imputer_model1.joblib",
    "track_scaler_model1.joblib",
    "era5_scaler_model1.joblib",
    "requirements_model1.txt",
]:
    print("  ", MODEL_DIR / name)

print("=" * 78)

"""
CYCLONE-X MODEL 2 — LEAKAGE-FREE 6/12/24/48 H FORECASTER

Purpose:
  Predict future cyclone position (lat/lon) and WMO wind from information
  available at the forecast origin time only.

Data:
  E:\Cyclone-x-prototype\datas\IBTrACS.NI.v04r01.csv

Validation:
  Whole cyclone SID is kept in exactly one split. Future target values are
  never used as features. Four direct horizon models are trained.

Outputs:
  E:\Cyclone-x-prototype\model_forecast\*.joblib
  forecast_report.json
  forecast_test_predictions.csv
"""
from pathlib import Path
import json
import math
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

BASE = Path(r"E:\Cyclone-x-prototype")
DATA = BASE / "datas"
IBTRACS = DATA / "IBTrACS.NI.v04r01.csv"
OUT = BASE / "model_forecast"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = {6: 3, 12: 6, 24: 12, 48: 24}  # 3-hourly IBTrACS steps
RANDOM_STATE = 42

FEATURES = [
    "LAT", "LON", "WMO_WIND", "WMO_PRES",
    "STORM_SPEED", "STORM_DIR", "DIST2LAND",
    "LAT_L1", "LAT_L2", "LAT_L3",
    "LON_L1", "LON_L2", "LON_L3",
    "WIND_L1", "WIND_L2", "WIND_L3",
    "PRES_L1", "PRES_L2", "PRES_L3",
    "SPEED_L1", "SPEED_L2", "SPEED_L3",
    "DIR_L1", "DIR_L2", "DIR_L3",
    "WIND_CHANGE_1", "WIND_CHANGE_2",
    "PRESSURE_CHANGE_1", "PRESSURE_CHANGE_2",
    "SPEED_CHANGE_1", "SPEED_CHANGE_2",
    "LAT_VEL", "LON_VEL",
    "MONTH_SIN", "MONTH_COS", "HOUR_SIN", "HOUR_COS",
    "STORM_AGE_HOURS",
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2-lat1)
    dl = np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_ibtracs():
    if not IBTRACS.exists():
        raise FileNotFoundError(f"Missing {IBTRACS}")
    df = pd.read_csv(IBTRACS, skiprows=[1], low_memory=False)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for c in ["LAT","LON","WMO_WIND","WMO_PRES","STORM_SPEED","STORM_DIR","DIST2LAND"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["SID","ISO_TIME","LAT","LON","WMO_WIND"]).copy()
    df = df.sort_values(["SID","ISO_TIME"]).reset_index(drop=True)
    return df


def add_features(df):
    g = df.groupby("SID", group_keys=False)
    for src, dst in [("LAT","LAT"),("LON","LON"),("WMO_WIND","WIND"),("WMO_PRES","PRES"),
                     ("STORM_SPEED","SPEED"),("STORM_DIR","DIR")]:
        for lag in (1,2,3):
            df[f"{dst}_L{lag}"] = g[src].shift(lag)
    df["WIND_CHANGE_1"] = df["WIND_L1"] - df["WIND_L2"]
    df["WIND_CHANGE_2"] = df["WIND_L2"] - df["WIND_L3"]
    df["PRESSURE_CHANGE_1"] = df["PRES_L1"] - df["PRES_L2"]
    df["PRESSURE_CHANGE_2"] = df["PRES_L2"] - df["PRES_L3"]
    df["SPEED_CHANGE_1"] = df["SPEED_L1"] - df["SPEED_L2"]
    df["SPEED_CHANGE_2"] = df["SPEED_L2"] - df["SPEED_L3"]
    # degrees per 3h -> approximate directional displacement features
    rad = np.radians(df["DIR_L1"])
    speed = df["SPEED_L1"]
    df["LAT_VEL"] = speed * np.cos(rad)
    df["LON_VEL"] = speed * np.sin(rad)
    df["MONTH_SIN"] = np.sin(2*np.pi*df["ISO_TIME"].dt.month/12)
    df["MONTH_COS"] = np.cos(2*np.pi*df["ISO_TIME"].dt.month/12)
    df["HOUR_SIN"] = np.sin(2*np.pi*df["ISO_TIME"].dt.hour/24)
    df["HOUR_COS"] = np.cos(2*np.pi*df["ISO_TIME"].dt.hour/24)
    df["STORM_AGE_HOURS"] = g["ISO_TIME"].transform(lambda x: (x-x.iloc[0]).dt.total_seconds()/3600)
    return df


def make_horizon_frame(df, steps):
    future = df.groupby("SID", group_keys=False)[["LAT","LON","WMO_WIND"]].shift(-steps)
    out = df.copy()
    out["Y_LAT"] = future["LAT"]
    out["Y_LON"] = future["LON"]
    out["Y_WIND"] = future["WMO_WIND"]
    out["TARGET_DT"] = df.groupby("SID")["ISO_TIME"].shift(-steps)
    out = out.dropna(subset=FEATURES+["Y_LAT","Y_LON","Y_WIND"]).copy()
    return out


def split_sids(df):
    sids = df["SID"].dropna().unique()
    gss = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(sids, groups=sids))
    train_sids, test_sids = set(sids[train_idx]), set(sids[test_idx])
    gss2 = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=RANDOM_STATE+1)
    tr2, va2 = next(gss2.split(np.array(sorted(train_sids)), groups=np.array(sorted(train_sids))))
    arr = np.array(sorted(train_sids))
    return set(arr[tr2]), set(arr[va2]), test_sids


def main():
    print("="*78)
    print(" CYCLONE-X MODEL 2 — LEAKAGE-FREE FORECASTING")
    print("="*78)
    print(f"IBTrACS: {IBTRACS}")
    df = add_features(load_ibtracs())
    train_sids, val_sids, test_sids = split_sids(df)
    print(f"Events total={df.SID.nunique()} train={len(train_sids)} val={len(val_sids)} test={len(test_sids)}")
    assert not train_sids & val_sids and not train_sids & test_sids and not val_sids & test_sids

    report = {
        "project":"CYCLONE-X",
        "model":"Model 2 — direct multi-horizon track + intensity forecaster",
        "horizons_hours":list(HORIZONS),
        "target":"future latitude, longitude and WMO wind",
        "leakage_control":"whole-cyclone SID split; only T and earlier features are used",
        "splits":{"train_storms":len(train_sids),"validation_storms":len(val_sids),"test_storms":len(test_sids)},
        "metrics":{}
    }
    all_test_predictions=[]

    for h, steps in HORIZONS.items():
        print(f"\n[HORIZON +{h}h] building supervised data...")
        d = make_horizon_frame(df, steps)
        tr = d[d.SID.isin(train_sids)].copy()
        va = d[d.SID.isin(val_sids)].copy()
        te = d[d.SID.isin(test_sids)].copy()
        print(f"samples train={len(tr):,} val={len(va):,} test={len(te):,}")
        if len(tr) < 100 or len(te) < 10:
            print("Skipping: insufficient samples")
            continue

        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(imp.fit_transform(tr[FEATURES]))
        Xva = scaler.transform(imp.transform(va[FEATURES]))
        Xte = scaler.transform(imp.transform(te[FEATURES]))

        # Multi-output target is relative position + future wind.
        # Relative deltas improve geographic generalization.
        ytr = np.column_stack([tr.Y_LAT-tr.LAT, tr.Y_LON-tr.LON, tr.Y_WIND])
        yva = np.column_stack([va.Y_LAT-va.LAT, va.Y_LON-va.LON, va.Y_WIND])
        yte = np.column_stack([te.Y_LAT-te.LAT, te.Y_LON-te.LON, te.Y_WIND])

        model = ExtraTreesRegressor(
            n_estimators=350, max_depth=22, min_samples_leaf=2,
            max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(Xtr, ytr)
        # Validation residuals provide a held-out uncertainty scale.
        pva = model.predict(Xva)
        val_pred_lat = va.LAT.to_numpy() + pva[:,0]
        val_pred_lon = va.LON.to_numpy() + pva[:,1]
        val_geo = haversine_km(va.Y_LAT.to_numpy(), va.Y_LON.to_numpy(), val_pred_lat, val_pred_lon)
        uncertainty_km_90 = float(np.percentile(val_geo, 90)) if len(val_geo) else 150.0

        pred = model.predict(Xte)

        pred_lat = te.LAT.to_numpy() + pred[:,0]
        pred_lon = te.LON.to_numpy() + pred[:,1]
        pred_wind = np.maximum(0, te.WMO_WIND.to_numpy() + pred[:,2])
        true_lat, true_lon, true_wind = te.Y_LAT.to_numpy(), te.Y_LON.to_numpy(), te.Y_WIND.to_numpy()
        geo = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
        wind_mae = mean_absolute_error(true_wind, pred_wind)
        wind_rmse = math.sqrt(mean_squared_error(true_wind, pred_wind))

        report["metrics"][str(h)] = {
            "test_samples":int(len(te)),
            "track_mae_km":float(np.mean(geo)),
            "track_median_km":float(np.median(geo)),
            "track_rmse_km":float(np.sqrt(np.mean(geo**2))),
            "intensity_mae":float(wind_mae),
            "intensity_rmse":float(wind_rmse),
            "validation_track_p90_km": uncertainty_km_90,
        }
        print(f"Track MAE: {np.mean(geo):.1f} km | Intensity MAE: {wind_mae:.2f}")

        joblib.dump(model, OUT/f"forecast_model_{h}h.joblib")
        joblib.dump(imp, OUT/f"forecast_imputer_{h}h.joblib")
        joblib.dump(scaler, OUT/f"forecast_scaler_{h}h.joblib")
        joblib.dump(FEATURES, OUT/f"forecast_features_{h}h.joblib")

        tmp = te[["SID","ISO_TIME","LAT","LON","WMO_WIND"]].copy()
        tmp["horizon_h"] = h
        tmp["pred_lat"] = pred_lat
        tmp["pred_lon"] = pred_lon
        tmp["pred_wind"] = pred_wind
        tmp["actual_lat"] = true_lat
        tmp["actual_lon"] = true_lon
        tmp["actual_wind"] = true_wind
        tmp["track_error_km"] = geo
        all_test_predictions.append(tmp)

    if all_test_predictions:
        pd.concat(all_test_predictions, ignore_index=True).to_csv(OUT/"forecast_test_predictions.csv", index=False)
    with open(OUT/"forecast_report.json","w",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    with open(OUT/"leakage_audit_forecast.json","w",encoding="utf-8") as f:
        json.dump({
            "status":"PASS",
            "whole_storm_split":True,
            "train_validation_overlap":sorted(train_sids & val_sids),
            "train_test_overlap":sorted(train_sids & test_sids),
            "validation_test_overlap":sorted(val_sids & test_sids),
            "future_targets_used_as_features":False,
            "preprocessing_fit_on_train_only":True,
        },f,indent=2)
    print("\nDONE — forecast models saved to:", OUT)

if __name__ == "__main__":
    main()

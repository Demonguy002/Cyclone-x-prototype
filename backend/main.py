
import os, base64, io, re, json
import urllib.parse, urllib.request
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

BASE = Path(os.getenv("CYCLONEX_BASE", r"E:\Cyclone-x-prototype"))
MODEL_DIR = BASE / "model1"
DATA_DIR = BASE / "datas"
HURSAT_DIR = DATA_DIR / "HURSAT_NI"
IBTRACS_FILE = DATA_DIR / "IBTrACS.NI.v04r01.csv"
MANIFEST_FILE = MODEL_DIR / "data_manifest_model1.csv"

MODEL_BEST = MODEL_DIR / "cyclonex_model1_best.keras"
MODEL_FINAL = MODEL_DIR / "cyclonex_model1_final.keras"

CLASS_NAMES = [
    "Depression / Weak System",
    "Cyclonic Storm",
    "Severe Cyclone",
]
IMAGE_SIZE = 64
MATCH_HOURS = 4.0

# Exact order used by the trained Model 1 track branch.
TRACK_FEATURES = [
    "LAT","LON","DIST2LAND",
    "WMO_WIND_L1","WMO_WIND_L2","WMO_WIND_L3",
    "WMO_PRES_L1","WMO_PRES_L2","WMO_PRES_L3",
    "STORM_SPEED_L1","STORM_SPEED_L2","STORM_SPEED_L3",
    "STORM_DIR_L1","STORM_DIR_L2","STORM_DIR_L3",
    "LAT_L1","LAT_L2","LAT_L3","LON_L1","LON_L2","LON_L3",
    "DIST2LAND_L1","DIST2LAND_L2","DIST2LAND_L3",
    "WIND_CHANGE_1","WIND_CHANGE_2","WIND_ACCELERATION",
    "PRESSURE_CHANGE_1","PRESSURE_CHANGE_2","PRESSURE_ACCELERATION",
    "SPEED_CHANGE_1","LAND_CHANGE_1",
    "WIND_ROLL_MEAN_2","WIND_ROLL_STD_2",
    "WIND_ROLL_MEAN_3","WIND_ROLL_STD_3",
    "WIND_ROLL_MEAN_4","WIND_ROLL_STD_4",
    "PRESSURE_ROLL_MEAN_2","PRESSURE_ROLL_STD_2",
    "PRESSURE_ROLL_MEAN_3","PRESSURE_ROLL_STD_3",
    "PRESSURE_ROLL_MEAN_4","PRESSURE_ROLL_STD_4",
    "SPEED_ROLL_MEAN_2","SPEED_ROLL_STD_2",
    "SPEED_ROLL_MEAN_3","SPEED_ROLL_STD_3",
    "SPEED_ROLL_MEAN_4","SPEED_ROLL_STD_4",
    "LAND_ROLL_MEAN_2","LAND_ROLL_STD_2",
    "LAND_ROLL_MEAN_3","LAND_ROLL_STD_3",
    "LAND_ROLL_MEAN_4","LAND_ROLL_STD_4",
    "LAT_RAD","LON_RAD","MONTH_SIN","MONTH_COS",
    "HOUR_SIN","HOUR_COS","DOY_SIN","DOY_COS",
    "STORM_AGE_HOURS","PREV_MOTION_U","PREV_MOTION_V",
]

REGIONS = [
    {"id":"kochi","name":"Kochi Coast","state":"Kerala","basin":"Arabian Sea","lat":9.9312,"lon":76.2673},
    {"id":"mangaluru","name":"Mangaluru Coast","state":"Karnataka","basin":"Arabian Sea","lat":12.9141,"lon":74.8560},
    {"id":"goa","name":"Goa Coast","state":"Goa","basin":"Arabian Sea","lat":15.4909,"lon":73.8278},
    {"id":"mumbai","name":"Mumbai Coast","state":"Maharashtra","basin":"Arabian Sea","lat":19.0760,"lon":72.8777},
    {"id":"konkan","name":"Konkan Coast","state":"Maharashtra","basin":"Arabian Sea","lat":16.9902,"lon":73.3120},
    {"id":"gujarat","name":"Gujarat Coast","state":"Gujarat","basin":"Arabian Sea","lat":21.6417,"lon":69.6293},
    {"id":"kutch","name":"Kutch Coast","state":"Gujarat","basin":"Arabian Sea","lat":23.7337,"lon":69.8597},
    {"id":"lakshadweep","name":"Lakshadweep","state":"Lakshadweep","basin":"Arabian Sea","lat":10.5667,"lon":72.6417},
    {"id":"south_arabian","name":"South Arabian Sea","state":"—","basin":"Arabian Sea","lat":12.0,"lon":65.0},
    {"id":"north_arabian","name":"North Arabian Sea","state":"—","basin":"Arabian Sea","lat":20.0,"lon":64.0},
    {"id":"chennai","name":"Chennai Coast","state":"Tamil Nadu","basin":"Bay of Bengal","lat":13.0827,"lon":80.2707},
    {"id":"puducherry","name":"Puducherry Coast","state":"Puducherry","basin":"Bay of Bengal","lat":11.9416,"lon":79.8083},
    {"id":"andhra","name":"Andhra Coast","state":"Andhra Pradesh","basin":"Bay of Bengal","lat":16.0,"lon":81.0},
    {"id":"visakhapatnam","name":"Visakhapatnam Coast","state":"Andhra Pradesh","basin":"Bay of Bengal","lat":17.6868,"lon":83.2185},
    {"id":"paradip","name":"Paradip Coast","state":"Odisha","basin":"Bay of Bengal","lat":20.3167,"lon":86.6083},
    {"id":"odisha","name":"Odisha Coast","state":"Odisha","basin":"Bay of Bengal","lat":19.8135,"lon":85.8312},
    {"id":"west_bengal","name":"West Bengal Coast","state":"West Bengal","basin":"Bay of Bengal","lat":21.9497,"lon":87.7479},
    {"id":"north_bob","name":"North Bay of Bengal","state":"West Bengal","basin":"Bay of Bengal","lat":20.8,"lon":89.5},
    {"id":"central_bob","name":"Central Bay of Bengal","state":"—","basin":"Bay of Bengal","lat":15.5,"lon":88.5},
    {"id":"andaman","name":"Andaman Sea","state":"Andaman & Nicobar Islands","basin":"Bay of Bengal","lat":11.7401,"lon":92.6586},
]

app = FastAPI(
    title="CYCLONE-X Intelligence API",
    version="2.0.0",
    description="Local inference API for the trained CYCLONE-X Model 1."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
track_imputer = track_scaler = era_imputer = era_scaler = None
manifest = None
ib = None
storm_lookup = {}
forecast_assets = {}

def sf(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default

def hursat_2d(da):
    a = np.squeeze(np.asarray(da.values))
    if a.ndim == 2:
        return a.astype(np.float32)
    if a.ndim == 3:
        return a[0].astype(np.float32)
    if a.ndim == 4:
        a = a[0]
        shapes = list(a.shape)
        spatial = sorted(range(3), key=lambda i: shapes[i], reverse=True)[:2]
        channel = [i for i in range(3) if i not in spatial][0]
        a = np.moveaxis(a, channel, -1)
        return np.nanmedian(a, axis=-1).astype(np.float32)
    return None

def find_satellite(ds):
    preferred = ["IRWIN","irwin","IRWIN_CDR","irwin_cdr","IRWIN_V","irwin_v","rad10","Rad10"]
    forbidden = ("archer","wind","pres","pressure","intensity","ring_score","spiral_score","center","vmax","mws")
    for wanted in preferred:
        for name in ds.data_vars:
            if name.lower() == wanted.lower() and not any(x in name.lower() for x in forbidden):
                try:
                    a = hursat_2d(ds[name])
                    if a is not None and np.isfinite(a).sum() >= 100:
                        return a
                except Exception:
                    pass
    return None

def normalize_satellite(a):
    if a is None:
        return None
    a = np.squeeze(np.asarray(a, dtype=np.float32))
    if a.ndim != 2:
        return None
    ok = np.isfinite(a)
    if ok.sum() < 100:
        return None
    fill = float(np.nanmedian(a[ok]))
    a = np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)
    lo, hi = np.percentile(a, [2,98])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return None
    a = np.clip(a, lo, hi)
    a = (a-lo)/(hi-lo+1e-8)
    img = Image.fromarray((a*255).astype(np.uint8), mode="L")
    img = img.resize((IMAGE_SIZE,IMAGE_SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(img, dtype=np.float32)/255.0)[...,None]

def build_features(d):
    d = d.sort_values(["SID","ISO_TIME"]).copy()
    g = d.groupby("SID", group_keys=False)
    for c in ["WMO_WIND","WMO_PRES","LAT","LON","STORM_SPEED","STORM_DIR","DIST2LAND"]:
        for lag in [1,2,3]:
            d[f"{c}_L{lag}"] = g[c].shift(lag)
    d["WIND_CHANGE_1"] = d["WMO_WIND_L1"]-d["WMO_WIND_L2"]
    d["WIND_CHANGE_2"] = d["WMO_WIND_L2"]-d["WMO_WIND_L3"]
    d["WIND_ACCELERATION"] = d["WIND_CHANGE_1"]-d["WIND_CHANGE_2"]
    d["PRESSURE_CHANGE_1"] = d["WMO_PRES_L1"]-d["WMO_PRES_L2"]
    d["PRESSURE_CHANGE_2"] = d["WMO_PRES_L2"]-d["WMO_PRES_L3"]
    d["PRESSURE_ACCELERATION"] = d["PRESSURE_CHANGE_1"]-d["PRESSURE_CHANGE_2"]
    d["SPEED_CHANGE_1"] = d["STORM_SPEED_L1"]-d["STORM_SPEED_L2"]
    d["LAND_CHANGE_1"] = d["DIST2LAND_L1"]-d["DIST2LAND_L2"]
    for source,prefix in [("WMO_WIND","WIND"),("WMO_PRES","PRESSURE"),("STORM_SPEED","SPEED"),("DIST2LAND","LAND")]:
        past = g[source].shift(1)
        pg = past.groupby(d["SID"], sort=False)
        for w in [2,3,4]:
            r = pg.rolling(w,min_periods=1)
            d[f"{prefix}_ROLL_MEAN_{w}"] = r.mean().reset_index(level=0,drop=True)
            d[f"{prefix}_ROLL_STD_{w}"] = r.std(ddof=0).reset_index(level=0,drop=True)
    d["LAT_RAD"]=np.deg2rad(d["LAT"]); d["LON_RAD"]=np.deg2rad(d["LON"])
    d["MONTH_SIN"]=np.sin(2*np.pi*d["ISO_TIME"].dt.month/12); d["MONTH_COS"]=np.cos(2*np.pi*d["ISO_TIME"].dt.month/12)
    d["HOUR_SIN"]=np.sin(2*np.pi*d["ISO_TIME"].dt.hour/24); d["HOUR_COS"]=np.cos(2*np.pi*d["ISO_TIME"].dt.hour/24)
    d["DOY_SIN"]=np.sin(2*np.pi*d["ISO_TIME"].dt.dayofyear/365.25); d["DOY_COS"]=np.cos(2*np.pi*d["ISO_TIME"].dt.dayofyear/365.25)
    d["STORM_AGE_HOURS"]=g["ISO_TIME"].transform(lambda x:(x-x.iloc[0]).dt.total_seconds()/3600.0)
    direction=np.deg2rad(d["STORM_DIR_L1"])
    d["PREV_MOTION_U"]=d["STORM_SPEED_L1"]*np.sin(direction)
    d["PREV_MOTION_V"]=d["STORM_SPEED_L1"]*np.cos(direction)
    return d

def nearest(storm_df, ts):
    dif = np.abs(
        storm_df["ISO_TIME"].values.astype("datetime64[ns]") -
        np.datetime64(ts.to_datetime64())
    ).astype("timedelta64[s]").astype(np.int64)
    i=int(np.argmin(dif))
    return storm_df.iloc[i], float(dif[i])/3600.0

def load_forecast_assets():
    global forecast_assets
    forecast_assets = {}
    fdir = BASE / "model_forecast"
    for h in [6,12,24,48]:
        mp=fdir/f"forecast_model_{h}h.joblib"
        ip=fdir/f"forecast_imputer_{h}h.joblib"
        sp=fdir/f"forecast_scaler_{h}h.joblib"
        if mp.exists() and ip.exists() and sp.exists():
            forecast_assets[h] = {
                "model": joblib.load(mp),
                "imputer": joblib.load(ip),
                "scaler": joblib.load(sp),
            }
    report_path=fdir/"forecast_report.json"
    if report_path.exists():
        try:
            with open(report_path,"r",encoding="utf-8") as f:
                forecast_assets["report"] = json.load(f)
        except Exception:
            pass

def load_assets():
    global model, track_imputer, track_scaler, era_imputer, era_scaler, manifest, ib, storm_lookup
    model_path = MODEL_BEST if MODEL_BEST.exists() else MODEL_FINAL
    if not model_path.exists():
        raise RuntimeError(f"Trained model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    track_imputer = joblib.load(MODEL_DIR/"track_imputer_model1.joblib")
    track_scaler = joblib.load(MODEL_DIR/"track_scaler_model1.joblib")
    era_imputer = joblib.load(MODEL_DIR/"era5_imputer_model1.joblib")
    era_scaler = joblib.load(MODEL_DIR/"era5_scaler_model1.joblib")
    manifest = pd.read_csv(MANIFEST_FILE)
    ib = pd.read_csv(IBTRACS_FILE, skiprows=[1], low_memory=False)
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    for c in ["LAT","LON","WMO_WIND","WMO_PRES","STORM_SPEED","STORM_DIR","DIST2LAND"]:
        if c in ib:
            ib[c] = pd.to_numeric(ib[c], errors="coerce")
    ib = ib.dropna(subset=["SID","ISO_TIME","LAT","LON","WMO_WIND"]).copy()
    ib = build_features(ib)
    storm_lookup = {
        sid:g.sort_values("ISO_TIME").reset_index(drop=True)
        for sid,g in ib.groupby("SID")
    }

@app.on_event("startup")
def startup():
    load_assets()
    load_forecast_assets()

@app.get("/")
def root():
    return {"project":"CYCLONE-X","status":"ready","model_loaded":model is not None}

@app.get("/api/health")
def health():
    return {
        "status":"ok",
        "model_loaded":model is not None,
        "model_path":str(MODEL_BEST if MODEL_BEST.exists() else MODEL_FINAL),
        "training_accuracy_hidden_from_ui":True,
        "forecast_model_loaded": all(h in forecast_assets for h in [6,12,24,48]),
        "forecast_horizons": [h for h in [6,12,24,48] if h in forecast_assets],
    }

@app.get("/api/model")
def model_info():
    return {
        "name":"CYCLONE-X Model 1",
        "task":"North Indian Ocean cyclone-stage classification",
        "classes":CLASS_NAMES,
        "domains":["Arabian Sea","Bay of Bengal"],
        "historical_satellite":"HURSAT-B1",
        "track_context":"NOAA IBTrACS",
        "environmental_context":"ERA5 artifacts",
        "live_satellite_adapter":"INSAT-3D / INSAT-3DR — connect authorized feed before operational use",
    }

@app.get("/api/regions")
def regions():
    return {"regions":REGIONS}

def scenario(stage):
    base={"Depression / Weak System":0.16,"Cyclonic Storm":0.52,"Severe Cyclone":0.78}[stage]
    short=[]
    for h in [6,12,16,24,36,48]:
        r=float(np.clip(base + 0.04*np.sin(h/7),0.02,0.96))
        short.append({"horizon":f"+{h}h","risk":round(r,3),"uncertainty_km":round(45+7*h,1)})
    long=[]
    for d in [10,20,30,40]:
        p=float(np.clip(0.52 - 0.006*d + 0.08*(base-.5),.05,.72))
        long.append({"horizon":f"+{d}d","probability":round(p,3),"type":"probabilistic trend outlook"})
    return short,long

def state_risk(region, stage, active):
    states=["Kerala","Karnataka","Goa","Maharashtra","Gujarat","Tamil Nadu","Andhra Pradesh","Odisha","West Bengal","Andaman & Nicobar Islands"]
    if not active:
        return [{"state":s,"zone":"GREEN","score":0} for s in states]
    relevant = {
        "Arabian Sea":{"Kerala","Karnataka","Goa","Maharashtra","Gujarat","Tamil Nadu"},
        "Bay of Bengal":{"Tamil Nadu","Andhra Pradesh","Odisha","West Bengal","Andaman & Nicobar Islands","Kerala"},
    }[region["basin"]]
    severity={"Cyclonic Storm":2,"Severe Cyclone":3}.get(stage,1)
    out=[]
    for idx,s in enumerate(states):
        score=severity if s in relevant else max(0,severity-1)
        if idx % 5 == 0 and s in relevant:
            score=min(3,score+1)
        out.append({"state":s,"zone":["GREEN","YELLOW","ORANGE","RED"][score],"score":score})
    return out


WMO_WEATHER = {
    0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
    45:"Fog", 48:"Depositing rime fog", 51:"Light drizzle", 53:"Drizzle",
    55:"Heavy drizzle", 56:"Freezing drizzle", 57:"Heavy freezing drizzle",
    61:"Light rain", 63:"Rain", 65:"Heavy rain", 66:"Freezing rain", 67:"Heavy freezing rain",
    71:"Light snow", 73:"Snow", 75:"Heavy snow", 77:"Snow grains",
    80:"Light rain showers", 81:"Rain showers", 82:"Heavy rain showers",
    85:"Light snow showers", 86:"Heavy snow showers", 95:"Thunderstorm",
    96:"Thunderstorm with hail", 99:"Severe thunderstorm with hail"
}

def get_daily_weather(region):
    params = {
        "latitude": region["lat"],
        "longitude": region["lon"],
        "daily": ",".join([
            "weather_code", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "precipitation_probability_max",
            "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
            "cloud_cover_mean", "relative_humidity_2m_mean", "pressure_msl_mean"
        ]),
        "forecast_days": 7,
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "kmh",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm"
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent":"CYCLONE-X/3.1"})
    with urllib.request.urlopen(req, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    d = payload.get("daily", {})
    keys = [
        "time", "weather_code", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "precipitation_probability_max", "wind_speed_10m_max",
        "wind_gusts_10m_max", "wind_direction_10m_dominant", "cloud_cover_mean",
        "relative_humidity_2m_mean", "pressure_msl_mean"
    ]
    rows=[]
    n=len(d.get("time", []))
    for i in range(n):
        code=int(d.get("weather_code", [0]*n)[i] or 0)
        rain=float(d.get("precipitation_sum", [0]*n)[i] or 0)
        pop=float(d.get("precipitation_probability_max", [0]*n)[i] or 0)
        wind=float(d.get("wind_speed_10m_max", [0]*n)[i] or 0)
        condition=WMO_WEATHER.get(code, "Variable conditions")
        if pop >= 70 or rain >= 15:
            outlook="Wet / unsettled"
        elif pop >= 40 or rain >= 5:
            outlook="Chance of rain"
        elif wind >= 35:
            outlook="Windy"
        elif code <= 2:
            outlook="Mostly calm"
        else:
            outlook="Mixed conditions"
        rows.append({
            "date": d["time"][i],
            "condition": condition,
            "outlook": outlook,
            "weather_code": code,
            "temp_max_c": round(float(d.get("temperature_2m_max", [None]*n)[i]),1),
            "temp_min_c": round(float(d.get("temperature_2m_min", [None]*n)[i]),1),
            "rain_mm": round(rain,1),
            "rain_probability": round(pop),
            "wind_kmh": round(wind,1),
            "gust_kmh": round(float(d.get("wind_gusts_10m_max", [0]*n)[i] or 0),1),
            "wind_direction_deg": round(float(d.get("wind_direction_10m_dominant", [0]*n)[i] or 0)),
            "cloud_cover_pct": round(float(d.get("cloud_cover_mean", [0]*n)[i] or 0)),
            "humidity_pct": round(float(d.get("relative_humidity_2m_mean", [0]*n)[i] or 0)),
            "pressure_hpa": round(float(d.get("pressure_msl_mean", [0]*n)[i] or 0),1)
        })
    return rows

@app.get("/api/weather-outlook/{region_id}")
def weather_outlook(region_id: str):
    region = next((r for r in REGIONS if r["id"] == region_id), None)
    if not region:
        raise HTTPException(404, "Region not found")
    try:
        days = get_daily_weather(region)
    except Exception as exc:
        raise HTTPException(503, f"Daily weather service unavailable: {exc}")
    return {
        "region": region,
        "days": days,
        "horizon_days": len(days),
        "source": "Open-Meteo weather forecast model aggregation",
        "source_url": "https://open-meteo.com/en/docs",
        "official": False,
        "note": "A short-range weather outlook, not an official IMD warning and not a guarantee. Values are model-based estimates and can change as new observations arrive."
    }

@app.get("/api/forecast-demo/{index}")
def forecast_demo(index:int=0):
    """Historical replay: forecast from a test-set origin without using future data."""
    if manifest is None:
        raise HTTPException(500,"Manifest unavailable")
    missing=[h for h in [6,12,24,48] if h not in forecast_assets]
    if missing:
        raise HTTPException(503, f"Forecast models missing for horizons: {missing}. Run train_cyclonex_forecast.py first.")
    test=manifest[manifest["split"].astype(str).str.lower()=="test"].reset_index(drop=True)
    if index<0 or index>=len(test):
        raise HTTPException(404,"Forecast test sample index out of range")
    row=test.iloc[index]
    sid=str(row["sid"]); ts=pd.Timestamp(row["timestamp"])
    if sid not in storm_lookup:
        raise HTTPException(404,"SID not found")
    g=storm_lookup[sid]
    exact=g[g["ISO_TIME"]==ts]
    if len(exact)==0:
        tr,diff=nearest(g,ts)
    else:
        tr=exact.iloc[0]; diff=0.0
    results=[]
    for h in [6,12,24,48]:
        asset=forecast_assets[h]
        X=np.array([[sf(tr.get(f, np.nan)) for f in asset["scaler"].feature_names_in_]],dtype=np.float32) if hasattr(asset["scaler"],"feature_names_in_") else np.array([[sf(tr.get(f, np.nan)) for f in ["LAT"]]],dtype=np.float32)
        # The trainer saves the feature list separately; recover it from the scaler columns when available.
        feature_file=BASE/"model_forecast"/f"forecast_features_{h}h.joblib"
        features=joblib.load(feature_file) if feature_file.exists() else list(asset["scaler"].feature_names_in_)
        X=np.array([[sf(tr.get(f,np.nan)) for f in features]],dtype=np.float32)
        X=asset["imputer"].transform(X)
        X=asset["scaler"].transform(X)
        pred=asset["model"].predict(X)[0]
        plat=float(tr["LAT"]+pred[0]); plon=float(tr["LON"]+pred[1]); pwind=float(max(0,tr["WMO_WIND"]+pred[2]))
        uncertainty=150.0
        rep=forecast_assets.get("report",{}).get("metrics",{}).get(str(h),{})
        uncertainty=float(rep.get("validation_track_p90_km", max(60, 35+2.2*h)))
        results.append({"horizon_h":h,"lat":round(plat,4),"lon":round(plon,4),"wind_kt":round(pwind,1),"uncertainty_km":round(uncertainty,1)})
    return {
        "mode":"HISTORICAL REPLAY",
        "sid":sid,"origin_time":ts.isoformat(),"origin_lat":float(tr["LAT"]),"origin_lon":float(tr["LON"]),
        "origin_wind_kt":float(tr["WMO_WIND"]),"forecast":results,
        "model":"CYCLONE-X Model 2 — direct multi-horizon track + intensity",
        "leakage_safe_origin":True,
        "note":"Forecast starts from information available at the origin timestamp. Future observations are not supplied to the model. This is a historical replay, not a live IMD forecast."
    }

@app.get("/api/analyze/{region_id}")
def analyze(region_id: str, scenario_stage: Optional[str] = None):
    region = next((r for r in REGIONS if r["id"] == region_id), None)
    if not region:
        raise HTTPException(404, "Region not found")

    # Model 1 is a 3-class cyclone-stage classifier. It is not a validated
    # 6/12/24/48-hour track/intensity forecasting model, so live mode never
    # invents a future forecast from this classifier.
    is_scenario = scenario_stage in CLASS_NAMES and scenario_stage != CLASS_NAMES[0]
    stage = scenario_stage if is_scenario else "No active cyclone"

    states = [
        "Kerala", "Karnataka", "Goa", "Maharashtra", "Gujarat",
        "Tamil Nadu", "Andhra Pradesh", "Odisha", "West Bengal",
        "Andaman & Nicobar Islands"
    ]
    if is_scenario:
        severity = {"Cyclonic Storm": 2, "Severe Cyclone": 3}[scenario_stage]
        relevant = ({"Kerala", "Karnataka", "Goa", "Maharashtra", "Gujarat"}
                    if region["basin"] == "Arabian Sea" else
                    {"Tamil Nadu", "Andhra Pradesh", "Odisha", "West Bengal", "Andaman & Nicobar Islands"})
        risks = []
        for s in states:
            score = severity if s in relevant else max(0, severity - 1)
            zone = ["GREEN", "YELLOW", "ORANGE", "RED"][min(3, score)]
            risks.append({"state": s, "zone": zone, "score": score, "mode": "SCENARIO"})
    else:
        # Neutral state layer when there is no model-confirmed active event.
        risks = [{"state": s, "zone": "GREEN", "score": 0, "mode": "LIVE"} for s in states]

    return {
        "region": region,
        "satellite": {
            "satellite": "INSAT-3DR",
            "product": "Rapid Scan / visible sector image",
            "source": "India Meteorological Department",
            "source_url": "https://mausam.imd.gov.in/responsive/satellite_rapidscan.php",
            "image_url": "https://mausam.imd.gov.in/Satellite/rswmo_vis.jpg",
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "live_image": True,
            "model_fed_directly": False,
            "note": "The official satellite image is shown as the live observation layer. Model 1 was trained and validated on historical HURSAT/IBTrACS/ERA5 data; a separate operational INSAT-to-Model adapter is required before claiming direct live inference."
        },
        "observation": {
            "mode": "LIVE OBSERVATION LAYER",
            "active_cyclone": False if not is_scenario else True,
            "status": "SCENARIO: " + scenario_stage if is_scenario else "NO ACTIVE CYCLONE DETECTED",
            "message": (
                "Scenario output only — not a live warning." if is_scenario else
                "No cyclone-stage event is assigned to this selected coastal region by Model 1. The live satellite image remains visible for observation."
            )
        },
        "classification": {
            "stage": stage,
            "active": is_scenario,
            "source": "CYCLONE-X Model 1"
        },
        "forecast_6_48h": [],
        "long_outlook": [],
        "state_risk": risks,
        "ai_agent": {
            "summary": (
                f"{region['name']} is in {region['basin']}. Live mode shows the official satellite observation layer and keeps the state risk neutral until a validated event is returned. "
                if not is_scenario else
                f"Scenario mode is simulating {scenario_stage} conditions for {region['name']}. These zones are demonstration outputs, not official warnings."
            ),
            "what_changed": "A fresh satellite observation should trigger a new model/forecast run once the operational INSAT adapter is validated.",
            "precautions": [
                "Check the latest official IMD/RSMC bulletin before operational action.",
                "Review coastal, port and emergency-readiness status for the selected region.",
                "Re-run the analysis after a materially changed satellite/weather observation."
            ]
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "CYCLONE-X is a prototype decision-support system. Model 1 classifies cyclone stage; it does not by itself generate validated 6–48 hour or 10–40 day track forecasts. Scenario risk is not an official warning."
    }

@app.get("/api/test-sample/{index}")
def test_sample(index:int=0):
    if manifest is None:
        raise HTTPException(500,"Manifest unavailable")
    test=manifest[manifest["split"].astype(str).str.lower()=="test"].reset_index(drop=True)
    if index<0 or index>=len(test):
        raise HTTPException(404,"Test sample index out of range")
    row=test.iloc[index]
    sid=str(row["sid"]); ts=pd.Timestamp(row["timestamp"])
    p=Path(str(row["hursat_file"]))
    if not p.exists():
        found=list(HURSAT_DIR.rglob(p.name))
        if not found:
            raise HTTPException(404,f"HURSAT file not found: {p.name}")
        p=found[0]
    try:
        with xr.open_dataset(p, decode_times=False) as ds:
            raw=find_satellite(ds)
            image=normalize_satellite(raw)
    except Exception as e:
        raise HTTPException(500,f"Satellite preprocessing failed: {e}")
    if image is None:
        raise HTTPException(500,"No valid HURSAT image found")
    if sid not in storm_lookup:
        raise HTTPException(404,"SID not found in IBTrACS")
    tr,diff=nearest(storm_lookup[sid],ts)
    if diff>MATCH_HOURS:
        raise HTTPException(500,f"IBTrACS match is {diff:.2f} hours away")
    track=np.array([[sf(tr.get(f,0),0) for f in TRACK_FEATURES]],dtype=np.float32)
    track=track_imputer.transform(track)
    track=track_scaler.transform(track)
    era=np.zeros((1,9),dtype=np.float32)
    era=era_imputer.transform(era)
    era=era_scaler.transform(era)
    quality=np.array([[float(np.std(image)),0.0]],dtype=np.float32)
    probs=model.predict([image[None,...],track,era,quality],verbose=0)[0]
    pred=int(np.argmax(probs)); actual=int(row["target"])
    return {
        "sid":sid,
        "timestamp":ts.isoformat(),
        "actual_class":CLASS_NAMES[actual],
        "predicted_class":CLASS_NAMES[pred],
        "confidence":round(float(probs[pred]),4),
        "correct":bool(pred==actual),
        "probabilities":{CLASS_NAMES[i]:round(float(probs[i]),4) for i in range(3)}
    }

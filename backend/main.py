import os
import json
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CYCLONE-X VERCEL LIGHTWEIGHT API GATEWAY
# ============================================================
#
# This Vercel version intentionally does NOT import:
#   TensorFlow
#   xarray
#   pandas
#   netCDF4
#   Pillow
#   scikit-learn
#   joblib
#
# Heavy Model 1 / Model 2 inference is delegated to the
# ML backend through CYCLONEX_ML_BACKEND_URL.
#
# Lightweight functionality remains directly on Vercel:
#   - Regions
#   - Dashboard API
#   - INSAT-3DR observation layer
#   - 7-day weather
#   - Scenario mode
#   - State risk
#   - Model information
#
# Heavy functionality:
#   - Model 1 HURSAT inference
#   - Model 2 6/12/24/48h forecasting
#
# ============================================================


BASE = Path(__file__).resolve().parent.parent

ML_BACKEND_URL = os.getenv(
    "CYCLONEX_ML_BACKEND_URL",
    ""
).rstrip("/")


CLASS_NAMES = [
    "Depression / Weak System",
    "Cyclonic Storm",
    "Severe Cyclone",
]


# ============================================================
# MONITORING REGIONS
# ============================================================

REGIONS = [
    {
        "id": "kochi",
        "name": "Kochi Coast",
        "state": "Kerala",
        "basin": "Arabian Sea",
        "lat": 9.9312,
        "lon": 76.2673,
    },
    {
        "id": "mangaluru",
        "name": "Mangaluru Coast",
        "state": "Karnataka",
        "basin": "Arabian Sea",
        "lat": 12.9141,
        "lon": 74.8560,
    },
    {
        "id": "goa",
        "name": "Goa Coast",
        "state": "Goa",
        "basin": "Arabian Sea",
        "lat": 15.4909,
        "lon": 73.8278,
    },
    {
        "id": "mumbai",
        "name": "Mumbai Coast",
        "state": "Maharashtra",
        "basin": "Arabian Sea",
        "lat": 19.0760,
        "lon": 72.8777,
    },
    {
        "id": "konkan",
        "name": "Konkan Coast",
        "state": "Maharashtra",
        "basin": "Arabian Sea",
        "lat": 16.9902,
        "lon": 73.3120,
    },
    {
        "id": "gujarat",
        "name": "Gujarat Coast",
        "state": "Gujarat",
        "basin": "Arabian Sea",
        "lat": 21.6417,
        "lon": 69.6293,
    },
    {
        "id": "kutch",
        "name": "Kutch Coast",
        "state": "Gujarat",
        "basin": "Arabian Sea",
        "lat": 23.7337,
        "lon": 69.8597,
    },
    {
        "id": "lakshadweep",
        "name": "Lakshadweep",
        "state": "Lakshadweep",
        "basin": "Arabian Sea",
        "lat": 10.5667,
        "lon": 72.6417,
    },
    {
        "id": "south_arabian",
        "name": "South Arabian Sea",
        "state": "—",
        "basin": "Arabian Sea",
        "lat": 12.0,
        "lon": 65.0,
    },
    {
        "id": "north_arabian",
        "name": "North Arabian Sea",
        "state": "—",
        "basin": "Arabian Sea",
        "lat": 20.0,
        "lon": 64.0,
    },
    {
        "id": "chennai",
        "name": "Chennai Coast",
        "state": "Tamil Nadu",
        "basin": "Bay of Bengal",
        "lat": 13.0827,
        "lon": 80.2707,
    },
    {
        "id": "puducherry",
        "name": "Puducherry Coast",
        "state": "Puducherry",
        "basin": "Bay of Bengal",
        "lat": 11.9416,
        "lon": 79.8083,
    },
    {
        "id": "andhra",
        "name": "Andhra Coast",
        "state": "Andhra Pradesh",
        "basin": "Bay of Bengal",
        "lat": 16.0,
        "lon": 81.0,
    },
    {
        "id": "visakhapatnam",
        "name": "Visakhapatnam Coast",
        "state": "Andhra Pradesh",
        "basin": "Bay of Bengal",
        "lat": 17.6868,
        "lon": 83.2185,
    },
    {
        "id": "paradip",
        "name": "Paradip Coast",
        "state": "Odisha",
        "basin": "Bay of Bengal",
        "lat": 20.3167,
        "lon": 86.6083,
    },
    {
        "id": "odisha",
        "name": "Odisha Coast",
        "state": "Odisha",
        "basin": "Bay of Bengal",
        "lat": 19.8135,
        "lon": 85.8312,
    },
    {
        "id": "west_bengal",
        "name": "West Bengal Coast",
        "state": "West Bengal",
        "basin": "Bay of Bengal",
        "lat": 21.9497,
        "lon": 87.7479,
    },
    {
        "id": "north_bob",
        "name": "North Bay of Bengal",
        "state": "West Bengal",
        "basin": "Bay of Bengal",
        "lat": 20.8,
        "lon": 89.5,
    },
    {
        "id": "central_bob",
        "name": "Central Bay of Bengal",
        "state": "—",
        "basin": "Bay of Bengal",
        "lat": 15.5,
        "lon": 88.5,
    },
    {
        "id": "andaman",
        "name": "Andaman Sea",
        "state": "Andaman & Nicobar Islands",
        "basin": "Bay of Bengal",
        "lat": 11.7401,
        "lon": 92.6586,
    },
]


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="CYCLONE-X Intelligence API",
    version="3.0.0",
    description=(
        "CYCLONE-X lightweight Vercel gateway for cyclone "
        "intelligence, official satellite observation, weather, "
        "scenario analysis and ML-backend proxying."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def find_region(region_id: str):
    return next(
        (
            r
            for r in REGIONS
            if r["id"] == region_id
        ),
        None,
    )


def require_ml_backend():

    if not ML_BACKEND_URL:
        raise HTTPException(
            status_code=503,
            detail={
                "message":
                    "Heavy ML backend is not configured.",
                "hint":
                    "Set CYCLONEX_ML_BACKEND_URL "
                    "to the Model 1/Model 2 backend URL.",
            },
        )


def proxy_ml_get(path: str):

    require_ml_backend()

    url = (
        ML_BACKEND_URL
        + "/"
        + path.lstrip("/")
    )

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent":
                    "CYCLONE-X-Vercel-Gateway/3.0"
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:

            payload = (
                response
                .read()
                .decode("utf-8")
            )

        return json.loads(payload)

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail={
                "message":
                    "Heavy ML backend unavailable.",
                "error":
                    str(exc),
            },
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "CYCLONE-X",
        "status": "ready",
        "version": "3.0.0",
        "deployment":
            "Vercel lightweight gateway",
        "ml_backend_configured":
            bool(ML_BACKEND_URL),
        "weather_service":
            "Open-Meteo",
        "satellite":
            "INSAT-3DR official observation layer",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "project": "CYCLONE-X",
        "platform": "Vercel",
        "lightweight_api": True,

        "ml_backend_configured":
            bool(ML_BACKEND_URL),

        "model_1":
            "Remote ML backend"
            if ML_BACKEND_URL
            else "ML backend not configured",

        "model_2":
            "Remote ML backend"
            if ML_BACKEND_URL
            else "ML backend not configured",

        "weather_service":
            "Open-Meteo",

        "satellite":
            "INSAT-3DR official observation layer",

        "memory_optimized":
            True,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/api/model")
def model_info():

    return {

        "name":
            "CYCLONE-X",

        "model_1": {

            "name":
                "CYCLONE-X Model 1",

            "task":
                "North Indian Ocean "
                "cyclone-stage classification",

            "classes":
                CLASS_NAMES,

            "historical_satellite":
                "HURSAT-B1",

            "track_context":
                "NOAA IBTrACS",

            "environmental_context":
                "ERA5",

            "execution":
                "Dedicated ML backend",
        },

        "model_2": {

            "name":
                "CYCLONE-X Model 2",

            "task":
                "6/12/24/48-hour track "
                "and intensity forecasting",

            "horizons":
                [6, 12, 24, 48],

            "execution":
                "Dedicated ML backend",
        },

        "domains": [
            "Arabian Sea",
            "Bay of Bengal",
        ],

        "live_satellite_adapter":
            "INSAT-3D / INSAT-3DR "
            "observation layer",
    }


# ============================================================
# REGIONS
# ============================================================

@app.get("/api/regions")
def regions():

    return {
        "regions": REGIONS
    }


# ============================================================
# SCENARIO GENERATOR
# ============================================================

def scenario(stage):

    base = {
        "Depression / Weak System":
            0.16,

        "Cyclonic Storm":
            0.52,

        "Severe Cyclone":
            0.78,
    }[stage]

    short = []

    for h in [
        6,
        12,
        16,
        24,
        36,
        48,
    ]:

        import math

        risk = max(
            0.02,
            min(
                0.96,
                base
                + 0.04
                * math.sin(h / 7),
            ),
        )

        short.append(
            {
                "horizon":
                    f"+{h}h",

                "risk":
                    round(risk, 3),

                "uncertainty_km":
                    round(
                        45 + 7 * h,
                        1,
                    ),
            }
        )

    long = []

    for d in [
        10,
        20,
        30,
        40,
    ]:

        probability = max(
            0.05,
            min(
                0.72,
                0.52
                - 0.006 * d
                + 0.08
                * (base - 0.5),
            ),
        )

        long.append(
            {
                "horizon":
                    f"+{d}d",

                "probability":
                    round(
                        probability,
                        3,
                    ),

                "type":
                    "probabilistic trend outlook",
            }
        )

    return short, long


# ============================================================
# STATE RISK
# ============================================================

def state_risk(
    region,
    stage,
    active,
):

    states = [
        "Kerala",
        "Karnataka",
        "Goa",
        "Maharashtra",
        "Gujarat",
        "Tamil Nadu",
        "Andhra Pradesh",
        "Odisha",
        "West Bengal",
        "Andaman & Nicobar Islands",
    ]

    if not active:

        return [
            {
                "state": s,
                "zone": "GREEN",
                "score": 0,
                "mode": "LIVE",
            }
            for s in states
        ]

    relevant = {

        "Arabian Sea": {
            "Kerala",
            "Karnataka",
            "Goa",
            "Maharashtra",
            "Gujarat",
            "Tamil Nadu",
        },

        "Bay of Bengal": {
            "Tamil Nadu",
            "Andhra Pradesh",
            "Odisha",
            "West Bengal",
            "Andaman & Nicobar Islands",
            "Kerala",
        },

    }[region["basin"]]

    severity = {

        "Cyclonic Storm":
            2,

        "Severe Cyclone":
            3,

    }.get(
        stage,
        1,
    )

    result = []

    for s in states:

        score = (
            severity
            if s in relevant
            else max(
                0,
                severity - 1,
            )
        )

        zone = [
            "GREEN",
            "YELLOW",
            "ORANGE",
            "RED",
        ][
            min(
                3,
                score,
            )
        ]

        result.append(
            {
                "state":
                    s,

                "zone":
                    zone,

                "score":
                    score,

                "mode":
                    "SCENARIO",
            }
        )

    return result


# ============================================================
# WEATHER CODES
# ============================================================

WMO_WEATHER = {

    0:
        "Clear sky",

    1:
        "Mainly clear",

    2:
        "Partly cloudy",

    3:
        "Overcast",

    45:
        "Fog",

    48:
        "Depositing rime fog",

    51:
        "Light drizzle",

    53:
        "Drizzle",

    55:
        "Heavy drizzle",

    56:
        "Freezing drizzle",

    57:
        "Heavy freezing drizzle",

    61:
        "Light rain",

    63:
        "Rain",

    65:
        "Heavy rain",

    66:
        "Freezing rain",

    67:
        "Heavy freezing rain",

    71:
        "Light snow",

    73:
        "Snow",

    75:
        "Heavy snow",

    77:
        "Snow grains",

    80:
        "Light rain showers",

    81:
        "Rain showers",

    82:
        "Heavy rain showers",

    85:
        "Light snow showers",

    86:
        "Heavy snow showers",

    95:
        "Thunderstorm",

    96:
        "Thunderstorm with hail",

    99:
        "Severe thunderstorm with hail",
}


# ============================================================
# 7-DAY WEATHER
# ============================================================

def get_daily_weather(region):

    params = {

        "latitude":
            region["lat"],

        "longitude":
            region["lon"],

        "daily":
            ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max",
                    "wind_direction_10m_dominant",
                    "cloud_cover_mean",
                    "relative_humidity_2m_mean",
                    "pressure_msl_mean",
                ]
            ),

        "forecast_days":
            7,

        "timezone":
            "Asia/Kolkata",

        "wind_speed_unit":
            "kmh",

        "temperature_unit":
            "celsius",

        "precipitation_unit":
            "mm",
    }

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urllib.parse.urlencode(params)
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "CYCLONE-X/3.1"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=10,
    ) as response:

        payload = json.loads(
            response
            .read()
            .decode("utf-8")
        )

    d = payload.get(
        "daily",
        {},
    )

    times = d.get(
        "time",
        [],
    )

    rows = []

    def value(
        key,
        i,
        default=0,
    ):

        values = d.get(
            key,
            [],
        )

        if (
            i >= len(values)
            or values[i] is None
        ):

            return default

        return values[i]

    for i, date in enumerate(times):

        code = int(
            value(
                "weather_code",
                i,
                0,
            )
            or 0
        )

        rain = float(
            value(
                "precipitation_sum",
                i,
                0,
            )
            or 0
        )

        pop = float(
            value(
                "precipitation_probability_max",
                i,
                0,
            )
            or 0
        )

        wind = float(
            value(
                "wind_speed_10m_max",
                i,
                0,
            )
            or 0
        )

        condition = WMO_WEATHER.get(
            code,
            "Variable conditions",
        )

        if (
            pop >= 70
            or rain >= 15
        ):

            outlook = (
                "Wet / unsettled"
            )

        elif (
            pop >= 40
            or rain >= 5
        ):

            outlook = (
                "Chance of rain"
            )

        elif wind >= 35:

            outlook = "Windy"

        elif code <= 2:

            outlook = "Mostly calm"

        else:

            outlook = (
                "Mixed conditions"
            )

        rows.append(
            {
                "date":
                    date,

                "condition":
                    condition,

                "outlook":
                    outlook,

                "weather_code":
                    code,

                "temp_max_c":
                    round(
                        float(
                            value(
                                "temperature_2m_max",
                                i,
                            )
                        ),
                        1,
                    ),

                "temp_min_c":
                    round(
                        float(
                            value(
                                "temperature_2m_min",
                                i,
                            )
                        ),
                        1,
                    ),

                "rain_mm":
                    round(
                        rain,
                        1,
                    ),

                "rain_probability":
                    round(pop),

                "wind_kmh":
                    round(
                        wind,
                        1,
                    ),

                "gust_kmh":
                    round(
                        float(
                            value(
                                "wind_gusts_10m_max",
                                i,
                            )
                        ),
                        1,
                    ),

                "wind_direction_deg":
                    round(
                        float(
                            value(
                                "wind_direction_10m_dominant",
                                i,
                            )
                        )
                    ),

                "cloud_cover_pct":
                    round(
                        float(
                            value(
                                "cloud_cover_mean",
                                i,
                            )
                        )
                    ),

                "humidity_pct":
                    round(
                        float(
                            value(
                                "relative_humidity_2m_mean",
                                i,
                            )
                        )
                    ),

                "pressure_hpa":
                    round(
                        float(
                            value(
                                "pressure_msl_mean",
                                i,
                            )
                        ),
                        1,
                    ),
            }
        )

    return rows


@app.get(
    "/api/weather-outlook/{region_id}"
)
def weather_outlook(
    region_id: str,
):

    region = find_region(
        region_id
    )

    if not region:

        raise HTTPException(
            404,
            "Region not found",
        )

    try:

        days = get_daily_weather(
            region
        )

    except Exception as exc:

        raise HTTPException(
            503,
            "Daily weather service unavailable: "
            + str(exc),
        )

    return {

        "region":
            region,

        "days":
            days,

        "horizon_days":
            len(days),

        "source":
            "Open-Meteo weather forecast model aggregation",

        "source_url":
            "https://open-meteo.com/en/docs",

        "official":
            False,

        "note":
            "A short-range weather outlook, "
            "not an official IMD warning and "
            "not a guarantee. Values are "
            "model-based estimates and can "
            "change as new observations arrive.",
    }


# ============================================================
# MODEL 2 FORECAST PROXY
# ============================================================

@app.get(
    "/api/forecast-demo/{index}"
)
def forecast_demo(
    index: int = 0,
):

    return proxy_ml_get(
        f"/api/forecast-demo/{index}"
    )


# ============================================================
# REGION ANALYSIS
# ============================================================

@app.get(
    "/api/analyze/{region_id}"
)
def analyze(
    region_id: str,
    scenario_stage: Optional[str] = None,
):

    region = find_region(
        region_id
    )

    if not region:

        raise HTTPException(
            404,
            "Region not found",
        )

    is_scenario = (
        scenario_stage in CLASS_NAMES
        and scenario_stage
        != CLASS_NAMES[0]
    )

    stage = (
        scenario_stage
        if is_scenario
        else "No active cyclone"
    )

    risks = state_risk(
        region,
        stage,
        is_scenario,
    )

    if is_scenario:

        short, long = scenario(
            scenario_stage
        )

        observation_status = (
            f"SCENARIO: {scenario_stage}"
        )

        observation_message = (
            "Scenario output only — "
            "not a live warning."
        )

    else:

        short = []
        long = []

        observation_status = (
            "NO ACTIVE CYCLONE DETECTED"
        )

        observation_message = (
            "No cyclone-stage event is assigned "
            "to this selected coastal region "
            "by the prototype live mode. "
            "The official satellite image "
            "remains visible for observation."
        )

    return {

        "region":
            region,

        "satellite": {

            "satellite":
                "INSAT-3DR",

            "product":
                "Rapid Scan / visible sector image",

            "source":
                "India Meteorological Department",

            "source_url":
                "https://mausam.imd.gov.in/responsive/satellite_rapidscan.php",

            "image_url":
                "https://mausam.imd.gov.in/Satellite/rswmo_vis.jpg",

            "refreshed_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "live_image":
                True,

            "model_fed_directly":
                False,

            "note":
                "The official satellite image is "
                "shown as the live observation layer. "
                "Model 1 was trained and validated on "
                "historical HURSAT/IBTrACS/ERA5 data; "
                "a separate operational INSAT-to-Model "
                "adapter is required before claiming "
                "direct live inference.",
        },

        "observation": {

            "mode":
                "LIVE OBSERVATION LAYER",

            "active_cyclone":
                is_scenario,

            "status":
                observation_status,

            "message":
                observation_message,
        },

        "classification": {

            "stage":
                stage,

            "active":
                is_scenario,

            "source":
                "CYCLONE-X Model 1",
        },

        "forecast_6_48h":
            short,

        "long_outlook":
            long,

        "state_risk":
            risks,

        "ai_agent": {

            "summary": (

                (
                    f"Scenario mode is simulating "
                    f"{scenario_stage} conditions for "
                    f"{region['name']}. These zones are "
                    "demonstration outputs, not official "
                    "warnings."
                )

                if is_scenario

                else

                (
                    f"{region['name']} is in "
                    f"{region['basin']}. Live mode shows "
                    "the official satellite observation "
                    "layer and keeps state risk neutral "
                    "until a validated event is returned."
                )
            ),

            "what_changed":
                "A fresh satellite observation "
                "should trigger a new model/forecast "
                "run once the operational INSAT "
                "adapter is validated.",

            "precautions": [

                "Check the latest official "
                "IMD/RSMC bulletin before "
                "operational action.",

                "Review coastal, port and "
                "emergency-readiness status "
                "for the selected region.",

                "Re-run the analysis after a "
                "materially changed satellite/"
                "weather observation.",
            ],
        },

        "updated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "disclaimer":
            "CYCLONE-X is a prototype "
            "decision-support system. "
            "Model 1 classifies cyclone stage; "
            "it does not by itself generate "
            "validated 6–48 hour or 10–40 day "
            "track forecasts. Scenario risk "
            "is not an official warning.",
    }


# ============================================================
# MODEL 1 TEST SAMPLE PROXY
# ============================================================

@app.get(
    "/api/test-sample/{index}"
)
def test_sample(
    index: int = 0,
):

    return proxy_ml_get(
        f"/api/test-sample/{index}"
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
    )

# CYCLONE-X — India / INSAT Command Console

This build replaces the earlier prototype dashboard with an India-first command console:

- 20 monitored Arabian Sea / Bay of Bengal regions.
- Real India basemap using Leaflet + OpenStreetMap.
- Official IMD INSAT-3DR visible rapid-scan image displayed as a satellite overlay.
- Satellite/base-map toggle.
- No fabricated live cyclone: live mode remains neutral when Model 1 does not return an active event.
- Scenario mode is explicitly labelled and can be used for demonstrations.
- Connected FastAPI backend and trained Model 1 inference assets.
- Dark/light theme, animated cyclone/vortex hero, glass UI, scroll-reveal motion.
- AI-agent explanation panel and print/save-to-PDF report flow.

## Important model boundary

The currently trained Model 1 is a three-class cyclone-stage classifier. It is **not** a validated 6/12/24/48-hour track/intensity model and it is not a validated 10–40 day cyclone forecast model. The UI therefore does not fabricate those forecasts. The corresponding cards explain that a dedicated forecasting layer is required.

The official INSAT-3DR image is shown as the live observation layer. Directly feeding live INSAT imagery into Model 1 requires a validated INSAT-to-model preprocessing/feature adapter; this build does not falsely claim that conversion is already operational.

## Run backend first

From the project root:

```cmd
cd /d E:\Cyclone-x-prototype
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Check:

- http://127.0.0.1:8000/api/health
- http://127.0.0.1:8000/api/model
- http://127.0.0.1:8000/api/test-sample/100

## Run frontend

Open a second CMD:

```cmd
cd /d E:\Cyclone-x-prototype\frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`.

## Folder layout

```text
E:\Cyclone-x-prototype
├── datas\
├── model1\
├── backend\
│   ├── main.py
│   └── requirements.txt
├── frontend\
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src\
│       ├── main.jsx
│       └── styles.css
└── README.md
```

## Official observation source

The UI uses the public IMD INSAT-3DR Rapid Scan service and the current visible image endpoint used by that service. See:
https://mausam.imd.gov.in/responsive/satellite_rapidscan.php

Use the official IMD/RSMC products for operational warnings and decisions.

## 7-Day Daily Weather Outlook

The dashboard includes a separate **7-Day Local Weather Outlook** for the selected region. It displays daily:
- maximum / minimum temperature
- precipitation probability and amount
- maximum 10 m wind speed
- wind gusts
- cloud cover / humidity / pressure context
- human-readable daily condition

This is a model-based weather outlook retrieved through the Open-Meteo forecast API, not an official IMD warning and not a guaranteed prediction. The UI deliberately labels it as an outlook and allows refresh when conditions change.

API endpoint:
`GET /api/weather-outlook/{region_id}`

The cyclone 6/12/24/48-hour forecast remains a separate CYCLONE-X Model 2 capability. The two views should not be confused: Model 2 forecasts cyclone track/intensity, while the daily outlook provides general local weather guidance.

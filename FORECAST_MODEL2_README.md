# CYCLONE-X Model 2 — Forecasting

This adds the forecasting layer that was missing from the first rebuilt frontend.

## What it predicts

Direct 6h / 12h / 24h / 48h forecasts of:
- latitude
- longitude
- cyclone wind intensity (WMO wind)
- validation-derived uncertainty radius

## Leakage protection

- Entire cyclone SID is assigned to train, validation, or test.
- Future latitude/longitude/wind are targets only.
- Only origin-time and lagged observations are features.
- Imputation/scaling are fitted on training storms only.
- `leakage_audit_forecast.json` must report PASS.

## Run

```cmd
cd /d E:\Cyclone-x-prototype
python backend\train_cyclonex_forecast.py
```

The models are saved in `model_forecast/`.

Then restart FastAPI. The dashboard's **Run 6–48h forecast** button calls a historical replay endpoint and draws the predicted path and uncertainty region.

## Important

This first Model 2 implementation is a real temporally supervised forecasting baseline using IBTrACS track/intensity history. It is not a claim that the final operational forecast is already fused directly with live INSAT-3DR. The next upgrade is to add the ERA5 environmental branch to Model 2 and then expose the adaptive reliability gate across satellite + track + environment.

# CYCLONE-X Daily Weather Outlook

Endpoint: `/api/weather-outlook/{region_id}`

The endpoint requests a 7-day daily aggregation from Open-Meteo using the selected region coordinates and Asia/Kolkata timezone.

Displayed variables include:
- weather condition
- maximum/minimum 2 m temperature
- precipitation amount
- maximum precipitation probability
- maximum 10 m wind speed and gusts
- dominant wind direction
- cloud cover
- mean relative humidity
- mean sea-level pressure

This is **not** an official IMD warning and should not be presented as guaranteed weather. It is short-range model guidance and can change with new observations.

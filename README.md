# Flight Delay Dashboard

Flight Delay Dashboard is a Flask + D3 web application for exploring U.S. flight delay patterns.
The backend serves precomputed JSON assets and model-based prediction APIs, and the frontend renders an interactive route map with airport and route tooltips.

## Overview

- Backend: Flask app in `app.py`
- Frontend: D3 UI in `templates/index.html`
- Precomputed data: `static/nodes.json`, `static/edges.json`, `static/breakdown.json`, `static/propagation.json`
- Model artifacts: `Outputs/gb_pipeline_50k.pkl`, `Outputs/rf_pipeline_50k.pkl`

## Main Features

- U.S. airport delay visualization on map
- Route lines between source and destination airports
- Airport detail panel with delay-cause breakdown
- API endpoints for network, airport, propagation, summary, and prediction

## Local Run

1. Install dependencies.
2. Run the app:

```bash
python app.py
```

or

```bash
flask run
```

3. Open:

```text
http://127.0.0.1:5000
```

## API Endpoints

- `GET /api/network` : full node/edge network payload
- `GET /api/breakdown` : delay-cause breakdown by airport
- `GET /api/airport/<code>` : airport-level stats and breakdown
- `GET /api/propagation/<code>` : correlated delay airports
- `GET /api/summary` : dashboard summary metrics
- `GET /api/route/<source_code>/<target_code>` : route-level detail
- `POST /api/predict` : delay probability inference
- `GET /health` : health check

## Notes on Model Loading

If the model artifact does not load, it is usually due to a version mismatch between the environment and the package versions used when the `.pkl` file was created.

## Change List

### 1) Lightened UI theme update

- Updated `templates/index.html` with a softer, lightly toned map background.
- Rebalanced panel, card, tooltip, and legend colors to match the light style.
- Kept readable contrast while avoiding pure white-heavy surfaces.

### 2) Route hover update (SRC/DST)

- Updated route hover interactions in `templates/index.html`.
- Hovering a flight line now shows `SRC -> DST`, flight count, average delay, and delay rate.
- Added route lookup endpoint in `app.py`: `GET /api/route/<source_code>/<target_code>`.

## Backups

- Backend backup: `app_backup.py`
- Frontend backup: `index_backup.html`

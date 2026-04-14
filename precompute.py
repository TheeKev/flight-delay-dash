"""
precompute.py
Reads the 8 full merged CSVs (sampling 200k rows per file) and generates
4 JSON files for the Flask API to serve to the D3 dashboard.

Outputs (all written to ./static/):
  nodes.json       - airport-level delay stats + coordinates
  edges.json       - route-level flight counts + avg delay
  breakdown.json   - delay cause breakdown per airport
  propagation.json - correlated delay airports per hub
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR_1  = Path(r"D:\School\GATech\CSE6242\Project\Full_Data\2015-2019")
DATA_DIR_2  = Path(r"D:\School\GATech\CSE6242\Project\Full_Data\2022-2024")
OUTPUT_DIR  = Path(r"D:\School\GATech\CSE6242\Project\Sample_50k rows\static")
OUTPUT_DIR.mkdir(exist_ok=True)

# Rows sampled per yearly file — 200k x 8 files = 1.6M rows total
SAMPLE_PER_FILE = 200000

# ── Airport coordinates ───────────────────────────────────────────────────────
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277), "LAX": (33.9425, -118.4081),
    "ORD": (41.9742, -87.9073), "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737),"JFK": (40.6413, -73.7781),
    "SFO": (37.6213, -122.3790),"SEA": (47.4502, -122.3088),
    "LAS": (36.0840, -115.1537),"MCO": (28.4312, -81.3081),
    "EWR": (40.6895, -74.1745), "PHX": (33.4373, -112.0078),
    "IAH": (29.9902, -95.3368), "MIA": (25.7959, -80.2870),
    "BOS": (42.3656, -71.0096), "MSP": (44.8848, -93.2223),
    "DTW": (42.2124, -83.3534), "CLT": (35.2140, -80.9431),
    "PHL": (39.8729, -75.2437), "LGA": (40.7772, -73.8726),
    "BWI": (39.1754, -76.6684), "SLC": (40.7884, -111.9778),
    "DCA": (38.8512, -77.0402), "SAN": (32.7338, -117.1933),
    "MDW": (41.7868, -87.7522), "TPA": (27.9755, -82.5332),
    "HNL": (21.3187, -157.9224),"PDX": (45.5898, -122.5951),
    "DAL": (32.8471, -96.8518), "STL": (38.7487, -90.3700),
    "BNA": (36.1245, -86.6782), "AUS": (30.1975, -97.6664),
    "HOU": (29.6454, -95.2789), "OAK": (37.7213, -122.2208),
    "MCI": (39.2976, -94.7139), "RDU": (35.8776, -78.7875),
    "SMF": (38.6954, -121.5908),"SJC": (37.3626, -121.9290),
    "MSY": (29.9934, -90.2580), "CLE": (41.4117, -81.8498),
    "IND": (39.7173, -86.2944), "PIT": (40.4915, -80.2329),
    "CMH": (39.9980, -82.8919), "SAT": (29.5337, -98.4698),
    "RSW": (26.5362, -81.7552), "BDL": (41.9389, -72.6832),
    "MKE": (42.9472, -87.8966), "OMA": (41.3032, -95.8941),
    "JAX": (30.4941, -81.6879), "CVG": (39.0533, -84.6630),
    "MEM": (35.0424, -89.9767), "OGG": (20.8986, -156.4305),
    "ABQ": (35.0402, -106.6090),"BUR": (34.2007, -118.3592),
    "ONT": (34.0560, -117.6012),"SNA": (33.6757, -117.8682),
    "PBI": (26.6832, -80.0956), "FLL": (26.0726, -80.1527),
    "TUS": (32.1161, -110.9410),"ELP": (31.8072, -106.3779),
    "BOI": (43.5644, -116.2228),"GEG": (47.6199, -117.5339),
    "MHT": (42.9326, -71.4357), "BUF": (42.9405, -78.7322),
    "SYR": (43.1112, -76.1063), "RIC": (37.5052, -77.3197),
    "ORF": (36.8976, -76.0132), "GSO": (36.0978, -79.9373),
    "CHS": (32.8986, -80.0405), "SAV": (32.1276, -81.2021),
    "PNS": (30.4734, -87.1866), "MOB": (30.6912, -88.2428),
    "LIT": (34.7294, -92.2243), "TUL": (36.1984, -95.8881),
    "OKC": (35.3931, -97.6007), "ICT": (37.6499, -97.4331),
    "DSM": (41.5340, -93.6631), "MSN": (43.1399, -89.3375),
    "GRR": (42.8808, -85.5228), "DAY": (39.9024, -84.2194),
    "HSV": (34.6372, -86.7751), "BHM": (33.5629, -86.7535),
    "CAE": (33.9389, -81.1195), "GSP": (34.8957, -82.2189),
}

# ── Load data — sample from full yearly files ─────────────────────────────────
def load_data(data_dir_1: Path, data_dir_2: Path,
              sample_per_file: int = SAMPLE_PER_FILE) -> pd.DataFrame:

    # Collect full yearly files only — skip small_sample files
    all_files = (
        sorted(data_dir_1.glob("MERGED_????.csv")) +
        sorted(data_dir_2.glob("MERGED_????.csv"))
    )

    if not all_files:
        raise FileNotFoundError("No MERGED_YYYY.csv files found in either folder")

    print(f"Found {len(all_files)} full yearly files:")
    frames = []
    for f in all_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  Loading {f.name} ({size_mb:.0f} MB) — sampling {sample_per_file:,} rows ...")
        df = pd.read_csv(
            f,
            low_memory=False,
            on_bad_lines="skip",
            nrows=sample_per_file        # read only the first N rows
        )
        frames.append(df)
        print(f"    Got {len(df):,} rows, {len(df.columns)} columns")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined):,}")
    print(f"Columns: {list(combined.columns[:10])} ...")
    return combined


# ── Helper ────────────────────────────────────────────────────────────────────
def clean(val):
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


# ── 1. nodes.json ─────────────────────────────────────────────────────────────
def build_nodes(df: pd.DataFrame) -> list:
    print("\nBuilding nodes.json ...")

    flights = df[(df["CANCELLED"] == 0) & df["ARR_DEL15"].notna()].copy()
    print(f"  Working with {len(flights):,} non-cancelled flights")

    agg = flights.groupby("ORIGIN").agg(
        total_flights=("ARR_DEL15", "count"),
        pct_delayed=("ARR_DEL15", "mean"),
        avg_arr_delay=("ARR_DELAY", "mean"),
    ).reset_index()

    nodes = []
    skipped = 0
    for _, row in agg.iterrows():
        code = row["ORIGIN"]
        lat, lon = AIRPORT_COORDS.get(code, (None, None))
        if lat is None:
            skipped += 1
        nodes.append({
            "id":            code,
            "total_flights": int(row["total_flights"]),
            "pct_delayed":   round(float(row["pct_delayed"]), 4),
            "avg_arr_delay": round(float(clean(row["avg_arr_delay"]) or 0), 2),
            "lat":           lat,
            "lon":           lon,
        })

    nodes.sort(key=lambda x: x["total_flights"], reverse=True)
    print(f"  {len(nodes)} airport nodes generated")
    print(f"  {skipped} airports missing coordinates (will still appear, no map pin)")
    return nodes


# ── 2. edges.json ─────────────────────────────────────────────────────────────
def build_edges(df: pd.DataFrame) -> list:
    print("\nBuilding edges.json ...")

    flights = df[(df["CANCELLED"] == 0) & df["ARR_DEL15"].notna()].copy()

    agg = flights.groupby(["ORIGIN", "DEST"]).agg(
        flight_count=("ARR_DEL15", "count"),
        pct_delayed=("ARR_DEL15", "mean"),
        avg_arr_delay=("ARR_DELAY", "mean"),
    ).reset_index()

    # Minimum 50 flights per route given larger dataset
    agg = agg[agg["flight_count"] >= 50]

    edges = []
    for _, row in agg.iterrows():
        edges.append({
            "source":        row["ORIGIN"],
            "target":        row["DEST"],
            "flight_count":  int(row["flight_count"]),
            "pct_delayed":   round(float(row["pct_delayed"]), 4),
            "avg_arr_delay": round(float(clean(row["avg_arr_delay"]) or 0), 2),
        })

    print(f"  {len(edges)} route edges generated")
    return edges


# ── 3. breakdown.json ─────────────────────────────────────────────────────────
def build_breakdown(df: pd.DataFrame) -> dict:
    print("\nBuilding breakdown.json ...")

    delay_cols = [
        "CARRIER_DELAY", "WEATHER_DELAY",
        "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
    ]

    flights = df[
        (df["CANCELLED"] == 0) &
        df["ARR_DEL15"].notna() &
        (df["ARR_DEL15"] == 1)
    ].copy()

    print(f"  Working with {len(flights):,} delayed flights")

    for col in delay_cols:
        if col in flights.columns:
            flights[col] = flights[col].fillna(0)

    agg = flights.groupby("ORIGIN")[delay_cols].mean().reset_index()

    breakdown = {}
    for _, row in agg.iterrows():
        code = row["ORIGIN"]
        breakdown[code] = {
            "carrier":       round(float(clean(row.get("CARRIER_DELAY", 0)) or 0), 2),
            "weather":       round(float(clean(row.get("WEATHER_DELAY", 0)) or 0), 2),
            "nas":           round(float(clean(row.get("NAS_DELAY", 0)) or 0), 2),
            "security":      round(float(clean(row.get("SECURITY_DELAY", 0)) or 0), 2),
            "late_aircraft": round(float(clean(row.get("LATE_AIRCRAFT_DELAY", 0)) or 0), 2),
        }

    print(f"  Breakdown built for {len(breakdown)} airports")
    return breakdown


# ── 4. propagation.json ───────────────────────────────────────────────────────
def build_propagation(df: pd.DataFrame, nodes: list,
                      top_n_hubs: int = 20) -> dict:
    print("\nBuilding propagation.json ...")

    flights = df[(df["CANCELLED"] == 0) & df["ARR_DEL15"].notna()].copy()

    hub_codes = [n["id"] for n in nodes[:top_n_hubs]]
    print(f"  Hubs: {hub_codes}")

    daily = flights.groupby(["FL_DATE", "ORIGIN"]).agg(
        daily_delay_rate=("ARR_DEL15", "mean")
    ).reset_index()

    # 10% threshold works well with larger dataset
    daily["high_delay_day"] = (daily["daily_delay_rate"] > 0.10).astype(int)

    pivot = daily.pivot_table(
        index="FL_DATE", columns="ORIGIN",
        values="high_delay_day", fill_value=0
    )

    print(f"  Pivot shape: {pivot.shape} (dates x airports)")

    propagation = {}
    for hub in hub_codes:
        if hub not in pivot.columns:
            propagation[hub] = []
            continue

        hub_series = pivot[hub]
        correlations = []

        for airport in pivot.columns:
            if airport == hub:
                continue
            corr = hub_series.corr(pivot[airport])
            if not np.isnan(corr) and corr > 0.05:
                correlations.append({
                    "airport":     airport,
                    "correlation": round(float(corr), 4)
                })

        correlations.sort(key=lambda x: x["correlation"], reverse=True)
        propagation[hub] = correlations[:15]
        print(f"  {hub}: {len(propagation[hub])} correlated airports")

    return propagation


# ── Save JSON ─────────────────────────────────────────────────────────────────
def save_json(data, filename: str):
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {filename} ({path.stat().st_size / 1024:.1f} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_data(DATA_DIR_1, DATA_DIR_2)

    nodes       = build_nodes(df)
    edges       = build_edges(df)
    breakdown   = build_breakdown(df)
    propagation = build_propagation(df, nodes, top_n_hubs=20)

    print("\nSaving JSON files ...")
    save_json(nodes,        "nodes.json")
    save_json(edges,        "edges.json")
    save_json(breakdown,    "breakdown.json")
    save_json(propagation,  "propagation.json")

    print("\nDone. Files written to:", OUTPUT_DIR)
    print(f"  nodes.json       — {len(nodes)} airports")
    print(f"  edges.json       — {len(edges)} routes")
    print(f"  breakdown.json   — {len(breakdown)} airports")
    print(f"  propagation.json — {len(propagation)} hubs")


if __name__ == "__main__":
    main()
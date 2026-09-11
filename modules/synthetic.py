"""
Synthetic data generation for the OILTRACE pipeline.

Produces, inside data/:
  - spill_metadata.json  (centroid, polygon, estimated age)
  - sample_sar.png        (a simulated dual-pol SAR preview)
  - sample_ais.csv         (a handful of AIS vessel tracks, one carrying a
                             deliberate AIS reporting gap near the spill time)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

VESSEL_TEMPLATE = [
    {"MMSI": 413210001, "VesselName": "MT OCEAN MARAUDER", "VesselType": "Crude Oil Tanker", "speed": 13.8, "gap": True},
    {"MMSI": 211567000, "VesselName": "MV POSEIDON ALPHA", "VesselType": "Container Ship", "speed": 18.5, "gap": False},
    {"MMSI": 354890000, "VesselName": "PACIFIC TRADER", "VesselType": "Bulk Carrier", "speed": 12.0, "gap": False},
]


def _generate_spill_metadata(min_lat, max_lat, min_lon, max_lon, age_hours):
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    slick_lat = center_lat + (lat_span * 0.05)
    slick_lon = center_lon + (lon_span * 0.05)

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    polygon = []
    for a in angles:
        r_lat = np.random.uniform(lat_span * 0.01, lat_span * 0.02)
        r_lon = np.random.uniform(lon_span * 0.01, lon_span * 0.03)
        polygon.append([
            round(slick_lat + r_lat * np.sin(a), 5),
            round(slick_lon + r_lon * np.cos(a), 5),
        ])

    metadata = {
        "centroid": [round(slick_lat, 5), round(slick_lon, 5)],
        "origin_point": [round(center_lat, 5), round(center_lon, 5)],
        "polygon": polygon,
        "estimated_age_hours": age_hours,
    }
    with open(DATA_DIR / "spill_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _generate_sar_image():
    h, w = 512, 512
    vv_ocean = rng.gamma(shape=4.0, scale=130 / 4.0, size=(h, w))
    vh_ocean = rng.gamma(shape=2.0, scale=45 / 2.0, size=(h, w))

    slick_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(slick_mask, (256, 256), (90, 45), 35, 0, 360, 255, -1)
    slick_mask = cv2.GaussianBlur(slick_mask, (21, 21), 7)

    vv_img = np.clip(np.where(slick_mask > 50, vv_ocean * 0.25, vv_ocean), 0, 255).astype(np.uint8)
    vh_img = np.clip(np.where(slick_mask > 50, vh_ocean * 0.80, vh_ocean), 0, 255).astype(np.uint8)

    cv2.imwrite(str(DATA_DIR / "sample_sar.png"), cv2.merge([vv_img, vh_img, vv_img]))


def _generate_ais_traffic(min_lat, max_lat, min_lon, max_lon):
    now_utc = datetime.now(timezone.utc)
    start_time = now_utc - timedelta(hours=24)
    steps = 48
    center_lon = (min_lon + max_lon) / 2.0
    lon_span = max_lon - min_lon

    vessel_paths = [
        {**VESSEL_TEMPLATE[0], "start": [min_lat, min_lon], "end": [max_lat, max_lon]},
        {**VESSEL_TEMPLATE[1], "start": [min_lat, center_lon + lon_span * 0.2], "end": [max_lat, center_lon + lon_span * 0.2]},
        {**VESSEL_TEMPLATE[2], "start": [max_lat, min_lon], "end": [min_lat, max_lon]},
    ]

    records = []
    for vessel in vessel_paths:
        lats = np.linspace(vessel["start"][0], vessel["end"][0], steps)
        lons = np.linspace(vessel["start"][1], vessel["end"][1], steps)

        for i in range(steps):
            time = start_time + timedelta(minutes=30 * i)
            spill_window = 22 <= i <= 26

            if vessel["gap"] and spill_window and i % 2 == 0:
                continue

            speed = 4.5 if (vessel["gap"] and spill_window) else vessel["speed"]

            records.append({
                "MMSI": vessel["MMSI"],
                "BaseDateTime": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "LAT": round(float(lats[i] + rng.normal(0, 0.002)), 5),
                "LON": round(float(lons[i] + rng.normal(0, 0.002)), 5),
                "SOG": round(speed + rng.normal(0, 0.5), 1),
                "VesselName": vessel["VesselName"],
                "VesselType": vessel["VesselType"],
                "AIS_Gap_Flag": int(vessel["gap"]),
            })

    pd.DataFrame(records).to_csv(DATA_DIR / "sample_ais.csv", index=False)


def generate_dynamic_dataset(min_lat, max_lat, min_lon, max_lon, age_hours=12.0):
    """
    Builds a synthetic spill footprint, a SAR preview image, and a bounded
    AIS traffic sample. Writes to data/ and returns
    (sar_image_path, ais_csv_path).
    """
    DATA_DIR.mkdir(exist_ok=True)
    _generate_spill_metadata(min_lat, max_lat, min_lon, max_lon, age_hours)
    _generate_sar_image()
    _generate_ais_traffic(min_lat, max_lat, min_lon, max_lon)
    return str(DATA_DIR / "sample_sar.png"), str(DATA_DIR / "sample_ais.csv")
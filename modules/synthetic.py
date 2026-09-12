"""
Explicit hardcoded preset and dynamic synthetic generator for OILTRACE.
Guarantees distinct SAR textures, slick shapes, coordinates, and AIS vessel profiles
for Bush Hill, GC600, Coal Oil Point, and Cantarell.
"""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------
# 4 EXPLICITLY HARDCODED PRESET PROFILES
# ---------------------------------------------------------------------
EXPLICIT_PRESETS = {
    "bush_hill": {
        "name": "Bush Hill GC-185 (Green Canyon)",
        "region": "Gulf of Mexico",
        "bbox": [27.200, 28.300, -91.800, -90.700],
        "centroid": [27.7917, -91.2500],
        "origin_point": [27.7500, -91.2000],
        "polygon": [
            [27.8150, -91.2720], [27.8280, -91.2550], [27.8220, -91.2310],
            [27.7980, -91.2220], [27.7710, -91.2380], [27.7650, -91.2610],
            [27.7790, -91.2790], [27.8010, -91.2820]
        ],
        "sar_slick_center": (240, 230),
        "sar_slick_axes": (110, 48),
        "sar_slick_angle": 35,
        "current_speed": 0.52,
        "current_dir": 125.0,
        "vessels": [
            {"MMSI": 367410001, "name": "MT OCEAN MARAUDER", "type": "Crude Oil Tanker", "speed": 13.8, "cpa_spd": 4.2, "gap": True},
            {"MMSI": 368520002, "name": "MV GULF PIONEER", "type": "Chemical Tanker", "speed": 14.5, "cpa_spd": 14.2, "gap": False},
            {"MMSI": 369630003, "name": "GALVESTON EXPRESS", "type": "Container Ship", "speed": 18.2, "cpa_spd": 18.0, "gap": False},
            {"MMSI": 367740004, "name": "DELTA TRADER", "type": "Bulk Carrier", "speed": 11.8, "cpa_spd": 11.5, "gap": False},
        ]
    },
    "gc600": {
        "name": "GC600 Mega Plume (Central Deep GoM)",
        "region": "Deepwater GoM",
        "bbox": [26.800, 27.700, -90.800, -89.700],
        "centroid": [27.2067, -90.2828],
        "origin_point": [27.1800, -90.2500],
        "polygon": [
            [27.2350, -90.3100], [27.2480, -90.2800], [27.2390, -90.2500],
            [27.2150, -90.2400], [27.1820, -90.2580], [27.1750, -90.2900],
            [27.1890, -90.3150], [27.2180, -90.3200]
        ],
        "sar_slick_center": (280, 260),
        "sar_slick_axes": (140, 36),
        "sar_slick_angle": 75,
        "current_speed": 0.48,
        "current_dir": 140.0,
        "vessels": [
            {"MMSI": 412550111, "name": "DEEPWATER VOYAGER", "type": "Crude Oil Tanker", "speed": 14.2, "cpa_spd": 4.8, "gap": True},
            {"MMSI": 413660222, "name": "CARIBBEAN SPIRIT", "type": "Product Tanker", "speed": 13.4, "cpa_spd": 13.0, "gap": False},
            {"MMSI": 414770333, "name": "MISSISSIPPI STAR", "type": "Container Ship", "speed": 17.6, "cpa_spd": 17.5, "gap": False},
            {"MMSI": 415880444, "name": "PELICAN STATE", "type": "Platform Supply Vessel", "speed": 10.2, "cpa_spd": 10.0, "gap": False},
        ]
    },
    "coal_oil": {
        "name": "Coal Oil Point (Santa Barbara Channel)",
        "region": "California Pacific",
        "bbox": [34.100, 34.600, -120.200, -119.500],
        "centroid": [34.4014, -119.8792],
        "origin_point": [34.3900, -119.8600],
        "polygon": [
            [34.4220, -119.9050], [34.4310, -119.8800], [34.4240, -119.8550],
            [34.4010, -119.8450], [34.3820, -119.8620], [34.3780, -119.8900],
            [34.3910, -119.9120], [34.4100, -119.9150]
        ],
        "sar_slick_center": (190, 210),
        "sar_slick_axes": (85, 62),
        "sar_slick_angle": 120,
        "current_speed": 0.32,
        "current_dir": 155.0,
        "vessels": [
            {"MMSI": 366110999, "name": "PACIFIC GLORY", "type": "Crude Oil Tanker", "speed": 13.9, "cpa_spd": 3.9, "gap": True},
            {"MMSI": 366220888, "name": "CALIFORNIA MARINER", "type": "Product Tanker", "speed": 13.1, "cpa_spd": 12.8, "gap": False},
            {"MMSI": 366330777, "name": "SANTA BARBARA CLIPPER", "type": "Container Ship", "speed": 19.5, "cpa_spd": 19.2, "gap": False},
            {"MMSI": 366440666, "name": "CHANNEL VOYAGER", "type": "General Cargo", "speed": 12.0, "cpa_spd": 11.7, "gap": False},
        ]
    },
    "cantarell": {
        "name": "Cantarell Complex Seeps (Bay of Campeche)",
        "region": "Campeche / Mexico",
        "bbox": [19.000, 19.900, -92.800, -91.800],
        "centroid": [19.4167, -92.3167],
        "origin_point": [19.4000, -92.3000],
        "polygon": [
            [19.4520, -92.3550], [19.4650, -92.3200], [19.4580, -92.2850],
            [19.4310, -92.2700], [19.3950, -92.2900], [19.3880, -92.3300],
            [19.4050, -92.3620], [19.4300, -92.3680]
        ],
        "sar_slick_center": (260, 310),
        "sar_slick_axes": (130, 55),
        "sar_slick_angle": 15,
        "current_speed": 0.40,
        "current_dir": 290.0,
        "vessels": [
            {"MMSI": 345110555, "name": "PEMEX VOYAGER", "type": "Crude Oil Tanker", "speed": 13.5, "cpa_spd": 4.1, "gap": True},
            {"MMSI": 345220666, "name": "CAMPECHE TRADER", "type": "Product Tanker", "speed": 12.8, "cpa_spd": 12.5, "gap": False},
            {"MMSI": 345330777, "name": "MAYA TRANSPORTER", "type": "Chemical Tanker", "speed": 14.1, "cpa_spd": 13.9, "gap": False},
            {"MMSI": 345440888, "name": "AGUILA AZTECA", "type": "Container Ship", "speed": 17.2, "cpa_spd": 17.0, "gap": False},
        ]
    }
}


def _match_preset(min_lat, max_lat, min_lon, max_lon):
    """Identifies which preset matches the given bounding box."""
    for key, p in EXPLICIT_PRESETS.items():
        b = p["bbox"]
        if abs(min_lat - b[0]) < 0.15 and abs(max_lat - b[1]) < 0.15 and abs(min_lon - b[2]) < 0.15 and abs(max_lon - b[3]) < 0.15:
            return key, p
    return "bush_hill", EXPLICIT_PRESETS["bush_hill"]


def generate_dynamic_dataset(min_lat, max_lat, min_lon, max_lon, age_hours=12.0):
    """
    Overwrites sample_sar.png, sample_ais.csv, and spill_metadata.json
    with preset-specific, non-identical data.
    """
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    preset_key, preset = _match_preset(min_lat, max_lat, min_lon, max_lon)

    # 1. WRITE PRESET METADATA
    metadata = {
        "preset_key": preset_key,
        "preset_name": preset["name"],
        "centroid": preset["centroid"],
        "origin_point": preset["origin_point"],
        "polygon": preset["polygon"],
        "estimated_age_hours": age_hours,
        "search": {
            "min_lat": float(min_lat),
            "max_lat": float(max_lat),
            "min_lon": float(min_lon),
            "max_lon": float(max_lon),
        },
    }
    with open(DATA_DIR / "spill_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 2. GENERATE PRESET-SPECIFIC SAR IMAGE (Unique slick shape, position, and orientation)
    h, w = 512, 512
    # Base background texture
    y, x = np.mgrid[:h, :w]
    freq = {"bush_hill": 0.04, "gc600": 0.025, "coal_oil": 0.055, "cantarell": 0.035}.get(preset_key, 0.04)
    waves = np.sin(x * freq + y * (freq * 0.5)) * 2.0 + np.cos(x * (freq * 0.4) - y * freq) * 1.5
    speckle = np.random.default_rng(len(preset_key)).gamma(shape=4.0, scale=1.0, size=(h, w))
    ocean = 135.0 + waves * 5.0 + (speckle - 4.0) * 12.0

    # Draw dark slick mask
    slick_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = preset["sar_slick_center"]
    ax, ay = preset["sar_slick_axes"]
    angle = preset["sar_slick_angle"]
    cv2.ellipse(slick_mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
    
    # Secondary trailing streamer
    streamer_offset = (cx + int(ax * 0.4), cy - int(ay * 0.5))
    cv2.ellipse(slick_mask, streamer_offset, (int(ax * 0.6), int(ay * 0.4)), angle + 25, 0, 360, 180, -1)
    slick_mask = cv2.GaussianBlur(slick_mask, (25, 25), 9)

    # Attenuate backscatter inside slick
    slick_factor = 1.0 - (slick_mask.astype(np.float32) / 255.0) * 0.72
    sar_mono = np.clip(ocean * slick_factor, 0, 255).astype(np.uint8)
    sar_composite = cv2.merge([sar_mono, sar_mono, sar_mono])
    cv2.imwrite(str(DATA_DIR / "sample_sar.png"), sar_composite)

    # 3. GENERATE PRESET-SPECIFIC AIS TRACKS (Unique vessels, MMSIs, and paths)
    now_utc = datetime.now(timezone.utc)
    start_time = now_utc - timedelta(hours=24)
    steps = 48
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    records = []
    vessels = preset["vessels"]

    for idx, v in enumerate(vessels):
        # Position tracks realistically across the preset box
        y_offset = 0.18 + (idx * 0.22)
        start_pt = [min_lat + lat_span * y_offset, min_lon + lon_span * 0.05]
        end_pt = [max_lat - lat_span * (0.40 - y_offset * 0.4), max_lon - lon_span * 0.05]

        # Ensure the suspect vessel crosses close to the origin point
        if v["gap"]:
            start_pt = [min_lat + lat_span * 0.15, min_lon + lon_span * 0.10]
            end_pt = [max_lat - lat_span * 0.15, max_lon - lon_span * 0.10]

        lats = np.linspace(start_pt[0], end_pt[0], steps)
        lons = np.linspace(start_pt[1], end_pt[1], steps)

        for i in range(steps):
            t_point = start_time + timedelta(minutes=30 * i)
            spill_window = 22 <= i <= 26

            if v["gap"] and spill_window:
                spd = v["cpa_spd"]
                if i % 2 == 0:  # AIS reporting blackout
                    continue
            else:
                spd = v["speed"]

            jitter_lat = np.random.normal(0, 0.001)
            jitter_lon = np.random.normal(0, 0.001)

            records.append({
                "MMSI": v["MMSI"],
                "BaseDateTime": t_point.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "LAT": round(float(lats[i] + jitter_lat), 5),
                "LON": round(float(lons[i] + jitter_lon), 5),
                "SOG": round(float(spd + np.random.normal(0, 0.15)), 1),
                "VesselName": v["name"],
                "VesselType": v["type"],
                "AIS_Gap_Flag": int(v["gap"] and spill_window),
            })

    pd.DataFrame(records).to_csv(DATA_DIR / "sample_ais.csv", index=False)
    return str(DATA_DIR / "sample_sar.png"), str(DATA_DIR / "sample_ais.csv")
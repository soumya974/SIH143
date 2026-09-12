"""
Dynamic synthetic data generation for OILTRACE.
Generates authentic SAR imagery and regional AIS vessel tracks that dynamically adapt
to the exact geographic bounding box coordinates and ocean basin.
"""

import json
import math
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Preset-specific & Regional Vessel Pools
REGIONAL_VESSEL_POOLS = {
    "Gulf of Mexico": [
        {"name": "MT OCEAN MARAUDER", "type": "Crude Oil Tanker", "speed": 13.8, "gap": True},
        {"name": "MV GULF PIONEER", "type": "Chemical Tanker", "speed": 14.2, "gap": False},
        {"name": "GALVESTON EXPRESS", "type": "Container Ship", "speed": 18.5, "gap": False},
        {"name": "DELTA TRADER", "type": "Bulk Carrier", "speed": 11.5, "gap": False},
        {"name": "BAYOU TRANSPORTER", "type": "Offshore Supply Vessel", "speed": 10.0, "gap": False},
    ],
    "Deepwater GoM": [
        {"name": "DEEPWATER VOYAGER", "type": "Crude Oil Tanker", "speed": 14.1, "gap": True},
        {"name": "CARIBBEAN SPIRIT", "type": "Product Tanker", "speed": 13.4, "gap": False},
        {"name": "MISSISSIPPI STAR", "type": "Container Ship", "speed": 17.8, "gap": False},
        {"name": "PELICAN STATE", "type": "Platform Supply Vessel", "speed": 10.5, "gap": False},
    ],
    "California Pacific": [
        {"name": "PACIFIC GLORY", "type": "Crude Oil Tanker", "speed": 14.0, "gap": True},
        {"name": "CALIFORNIA MARINER", "type": "Product Tanker", "speed": 13.2, "gap": False},
        {"name": "SANTA BARBARA CLIPPER", "type": "Container Ship", "speed": 19.1, "gap": False},
        {"name": "CHANNEL VOYAGER", "type": "General Cargo", "speed": 12.0, "gap": False},
        {"name": "PACIFIC TITAN", "type": "Tug / Supply", "speed": 9.5, "gap": False},
    ],
    "Campeche / Mexico": [
        {"name": "PEMEX VOYAGER", "type": "Crude Oil Tanker", "speed": 13.5, "gap": True},
        {"name": "CAMPECHE TRADER", "type": "Product Tanker", "speed": 12.8, "gap": False},
        {"name": "MAYA TRANSPORTER", "type": "Chemical Tanker", "speed": 14.0, "gap": False},
        {"name": "AGUILA AZTECA", "type": "Container Ship", "speed": 17.2, "gap": False},
    ],
    "Global": [
        {"name": "MT GLOBAL NAVIGATOR", "type": "Crude Oil Tanker", "speed": 13.8, "gap": True},
        {"name": "MV OCEAN FREIGHTER", "type": "Container Ship", "speed": 17.5, "gap": False},
        {"name": "EASTERN CARRIER", "type": "Bulk Carrier", "speed": 12.0, "gap": False},
    ]
}


def _get_region_name(lat, lon):
    if 26.5 <= lat <= 28.0 and -91.0 <= lon <= -89.0:
        return "Deepwater GoM"
    elif 26.0 <= lat <= 30.5 and -95.0 <= lon <= -87.0:
        return "Gulf of Mexico"
    elif 33.0 <= lat <= 36.0 and -122.0 <= lon <= -118.0:
        return "California Pacific"
    elif 18.0 <= lat <= 22.0 and -94.0 <= lon <= -90.0:
        return "Campeche / Mexico"
    return "Global"


def _generate_spill_metadata(min_lat, max_lat, min_lon, max_lon, age_hours=12.0, coord_hash=2026):
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    rng = np.random.default_rng(coord_hash)

    # Offset slick centroid naturally from geographic origin based on coordinate hash
    slick_lat = center_lat + float(rng.uniform(lat_span * 0.02, lat_span * 0.08))
    slick_lon = center_lon + float(rng.uniform(lon_span * 0.02, lon_span * 0.08))

    # Irregular polygonal footprint scaled to this preset's dimensions
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    polygon = []
    r_base_lat = lat_span * rng.uniform(0.020, 0.038)
    r_base_lon = lon_span * rng.uniform(0.030, 0.055)

    for a in angles:
        var = 1.0 + 0.35 * np.sin(2 * a) + 0.2 * np.cos(3 * a)
        polygon.append([
            round(float(slick_lat + r_base_lat * var * np.sin(a)), 5),
            round(float(slick_lon + r_base_lon * var * np.cos(a)), 5),
        ])

    metadata = {
        "centroid": [round(float(slick_lat), 5), round(float(slick_lon), 5)],
        "origin_point": [round(float(center_lat), 5), round(float(center_lon), 5)],
        "polygon": polygon,
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

    return metadata


def _generate_sar_image(coord_hash=2026):
    sar_path = DATA_DIR / "sample_sar.png"
    h, w = 480, 480
    rng = np.random.default_rng(coord_hash)
    y, x = np.mgrid[:h, :w]

    # Synthesize realistic ocean radar Bragg scattering texture
    waves = np.sin(x * 0.045 + y * 0.02) * 1.8 + np.cos(x * 0.02 - y * 0.035) * 1.2
    speckle = rng.gamma(shape=4.0, scale=1.0, size=(h, w))
    ocean = -14.0 + waves + (speckle - 4.0) * 1.6

    # Preset-specific slick location, elongation, and orientation
    shift_x = int(rng.uniform(-35, 35))
    shift_y = int(rng.uniform(-35, 35))
    scale_x = rng.uniform(80.0, 130.0)
    scale_y = rng.uniform(45.0, 75.0)

    dx = (x - (240 + shift_x)) / scale_x
    dy = (y - (240 + shift_y)) / scale_y
    angle = rng.uniform(0.2, 1.2)
    rot_x = dx * math.cos(angle) - dy * math.sin(angle)
    rot_y = dx * math.sin(angle) + dy * math.cos(angle)

    slick = np.exp(-((rot_x ** 2) / 0.90 + (rot_y ** 2) / 0.35))
    streamer = 0.40 * np.exp(-(((rot_x + 0.5) ** 2) / 0.25 + ((rot_y - 0.25) ** 2) / 0.55))

    sar_db = ocean - np.clip(slick + streamer, 0, 1) * 9.5
    norm_img = np.clip((sar_db - (-30.0)) / 25.0 * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(str(sar_path), cv2.merge([norm_img, norm_img, norm_img]))


def _generate_ais_traffic(min_lat, max_lat, min_lon, max_lon, coord_hash=2026):
    now_utc = datetime.now(timezone.utc)
    start_time = now_utc - timedelta(hours=24)
    steps = 48

    rng = np.random.default_rng(coord_hash)
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    region = _get_region_name(center_lat, center_lon)
    pool = REGIONAL_VESSEL_POOLS.get(region, REGIONAL_VESSEL_POOLS["Global"])

    records = []
    base_mmsi = 310000000 + (coord_hash % 50000000)

    for idx, template in enumerate(pool[:4]):
        v_mmsi = base_mmsi + (idx + 1) * 111111

        frac_y = 0.15 + idx * 0.22
        start_pt = [min_lat + lat_span * frac_y, min_lon + lon_span * 0.08]
        end_pt = [max_lat - lat_span * (0.45 - frac_y * 0.4), max_lon - lon_span * 0.08]

        lats = np.linspace(start_pt[0], end_pt[0], steps)
        lons = np.linspace(start_pt[1], end_pt[1], steps)

        for i in range(steps):
            time_val = start_time + timedelta(minutes=30 * i)
            spill_window = 22 <= i <= 26
            spd = template["speed"]

            if template["gap"] and spill_window:
                # Anomaly: rogue vessel throttles down and produces an AIS reporting gap
                spd = float(rng.uniform(3.8, 5.2))
                if i % 2 == 0:
                    continue

            p_lat = float(np.clip(lats[i] + rng.normal(0, lat_span * 0.0015), min_lat, max_lat))
            p_lon = float(np.clip(lons[i] + rng.normal(0, lon_span * 0.0015), min_lon, max_lon))

            records.append({
                "MMSI": int(v_mmsi),
                "BaseDateTime": time_val.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "LAT": round(p_lat, 5),
                "LON": round(p_lon, 5),
                "SOG": round(float(spd + rng.normal(0, 0.2)), 1),
                "VesselName": template["name"],
                "VesselType": template["type"],
                "AIS_Gap_Flag": int(template["gap"] and spill_window),
            })

    pd.DataFrame(records).to_csv(DATA_DIR / "sample_ais.csv", index=False)


def generate_dynamic_dataset(min_lat, max_lat, min_lon, max_lon, age_hours=12.0):
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    coord_str = f"{min_lat:.3f}_{max_lat:.3f}_{min_lon:.3f}_{max_lon:.3f}"
    coord_hash = int(hashlib.md5(coord_str.encode()).hexdigest(), 16) % (2**31)

    _generate_spill_metadata(min_lat, max_lat, min_lon, max_lon, age_hours, coord_hash)
    _generate_sar_image(coord_hash)
    _generate_ais_traffic(min_lat, max_lat, min_lon, max_lon, coord_hash)

    return str(DATA_DIR / "sample_sar.png"), str(DATA_DIR / "sample_ais.csv")
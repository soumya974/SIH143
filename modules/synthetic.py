"""
Dynamic synthetic data generation for OILTRACE.
Generates authentic SAR imagery metadata and regional AIS vessel tracks that dynamically adapt
based on the exact geographic bounding box coordinates entered by the user.
"""

import json
import math
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

REGIONAL_VESSEL_POOLS = {
    "Gulf of Mexico": [
        {"name": "MT OCEAN MARAUDER", "type": "Crude Oil Tanker", "speed": 13.8, "gap": True},
        {"name": "MV GULF PIONEER", "type": "Chemical Tanker", "speed": 14.2, "gap": False},
        {"name": "GALVESTON EXPRESS", "type": "Container Ship", "speed": 18.5, "gap": False},
        {"name": "DELTA TRADER", "type": "Bulk Carrier", "speed": 11.5, "gap": False},
        {"name": "BAYOU TRANSPORTER", "type": "Offshore Supply Vessel", "speed": 10.0, "gap": False},
    ],
    "California Pacific": [
        {"name": "PACIFIC GLORY", "type": "Crude Oil Tanker", "speed": 14.0, "gap": True},
        {"name": "CALIFORNIA MARINER", "type": "Product Tanker", "speed": 13.2, "gap": False},
        {"name": "SANTA BARBARA CLIPPER", "type": "Container Ship", "speed": 19.1, "gap": False},
        {"name": "CHANNEL VOYAGER", "type": "General Cargo", "speed": 12.0, "gap": False},
    ],
    "North Sea": [
        {"name": "NORDIC TITAN", "type": "Crude Oil Tanker", "speed": 13.5, "gap": True},
        {"name": "BERGEN ENTERPRISE", "type": "Shuttle Tanker", "speed": 12.8, "gap": False},
        {"name": "NORTH SEA TRADER", "type": "Platform Supply Vessel", "speed": 11.0, "gap": False},
    ],
    "Global": [
        {"name": "MT GLOBAL NAVIGATOR", "type": "Crude Oil Tanker", "speed": 13.8, "gap": True},
        {"name": "MV OCEAN FREIGHTER", "type": "Container Ship", "speed": 17.5, "gap": False},
        {"name": "EASTERN CARRIER", "type": "Bulk Carrier", "speed": 12.0, "gap": False},
    ]
}


def _get_region_name(lat, lon):
    if 24.0 <= lat <= 31.0 and -98.0 <= lon <= -80.0:
        return "Gulf of Mexico"
    elif 32.0 <= lat <= 38.0 and -123.0 <= lon <= -116.0:
        return "California Pacific"
    elif 52.0 <= lat <= 64.0 and -4.0 <= lon <= 10.0:
        return "North Sea"
    return "Global"


def _generate_spill_metadata(min_lat, max_lat, min_lon, max_lon, age_hours=12.0):
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    coord_hash = int(hashlib.md5(f"{min_lat:.3f}_{max_lat:.3f}_{min_lon:.3f}_{max_lon:.3f}".encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(coord_hash)

    slick_lat = center_lat + (lat_span * 0.04) + float(rng.uniform(-lat_span * 0.02, lat_span * 0.02))
    slick_lon = center_lon + (lon_span * 0.04) + float(rng.uniform(-lon_span * 0.02, lon_span * 0.02))

    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    polygon = []
    for a in angles:
        r_lat = (lat_span * 0.025) * (1.0 + 0.3 * np.sin(2 * a) + 0.15 * np.cos(3 * a))
        r_lon = (lon_span * 0.040) * (1.0 + 0.3 * np.cos(2 * a) - 0.15 * np.sin(a))
        polygon.append([
            round(float(slick_lat + r_lat), 5),
            round(float(slick_lon + r_lon), 5),
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
    
    waves = np.sin(x * 0.045 + y * 0.02) * 1.8 + np.cos(x * 0.02 - y * 0.035) * 1.2
    speckle = rng.gamma(shape=4.0, scale=1.0, size=(h, w))
    ocean = -14.0 + waves + (speckle - 4.0) * 1.6
    
    shift_x = int(rng.uniform(-30, 30))
    shift_y = int(rng.uniform(-30, 30))
    dx = (x - (240 + shift_x)) / 120.0
    dy = (y - (230 + shift_y)) / 65.0
    angle = rng.uniform(0.3, 0.8)
    rot_x = dx * math.cos(angle) - dy * math.sin(angle)
    rot_y = dx * math.sin(angle) + dy * math.cos(angle)
    slick = np.exp(-((rot_x ** 2) / 0.90 + (rot_y ** 2) / 0.35))
    streamer = 0.35 * np.exp(-(((rot_x + 0.5) ** 2) / 0.25 + ((rot_y - 0.2) ** 2) / 0.55))
    
    sar_db = ocean - np.clip(slick + streamer, 0, 1) * 9.5
    norm_img = np.clip((sar_db - (-30)) / 25.0 * 255.0, 0, 255).astype(np.uint8)
    
    if cv2 is not None:
        cv2.imwrite(str(sar_path), cv2.merge([norm_img, norm_img, norm_img]))
    elif Image is not None:
        Image.fromarray(norm_img).save(str(sar_path))


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
    base_mmsi = 413000000 + (coord_hash % 100000)

    for idx, template in enumerate(pool[:3]):
        v_mmsi = base_mmsi + idx * 11111
        
        frac_y = 0.15 + idx * 0.25
        start_pt = [min_lat + lat_span * frac_y, min_lon + lon_span * 0.10]
        end_pt = [max_lat - lat_span * (0.50 - frac_y * 0.5), max_lon - lon_span * 0.10]

        lats = np.linspace(start_pt[0], end_pt[0], steps)
        lons = np.linspace(start_pt[1], end_pt[1], steps)

        for i in range(steps):
            time_val = start_time + timedelta(minutes=30 * i)
            spill_window = 22 <= i <= 26
            spd = template["speed"]

            if template["gap"] and spill_window:
                spd = 4.2

            p_lat = float(np.clip(lats[i] + rng.normal(0, lat_span * 0.002), min_lat, max_lat))
            p_lon = float(np.clip(lons[i] + rng.normal(0, lon_span * 0.002), min_lon, max_lon))

            records.append({
                "MMSI": v_mmsi,
                "BaseDateTime": time_val.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "LAT": round(p_lat, 5),
                "LON": round(p_lon, 5),
                "SOG": round(float(spd + rng.normal(0, 0.2)), 1),
                "VesselName": template["name"],
                "VesselType": template["type"],
                "AIS_Gap_Flag": int(template["gap"]),
            })

    pd.DataFrame(records).to_csv(DATA_DIR / "sample_ais.csv", index=False)


def generate_dynamic_dataset(min_lat, max_lat, min_lon, max_lon, age_hours=12.0):
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    coord_hash = int(hashlib.md5(f"{min_lat:.3f}_{max_lat:.3f}_{min_lon:.3f}_{max_lon:.3f}".encode()).hexdigest(), 16) % (2**31)
    
    _generate_spill_metadata(min_lat, max_lat, min_lon, max_lon, age_hours)
    _generate_sar_image(coord_hash)
    _generate_ais_traffic(min_lat, max_lat, min_lon, max_lon, coord_hash)
    
    return str(DATA_DIR / "sample_sar.png"), str(DATA_DIR / "sample_ais.csv")
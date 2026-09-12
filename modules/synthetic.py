"""
Realistic Preset Synthetic Generator for OILTRACE.
Generates distinct SAR radar backscatter, authentic feathered slick boundaries,
and realistic regional AIS vessel tracks per preset location.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import cv2
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

PRESET_CONFIGS = {
    "bush_hill": {
        "name": "Bush Hill GC-185 (Green Canyon)",
        "bbox": [27.200, 28.300, -91.800, -90.700],
        "centroid": [27.7917, -91.2500],
        "origin_point": [27.7550, -91.2150],
        "polygon": [
            [27.8210, -91.2750], [27.8340, -91.2480], [27.8180, -91.2180],
            [27.7890, -91.2050], [27.7620, -91.2290], [27.7530, -91.2610],
            [27.7720, -91.2880], [27.8010, -91.2910]
        ],
        "sar_seed": 101,
        "sar_slick_center": (220, 240),
        "sar_params": {"length": 130, "width": 45, "angle": 40, "curv": 0.35},
        "vessels": [
            {"MMSI": 367410001, "name": "MT OCEAN MARAUDER", "type": "Crude Oil Tanker", "speed": 13.8, "cpa_spd": 4.2, "gap": True},
            {"MMSI": 368520002, "name": "MV GULF PIONEER", "type": "Chemical Tanker", "speed": 14.5, "cpa_spd": 14.1, "gap": False},
            {"MMSI": 369630003, "name": "GALVESTON EXPRESS", "type": "Container Ship", "speed": 18.2, "cpa_spd": 17.9, "gap": False},
            {"MMSI": 367740004, "name": "DELTA TRADER", "type": "Bulk Carrier", "speed": 11.8, "cpa_spd": 11.5, "gap": False},
        ]
    },
    "gc600": {
        "name": "GC600 Mega Plume (Green Canyon 600)",
        "bbox": [26.800, 27.700, -90.800, -89.700],
        "centroid": [27.2067, -90.2828],
        "origin_point": [27.1720, -90.2450],
        "polygon": [
            [27.2410, -90.3150], [27.2560, -90.2780], [27.2380, -90.2410],
            [27.2110, -90.2310], [27.1780, -90.2520], [27.1690, -90.2880],
            [27.1850, -90.3190], [27.2150, -90.3240]
        ],
        "sar_seed": 202,
        "sar_slick_center": (280, 270),
        "sar_params": {"length": 165, "width": 38, "angle": 80, "curv": 0.60},
        "vessels": [
            {"MMSI": 412550111, "name": "DEEPWATER VOYAGER", "type": "Crude Oil Tanker", "speed": 14.2, "cpa_spd": 4.8, "gap": True},
            {"MMSI": 413660222, "name": "CARIBBEAN SPIRIT", "type": "Product Tanker", "speed": 13.4, "cpa_spd": 13.1, "gap": False},
            {"MMSI": 414770333, "name": "MISSISSIPPI STAR", "type": "Container Ship", "speed": 17.6, "cpa_spd": 17.4, "gap": False},
            {"MMSI": 415880444, "name": "PELICAN STATE", "type": "Platform Supply Vessel", "speed": 10.2, "cpa_spd": 10.0, "gap": False},
        ]
    },
    "coal_oil": {
        "name": "Coal Oil Point (Santa Barbara Channel)",
        "bbox": [34.100, 34.600, -120.200, -119.500],
        "centroid": [34.4014, -119.8792],
        "origin_point": [34.3850, -119.8550],
        "polygon": [
            [34.4280, -119.9120], [34.4360, -119.8790], [34.4250, -119.8480],
            [34.3990, -119.8390], [34.3780, -119.8590], [34.3710, -119.8920],
            [34.3880, -119.9190], [34.4120, -119.9220]
        ],
        "sar_seed": 303,
        "sar_slick_center": (180, 200),
        "sar_params": {"length": 95, "width": 60, "angle": 125, "curv": 0.20},
        "vessels": [
            {"MMSI": 366110999, "name": "PACIFIC GLORY", "type": "Crude Oil Tanker", "speed": 13.9, "cpa_spd": 3.9, "gap": True},
            {"MMSI": 366220888, "name": "CALIFORNIA MARINER", "type": "Product Tanker", "speed": 13.1, "cpa_spd": 12.8, "gap": False},
            {"MMSI": 366330777, "name": "SANTA BARBARA CLIPPER", "type": "Container Ship", "speed": 19.5, "cpa_spd": 19.2, "gap": False},
            {"MMSI": 366440666, "name": "CHANNEL VOYAGER", "type": "General Cargo", "speed": 12.0, "cpa_spd": 11.8, "gap": False},
        ]
    },
    "cantarell": {
        "name": "Cantarell Complex Seeps (Bay of Campeche)",
        "bbox": [19.000, 19.900, -92.800, -91.800],
        "centroid": [19.4167, -92.3167],
        "origin_point": [19.3950, -92.2900],
        "polygon": [
            [19.4610, -92.3620], [19.4750, -92.3180], [19.4620, -92.2780],
            [19.4320, -92.2610], [19.3890, -92.2850], [19.3790, -92.3310],
            [19.4010, -92.3690], [19.4350, -92.3750]
        ],
        "sar_seed": 404,
        "sar_slick_center": (260, 310),
        "sar_params": {"length": 150, "width": 50, "angle": 20, "curv": 0.45},
        "vessels": [
            {"MMSI": 345110555, "name": "PEMEX VOYAGER", "type": "Crude Oil Tanker", "speed": 13.5, "cpa_spd": 4.1, "gap": True},
            {"MMSI": 345220666, "name": "CAMPECHE TRADER", "type": "Product Tanker", "speed": 12.8, "cpa_spd": 12.5, "gap": False},
            {"MMSI": 345330777, "name": "MAYA TRANSPORTER", "type": "Chemical Tanker", "speed": 14.1, "cpa_spd": 13.9, "gap": False},
            {"MMSI": 345440888, "name": "AGUILA AZTECA", "type": "Container Ship", "speed": 17.2, "cpa_spd": 17.0, "gap": False},
        ]
    }
}


def _match_preset(min_lat, max_lat, min_lon, max_lon):
    for key, p in PRESET_CONFIGS.items():
        b = p["bbox"]
        if abs(min_lat - b[0]) < 0.25 and abs(max_lat - b[1]) < 0.25 and abs(min_lon - b[2]) < 0.25 and abs(max_lon - b[3]) < 0.25:
            return key, p
    return "bush_hill", PRESET_CONFIGS["bush_hill"]


def _synthesize_realistic_sar(preset_cfg):
    """
    Synthesizes authentic dual-polarization Sentinel-1 SAR imagery with
    speckle noise and capillary-damped organic slicks.
    """
    h, w = 512, 512
    rng = np.random.default_rng(preset_cfg["sar_seed"])

    # Bragg sea-surface texture
    y, x = np.mgrid[:h, :w]
    freq = 0.035
    swell = np.sin(x * freq + y * 0.015) * 3.5 + np.cos(x * 0.01 - y * freq) * 2.8
    speckle_vv = rng.gamma(shape=3.8, scale=1.1, size=(h, w))
    vv_db = -12.0 + swell + (speckle_vv - 4.0) * 2.2
    vh_db = vv_db - rng.uniform(6.5, 8.5)

    # Damped slick mask with feathered dispersion tails
    cx, cy = preset_cfg["sar_slick_center"]
    p = preset_cfg["sar_params"]
    ang_rad = math.radians(p["angle"])

    dx = (x - cx) * math.cos(ang_rad) + (y - cy) * math.sin(ang_rad)
    dy = -(x - cx) * math.sin(ang_rad) + (y - cy) * math.cos(ang_rad)
    dy_curved = dy - p["curv"] * (dx ** 2) / float(p["length"])

    dist_sq = (dx / (p["length"] / 2.0)) ** 2 + (dy_curved / (p["width"] / 2.0)) ** 2
    slick_core = np.exp(-dist_sq * 1.5)

    # Trailing wind/current streamers
    streamer = 0.35 * np.exp(-(((dx + 40.0) / 70.0) ** 2 + ((dy_curved - 15.0) / 18.0) ** 2))
    attenuation_mask = np.clip(slick_core + streamer, 0.0, 1.0)
    attenuation_mask = cv2.GaussianBlur(attenuation_mask.astype(np.float32), (17, 17), 5)

    # Oil dampens Bragg scattering by ~9 to 14 dB
    vv_slick = vv_db - (attenuation_mask * 11.5)
    vh_slick = vh_db - (attenuation_mask * 4.0)

    # Normalized 8-bit composite: R=VH, G=VV-VH ratio, B=VV
    r_chan = np.clip((vh_slick - (-28.0)) / 22.0 * 255.0, 0, 255).astype(np.uint8)
    b_chan = np.clip((vv_slick - (-22.0)) / 20.0 * 255.0, 0, 255).astype(np.uint8)
    g_chan = np.clip((b_chan.astype(float) - r_chan.astype(float) + 50.0) * 1.5, 0, 255).astype(np.uint8)

    composite = cv2.merge([b_chan, g_chan, r_chan])
    cv2.imwrite(str(DATA_DIR / "sample_sar.png"), composite)


def _synthesize_ais_tracks(preset_cfg, min_lat, max_lat, min_lon, max_lon):
    now_utc = datetime.now(timezone.utc)
    start_time = now_utc - timedelta(hours=24)
    steps = 48
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    records = []
    vessels = preset_cfg["vessels"]
    rng = np.random.default_rng(preset_cfg["sar_seed"] + 10)

    for idx, v in enumerate(vessels):
        frac = 0.16 + (idx * 0.22)
        start_pt = [min_lat + lat_span * frac, min_lon + lon_span * 0.05]
        end_pt = [max_lat - lat_span * (0.35 - frac * 0.3), max_lon - lon_span * 0.05]

        if v["gap"]:
            # Route suspect vessel directly through the spill origin
            origin = preset_cfg["origin_point"]
            start_pt = [origin[0] - lat_span * 0.35, origin[1] - lon_span * 0.40]
            end_pt = [origin[0] + lat_span * 0.35, origin[1] + lon_span * 0.40]

        lats = np.linspace(start_pt[0], end_pt[0], steps)
        lons = np.linspace(start_pt[1], end_pt[1], steps)

        for i in range(steps):
            t_point = start_time + timedelta(minutes=30 * i)
            spill_window = 22 <= i <= 26

            if v["gap"] and spill_window:
                spd = v["cpa_spd"]
                if i % 2 == 0:  # Simulated AIS intentional blackout
                    continue
            else:
                spd = v["speed"]

            jitter_lat = float(rng.normal(0, 0.0008))
            jitter_lon = float(rng.normal(0, 0.0008))

            records.append({
                "MMSI": int(v["MMSI"]),
                "BaseDateTime": t_point.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "LAT": round(float(lats[i] + jitter_lat), 5),
                "LON": round(float(lons[i] + jitter_lon), 5),
                "SOG": round(float(spd + rng.normal(0, 0.15)), 1),
                "VesselName": v["name"],
                "VesselType": v["type"],
                "AIS_Gap_Flag": int(v["gap"] and spill_window),
            })

    pd.DataFrame(records).to_csv(DATA_DIR / "sample_ais.csv", index=False)


def generate_dynamic_dataset(min_lat, max_lat, min_lon, max_lon, age_hours=12.0):
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    preset_key, preset_cfg = _match_preset(min_lat, max_lat, min_lon, max_lon)

    # Store real metadata for the selected preset
    metadata = {
        "preset_key": preset_key,
        "preset_name": preset_cfg["name"],
        "centroid": preset_cfg["centroid"],
        "origin_point": preset_cfg["origin_point"],
        "polygon": preset_cfg["polygon"],
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

    _synthesize_realistic_sar(preset_cfg)
    _synthesize_ais_tracks(preset_cfg, min_lat, max_lat, min_lon, max_lon)

    return str(DATA_DIR / "sample_sar.png"), str(DATA_DIR / "sample_ais.csv")
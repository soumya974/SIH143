import os
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def generate_main_dataset(n=1000):
    classes = rng.choice(
        ["no_oil", "lookalike", "low_oil", "high_oil"],
        size=n,
        p=[0.35, 0.25, 0.25, 0.15]
    )

    sea_states = rng.choice(
        ["calm", "moderate", "rough"],
        n,
        p=[0.40, 0.45, 0.15]
    )

    wind = np.clip(rng.normal(6.5, 3.0, n), 0.5, 18)
    wave = np.clip(0.35 + 0.09 * wind + rng.normal(0, 0.25, n), 0.05, 3.5)
    incidence = np.clip(rng.normal(34, 7, n), 20, 48)

    vv_mean = rng.normal(-17.0, 2.2, n)
    vh_mean = rng.normal(-24.0, 2.0, n)

    contrast = np.zeros(n)
    texture = np.zeros(n)
    compactness = np.zeros(n)
    elongation = np.zeros(n)
    dark_fraction = np.zeros(n)
    edge_strength = np.zeros(n)

    for i, c in enumerate(classes):
        if c == "no_oil":
            contrast[i] = rng.normal(1.5, 1.0)
            texture[i] = rng.normal(0.65, 0.22)
            compactness[i] = rng.normal(0.55, 0.18)
            elongation[i] = rng.normal(2.2, 0.9)
            dark_fraction[i] = rng.normal(0.14, 0.07)
            edge_strength[i] = rng.normal(0.55, 0.18)

        elif c == "lookalike":
            contrast[i] = rng.normal(5.2, 1.7)
            texture[i] = rng.normal(0.28, 0.13)
            compactness[i] = rng.normal(0.38, 0.16)
            elongation[i] = rng.normal(5.0, 2.0)
            dark_fraction[i] = rng.normal(0.30, 0.10)
            edge_strength[i] = rng.normal(0.30, 0.12)

        elif c == "low_oil":
            contrast[i] = rng.normal(4.2, 1.3)
            texture[i] = rng.normal(0.32, 0.12)
            compactness[i] = rng.normal(0.58, 0.15)
            elongation[i] = rng.normal(3.5, 1.3)
            dark_fraction[i] = rng.normal(0.27, 0.08)
            edge_strength[i] = rng.normal(0.34, 0.12)

        else:
            contrast[i] = rng.normal(8.0, 2.0)
            texture[i] = rng.normal(0.22, 0.09)
            compactness[i] = rng.normal(0.67, 0.13)
            elongation[i] = rng.normal(3.0, 1.1)
            dark_fraction[i] = rng.normal(0.45, 0.12)
            edge_strength[i] = rng.normal(0.25, 0.10)

    calm = sea_states == "calm"
    rough = sea_states == "rough"

    contrast += calm * rng.normal(1.0, 0.5, n)
    dark_fraction += calm * rng.normal(0.05, 0.025, n)
    texture += rough * rng.normal(0.12, 0.05, n)

    overlap = np.isin(classes, ["lookalike", "low_oil", "high_oil"])
    contrast[overlap] += rng.normal(0, 0.7, overlap.sum())

    contrast = np.clip(contrast, 0.05, 15)
    texture = np.clip(texture, 0.03, 1.5)
    compactness = np.clip(compactness, 0.05, 1.0)
    elongation = np.clip(elongation, 1.0, 12)
    dark_fraction = np.clip(dark_fraction, 0.01, 0.90)
    edge_strength = np.clip(edge_strength, 0.03, 1.0)

    vessel_present = rng.binomial(1, 0.58, n)
    vessel_distance = np.where(
        vessel_present == 1,
        np.clip(rng.gamma(2.2, 3.0, n), 0.2, 30),
        np.nan
    )

    ais_gap = np.zeros(n, dtype=int)
    speed_drop = np.zeros(n)
    course_change = np.zeros(n)

    for i, c in enumerate(classes):
        if c in ["low_oil", "high_oil"]:
            ais_gap[i] = rng.binomial(1, 0.35)
            speed_drop[i] = np.clip(rng.normal(0.35, 0.18), 0, 0.95)
            course_change[i] = np.clip(abs(rng.normal(18, 12)), 0, 90)
        elif c == "lookalike":
            ais_gap[i] = rng.binomial(1, 0.12)
            speed_drop[i] = np.clip(rng.normal(0.12, 0.10), 0, 0.70)
            course_change[i] = np.clip(abs(rng.normal(10, 9)), 0, 70)
        else:
            ais_gap[i] = rng.binomial(1, 0.08)
            speed_drop[i] = np.clip(rng.normal(0.08, 0.08), 0, 0.60)
            course_change[i] = np.clip(abs(rng.normal(8, 8)), 0, 60)

    area = np.zeros(n)
    for i, c in enumerate(classes):
        if c == "no_oil":
            area[i] = np.clip(rng.lognormal(-1.5, 0.7), 0.02, 5)
        elif c == "lookalike":
            area[i] = np.clip(rng.lognormal(-0.8, 0.8), 0.03, 15)
        elif c == "low_oil":
            area[i] = np.clip(rng.lognormal(-0.5, 0.7), 0.05, 20)
        else:
            area[i] = np.clip(rng.lognormal(1.0, 0.65), 0.5, 80)

    perimeter = np.clip(
        2 * np.sqrt(np.pi * area) * rng.normal(1.8, 0.25, n),
        0.1, 100
    )

    rain = np.clip(rng.beta(1.5, 5, n), 0, 1)
    sst = rng.normal(27.5, 2.2, n)
    current_speed = np.clip(rng.normal(0.55, 0.25, n), 0.02, 1.8)
    current_direction = rng.uniform(0, 360, n)

    oil_present = np.isin(classes, ["low_oil", "high_oil"]).astype(int)
    severity = pd.Series(classes).map({
        "no_oil": 0, "lookalike": 0, "low_oil": 1, "high_oil": 2
    }).to_numpy()

    detection_score = np.clip(
        0.22 * np.clip(contrast / 10, 0, 1) + 
        0.18 * np.clip(dark_fraction / 0.6, 0, 1) + 
        0.16 * np.clip(1 - texture, 0, 1) + 
        0.12 * compactness + 
        0.10 * np.clip(1 / elongation, 0, 1) + 
        0.08 * speed_drop + 
        0.07 * ais_gap + 
        0.07 * np.clip(1 / (1 + np.nan_to_num(vessel_distance, nan=30)), 0, 1) +
        rng.normal(0, 0.08, n),
        0, 1
    )

    df = pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "class": classes,
        "oil_present": oil_present,
        "oil_severity": severity,
        "sea_state": sea_states,
        "wind_speed_mps": np.round(wind, 2),
        "wave_height_m": np.round(wave, 2),
        "rain_probability": np.round(rain, 3),
        "sea_surface_temperature_c": np.round(sst, 2),
        "current_speed_mps": np.round(current_speed, 2),
        "current_direction_deg": np.round(current_direction, 1),
        "incidence_angle_deg": np.round(incidence, 2),
        "sar_vv_mean_db": np.round(vv_mean, 2),
        "sar_vh_mean_db": np.round(vh_mean, 2),
        "sar_contrast_db": np.round(contrast, 2),
        "sar_texture": np.round(texture, 3),
        "dark_pixel_fraction": np.round(dark_fraction, 3),
        "edge_strength": np.round(edge_strength, 3),
        "shape_compactness": np.round(compactness, 3),
        "shape_elongation": np.round(elongation, 2),
        "slick_area_km2": np.round(area, 3),
        "slick_perimeter_km": np.round(perimeter, 3),
        "vessel_present_nearby": vessel_present,
        "vessel_distance_km": np.round(vessel_distance, 2),
        "ais_gap_flag": ais_gap,
        "speed_drop_fraction": np.round(speed_drop, 3),
        "course_change_deg": np.round(course_change, 1),
        "detection_score": np.round(detection_score, 3)
    })

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def generate_spatial_dataset(main_df, min_lat, max_lat, min_lon, max_lon):
    n = len(main_df)
    locations = rng.choice(
        ["open_ocean", "near_coast", "island_adjacent", "inside_land"],
        n, p=[0.42, 0.28, 0.22, 0.08]
    )

    latitude = rng.uniform(min_lat, max_lat, n)
    longitude = rng.uniform(min_lon, max_lon, n)

    distance_land = np.zeros(n)
    distance_coast = np.zeros(n)
    land_fraction = np.zeros(n)
    island_fraction = np.zeros(n)

    for i, loc in enumerate(locations):
        if loc == "open_ocean":
            distance_land[i] = rng.uniform(25, 250)
            distance_coast[i] = rng.uniform(25, 250)
        elif loc == "near_coast":
            distance_land[i] = rng.uniform(0.5, 15)
            distance_coast[i] = rng.uniform(0.2, 12)
            land_fraction[i] = rng.uniform(0, 0.35)
            island_fraction[i] = rng.uniform(0, 0.08)
        elif loc == "island_adjacent":
            distance_land[i] = rng.uniform(0.2, 8)
            distance_coast[i] = rng.uniform(0.1, 7)
            land_fraction[i] = rng.uniform(0, 0.45)
            island_fraction[i] = rng.uniform(0.02, 0.50)
        else:
            distance_land[i] = 0
            distance_coast[i] = 0
            land_fraction[i] = rng.uniform(0.60, 1.0)
            island_fraction[i] = rng.uniform(0, 0.15)

    land_risk = np.clip(1 - distance_land / 25 + rng.normal(0, 0.08, n), 0, 1)
    coastal_fp = ((locations != "open_ocean") & (rng.random(n) < 0.45)).astype(int)

    return pd.DataFrame({
        "record_id": main_df["record_id"],
        "location_context": locations,
        "latitude": np.round(latitude, 5),
        "longitude": np.round(longitude, 5),
        "distance_to_land_km": np.round(distance_land, 3),
        "distance_to_coast_km": np.round(distance_coast, 3),
        "land_fraction_in_patch": np.round(land_fraction, 3),
        "island_fraction_in_patch": np.round(island_fraction, 3),
        "land_contamination_risk": np.round(land_risk, 3),
        "coastal_false_positive": coastal_fp
    })


def generate_dynamic_dataset(min_lat, max_lat, min_lon, max_lon, age_hours=12.0, use_real_ais=False):
    """
    Core function called by app.py to build bounded synthetic inputs matching statistical rigor.
    """
    os.makedirs("data", exist_ok=True)
    
    # Generate underlying statistical master table
    main_df = generate_main_dataset(n=250)
    spatial_df = generate_spatial_dataset(main_df, min_lat, max_lat, min_lon, max_lon)
    
    master_df = main_df.merge(spatial_df, on="record_id", how="left")
    master_df.to_csv(DATA_DIR / "master_oil_spill_dataset_250.csv", index=False)

    # Derive representative centroid from dataset bounds
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
            round(slick_lon + r_lon * np.cos(a), 5)
        ])

    spill_meta = {
        "centroid": [round(slick_lat, 5), round(slick_lon, 5)],
        "origin_point": [round(center_lat, 5), round(center_lon, 5)],
        "polygon": polygon,
        "estimated_age_hours": age_hours
    }
    
    with open(DATA_DIR / "spill_metadata.json", "w") as f:
        json.dump(spill_meta, f, indent=2)

    # Dual-Pol SAR Image Simulation (VV and VH)
    h, w = 512, 512
    vv_ocean = rng.gamma(shape=4.0, scale=130 / 4.0, size=(h, w))
    vh_ocean = rng.gamma(shape=2.0, scale=45 / 2.0, size=(h, w))
    
    slick_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(slick_mask, (256, 256), (90, 45), 35, 0, 360, 255, -1)
    slick_mask = cv2.GaussianBlur(slick_mask, (21, 21), 7)
    
    vv_img = np.clip(np.where(slick_mask > 50, vv_ocean * 0.25, vv_ocean), 0, 255).astype(np.uint8)
    vh_img = np.clip(np.where(slick_mask > 50, vh_ocean * 0.80, vh_ocean), 0, 255).astype(np.uint8)

    cv2.imwrite(str(DATA_DIR / "sample_sar_vv.png"), vv_img)
    cv2.imwrite(str(DATA_DIR / "sample_sar_vh.png"), vh_img)
    cv2.imwrite(str(DATA_DIR / "sample_sar.png"), cv2.merge([vv_img, vh_img, vv_img]))

    # AIS Incident Candidates & Traffic Integration
    now_utc = datetime.now(timezone.utc)
    start_time = now_utc - timedelta(hours=24)
    steps = 48
    records = []

    vessels = [
        {
            "MMSI": 413210001,
            "VesselName": "MT OCEAN MARAUDER",
            "VesselType": "Crude Oil Tanker",
            "start": [min_lat, min_lon],
            "end": [max_lat, max_lon],
            "speed": 13.8,
            "gap": True
        },
        {
            "MMSI": 211567000,
            "VesselName": "MV POSEIDON ALPHA",
            "VesselType": "Container Ship",
            "start": [min_lat, center_lon + (lon_span * 0.2)],
            "end": [max_lat, center_lon + (lon_span * 0.2)],
            "speed": 18.5,
            "gap": False
        },
        {
            "MMSI": 354890000,
            "VesselName": "PACIFIC TRADER",
            "VesselType": "Bulk Carrier",
            "start": [max_lat, min_lon],
            "end": [min_lat, max_lon],
            "speed": 12.0,
            "gap": False
        }
    ]

    for vessel in vessels:
        lats = np.linspace(vessel["start"][0], vessel["end"][0], steps)
        lons = np.linspace(vessel["start"][1], vessel["end"][1], steps)

        for i in range(steps):
            time = start_time + timedelta(minutes=30 * i)
            spill_window = 22 <= i <= 26

            if vessel["gap"] and spill_window and i % 2 == 0:
                continue

            speed = vessel["speed"]
            if vessel["gap"] and spill_window:
                speed = 4.5

            records.append({
                "MMSI": vessel["MMSI"],
                "BaseDateTime": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "LAT": round(float(lats[i] + rng.normal(0, 0.002)), 5),
                "LON": round(float(lons[i] + rng.normal(0, 0.002)), 5),
                "SOG": round(speed + rng.normal(0, 0.5), 1),
                "VesselName": vessel["VesselName"],
                "VesselType": vessel["VesselType"],
                "AIS_Gap_Flag": int(vessel["gap"])
            })

    pd.DataFrame(records).to_csv(DATA_DIR / "sample_ais.csv", index=False)
    return str(DATA_DIR / "sample_sar.png"), str(DATA_DIR / "sample_ais.csv")
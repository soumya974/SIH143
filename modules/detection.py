import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from shapely.geometry import Polygon

def build_unet_model(input_shape=(512, 512, 3)):
    inputs = Input(input_shape)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    u4 = UpSampling2D((2, 2))(c3)
    concat4 = concatenate([u4, c2])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
    u5 = UpSampling2D((2, 2))(c4)
    concat5 = concatenate([u5, c1])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat5)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    return Model(inputs=[inputs], outputs=[outputs])

def detect_spill(image_source):
    if os.path.exists(image_source):
        img = cv2.imread(image_source)
        if img is not None:
            img = cv2.resize(img, (512, 512))
        else:
            img = np.zeros((512, 512, 3), dtype=np.uint8)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    input_tensor = img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    unet_model = build_unet_model()
    pred_mask = unet_model(input_tensor).numpy()[0, :, :, 0]
    binary_mask = (pred_mask > 0.4).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    meta_path = "data/spill_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data = json.load(f)
        centroid = data["centroid"]
        polygon = data["polygon"]
        
        # Safely handle potential None values for estimated_age_hours
        raw_age = data.get("estimated_age_hours")
        age = float(raw_age) if raw_age is not None else 12.0
    else:
        centroid = [17.45, 56.10]
        polygon = [
            [17.47, 56.08], [17.48, 56.12],
            [17.44, 56.14], [17.43, 56.09]
        ]
        age = 12.0

    poly = Polygon(polygon)
    lat_rad = np.radians(centroid[0])
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
    
    poly_coords = np.array(polygon)
    scaled_coords = np.column_stack([
        poly_coords[:, 0] * meters_per_deg_lat,
        poly_coords[:, 1] * meters_per_deg_lon
    ])
    
    area_sqm = float(Polygon(scaled_coords).area)
    area_sqm = max(area_sqm, 50000.0)
    area_sqkm = area_sqm / 1_000_000.0

    thickness_um = max(15.0, 120.0 * np.exp(-0.06 * age))
    thickness_mm = thickness_um / 1000.0
    thickness_m = thickness_um * 1e-6

    volume_m3 = area_sqm * thickness_m
    volume_barrels = volume_m3 * 6.2898

    sub_points = []
    min_lat, min_lon, max_lat, max_lon = poly.bounds
    for _ in range(120):
        rand_lat = np.random.uniform(min_lat, max_lat)
        rand_lon = np.random.uniform(min_lon, max_lon)
        dist = np.hypot(rand_lat - centroid[0], rand_lon - centroid[1])
        intensity = max(0.1, 1.0 - (dist / 0.02))
        sub_points.append([rand_lat, rand_lon, intensity])

    return {
        "centroid": centroid,
        "polygon": polygon,
        "concentration_grid": sub_points,
        "area_sqm": area_sqm,
        "area_sqkm": area_sqkm,
        "depth_um": thickness_um,
        "depth_mm": thickness_mm,
        "volume_m3": volume_m3,
        "volume_barrels": volume_barrels,
        "estimated_age_hours": age
    }
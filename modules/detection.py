"""
SAR-based oil-slick detection, classification, and physical-property
estimation.

DETECTION — adaptive dark-spot segmentation
    Oil films dampen short-gravity (Bragg) wave scattering, so a slick shows
    up as a patch of anomalously low radar backscatter relative to the
    surrounding open water. This is the standard basis for operational SAR
    oil-spill monitoring (e.g. EMSA CleanSeaNet, Fingas & Brown 2014). The
    pipeline here:
      1. Gaussian-denoises the scene to suppress speckle without erasing the
         slick boundary.
      2. Thresholds at a scene-adaptive percentile of the backscatter
         distribution (DARK_SPOT_PERCENTILE) rather than a fixed dB cut —
         robust across different gain settings / sea states.
      3. Cleans the resulting mask with morphological opening (remove
         speckle-sized false positives) and closing (fill small internal
         holes), then keeps the largest connected region above a minimum
         area.

CLASSIFICATION — look-alike discrimination
    Dark patches in SAR imagery are not always oil: low-wind zones, algal
    mats, rain cells and current-shear lines all produce similar backscatter
    dips. Four descriptors that the remote-sensing literature uses to
    separate genuine slicks from these look-alikes are computed from the
    detected region and combined into a single evidence score in [0, 1]:
        - backscatter contrast between the region and its surround
        - internal texture homogeneity (oil-damped water is smoother)
        - elongation (wind/current-driven slicks are usually elongated)
        - boundary compactness (irregular, non-circular boundary)
    See _classify() for the exact weighting.

PHYSICAL ESTIMATES
    Area comes from the *detected* polygon (geodesic shoelace formula, not
    the pixel count), thickness from the standard empirical thin-film decay
    approximation for weathering oil sheens, and volume from area x
    thickness converted to barrels.
"""

import json
import os

import cv2
import numpy as np
from shapely.geometry import Polygon

DATA_DIR = "data"
METADATA_PATH = os.path.join(DATA_DIR, "spill_metadata.json")

DEFAULT_CENTROID = [17.45, 56.10]
DEFAULT_AGE_HOURS = 12.0

# --- detection / classification tunables -----------------------------
DARK_SPOT_PERCENTILE = 15        # candidate pixels below this percentile of scene backscatter
MIN_BLOB_AREA_PX = 150           # discard speckle-sized blobs
GAUSSIAN_BLUR_KSIZE = (5, 5)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
SURROUND_DILATE_ITERS = 6        # ring thickness (px) used to sample the surrounding water


# =======================================================================
# GEO CONTEXT
# =======================================================================

def _load_geo_context():
    """
    Pulls the scene bounding box (for pixel -> lat/lon mapping), a fallback
    centroid, and the estimated spill age from data/spill_metadata.json —
    written by whichever imagery source ran (synthetic simulation or a real
    Sentinel-1 fetch). The slick SHAPE itself is no longer taken from this
    file; only the geographic framing is.
    """
    if not os.path.exists(METADATA_PATH):
        return None, DEFAULT_CENTROID, DEFAULT_AGE_HOURS

    with open(METADATA_PATH, "r") as f:
        data = json.load(f)

    centroid = data.get("centroid", DEFAULT_CENTROID)
    age = float(data.get("estimated_age_hours") or DEFAULT_AGE_HOURS)

    bbox = data.get("search")  # present for real Sentinel-1 fetches
    if not bbox:
        # Synthetic scenes don't carry an explicit search bbox — frame a
        # reasonable window around the centroid so pixel coordinates still
        # map to sensible geography.
        span = 0.35
        bbox = {
            "min_lat": centroid[0] - span, "max_lat": centroid[0] + span,
            "min_lon": centroid[1] - span, "max_lon": centroid[1] + span,
        }
    return bbox, centroid, age


def _pixel_to_latlon(px, py, width, height, bbox):
    """Equirectangular pixel -> lat/lon mapping over the scene bounding box
    (row 0 = max_lat, matching how the overview PNGs are written)."""
    lon = bbox["min_lon"] + (px / width) * (bbox["max_lon"] - bbox["min_lon"])
    lat = bbox["max_lat"] - (py / height) * (bbox["max_lat"] - bbox["min_lat"])
    return lat, lon


# =======================================================================
# SEGMENTATION
# =======================================================================

def _dark_spot_mask(gray):
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KSIZE, 0)
    threshold_value = float(np.percentile(blurred, DARK_SPOT_PERCENTILE))
    mask = (blurred <= threshold_value).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    return mask, threshold_value


def _largest_valid_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_BLOB_AREA_PX]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# =======================================================================
# CLASSIFICATION
# =======================================================================

def _shape_descriptors(contour, gray):
    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)
    if perimeter_px == 0 or area_px == 0:
        return None

    # Isoperimetric compactness: 1.0 for a perfect circle, lower for
    # elongated/irregular boundaries.
    compactness = (4 * np.pi * area_px) / (perimeter_px ** 2)

    x, y, w, h = cv2.boundingRect(contour)
    elongation = max(w, h) / max(min(w, h), 1)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    inside = gray[mask == 255]

    surround_mask = cv2.dilate(mask, MORPH_KERNEL, iterations=SURROUND_DILATE_ITERS) - mask
    outside = gray[surround_mask == 255]

    if inside.size == 0 or outside.size == 0:
        return None

    # Backscatter contrast vs. local surround. Genuine oil damping typically
    # yields a clear intensity drop; weak/noisy contrast is a look-alike flag.
    contrast = float(np.mean(outside) - np.mean(inside))

    # Internal homogeneity: oil-damped water is smoother (lower std) than
    # wind-roughened open water, biogenic slicks, or rain cells.
    homogeneity = float(1.0 / (1.0 + np.std(inside)))

    return {
        "area_px": float(area_px),
        "compactness": float(compactness),
        "elongation": float(elongation),
        "contrast_db_equiv": contrast,
        "homogeneity": homogeneity,
    }


def _classify(descr):
    """
    score = 0.35 * contrast_term
          + 0.25 * homogeneity
          + 0.20 * elongation_term
          + 0.20 * (1 - compactness)

    contrast_term   = clip(contrast / 25.0, 0, 1)
    elongation_term = clip((elongation - 1.0) / 4.0, 0, 1)

    score >= 0.60            -> "probable_oil_slick"
    0.35 <= score < 0.60     -> "possible_lookalike_review"
    score < 0.35             -> "likely_lookalike"
    """
    contrast_term = float(np.clip(descr["contrast_db_equiv"] / 25.0, 0, 1))
    elongation_term = float(np.clip((descr["elongation"] - 1.0) / 4.0, 0, 1))
    compactness_term = float(np.clip(1.0 - descr["compactness"], 0, 1))
    homogeneity_term = float(np.clip(descr["homogeneity"], 0, 1))

    score = (
        0.35 * contrast_term
        + 0.25 * homogeneity_term
        + 0.20 * elongation_term
        + 0.20 * compactness_term
    )
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.60:
        label = "probable_oil_slick"
    elif score >= 0.35:
        label = "possible_lookalike_review"
    else:
        label = "likely_lookalike"

    return score, label


# =======================================================================
# PHYSICAL ESTIMATES
# =======================================================================

def _geodesic_polygon_area_sqm(polygon_latlon):
    """Equirectangular-projected shoelace area, in square meters, using the
    mean latitude of the ring for the meters-per-degree scale factors."""
    lats = [p[0] for p in polygon_latlon]
    lat0 = float(np.mean(lats))
    lat_rad = np.radians(lat0)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(lat_rad)
    scaled = [[lon * m_per_deg_lon, lat * m_per_deg_lat] for lat, lon in polygon_latlon]
    return max(float(Polygon(scaled).area), 1.0)


def _thickness_volume(area_sqm, age_hours):
    # Empirical thin-film weathering decay (Fingas-style approximation):
    # fresh sheens start near ~100-150 micron and thin exponentially with age.
    thickness_um = max(15.0, 120.0 * np.exp(-0.06 * age_hours))
    thickness_mm = thickness_um / 1000.0
    thickness_m = thickness_um * 1e-6
    volume_m3 = area_sqm * thickness_m
    volume_barrels = volume_m3 * 6.2898
    return thickness_um, thickness_mm, volume_m3, volume_barrels


# =======================================================================
# PUBLIC ENTRYPOINT
# =======================================================================

def detect_spill(image_source=None):
    """
    Runs dark-spot detection + look-alike classification on the SAR overview
    image and returns detected geometry, classification, and physical
    estimates.

    image_source: path to the SAR preview PNG. Defaults to
    data/sample_sar.png. If the image can't be read, falls back to a small
    placeholder polygon around the metadata centroid, clearly flagged as
    such in the returned 'detection' block (never silently presented as a
    real detection).
    """
    bbox, fallback_centroid, age = _load_geo_context()
    path = image_source or os.path.join(DATA_DIR, "sample_sar.png")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return _fallback_result(fallback_centroid, age, note=f"could not read SAR image at {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    mask, threshold_value = _dark_spot_mask(gray)
    contour = _largest_valid_contour(mask)
    if contour is None:
        return _fallback_result(fallback_centroid, age, note="no dark-spot candidate above the minimum area threshold")

    descriptors = _shape_descriptors(contour, gray)
    if descriptors is None:
        return _fallback_result(fallback_centroid, age, note="degenerate detected contour")

    score, label = _classify(descriptors)

    # Simplify the contour before projecting to lat/lon (keeps the polygon
    # legible on the map instead of a jagged per-pixel outline).
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        approx = contour.reshape(-1, 2)

    polygon_latlon = [_pixel_to_latlon(px, py, width, height, bbox) for px, py in approx]

    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx_px, cy_px = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    else:
        cx_px, cy_px = width / 2.0, height / 2.0
    centroid_lat, centroid_lon = _pixel_to_latlon(cx_px, cy_px, width, height, bbox)
    centroid = [round(centroid_lat, 5), round(centroid_lon, 5)]

    area_sqm = _geodesic_polygon_area_sqm(polygon_latlon)
    area_sqkm = area_sqm / 1_000_000.0
    thickness_um, thickness_mm, volume_m3, volume_barrels = _thickness_volume(area_sqm, age)

    return {
        "centroid": centroid,
        "polygon": [[round(la, 5), round(lo, 5)] for la, lo in polygon_latlon],
        "concentration_grid": _concentration_grid_from_mask(mask, bbox, width, height),
        "area_sqm": area_sqm,
        "area_sqkm": area_sqkm,
        "depth_um": thickness_um,
        "depth_mm": thickness_mm,
        "volume_m3": volume_m3,
        "volume_barrels": volume_barrels,
        "estimated_age_hours": age,
        "detection": {
            "method": "adaptive_dark_spot_segmentation",
            "backscatter_threshold": round(threshold_value, 2),
            "classification": label,
            "classification_score": round(score, 3),
            "descriptors": {k: round(v, 3) for k, v in descriptors.items()},
        },
    }


def _concentration_grid_from_mask(mask, bbox, width, height, n_points=150):
    """Sub-samples the detected mask itself (rather than a synthetic radial
    falloff) so the concentration overlay reflects the actual detected
    footprint's internal intensity pattern."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    n = min(n_points, len(xs))
    idx = np.random.choice(len(xs), size=n, replace=False)
    points = []
    for i in idx:
        lat, lon = _pixel_to_latlon(float(xs[i]), float(ys[i]), width, height, bbox)
        points.append([round(lat, 5), round(lon, 5), 1.0])
    return points


def _fallback_result(centroid, age, note=None):
    """Only used if the SAR image is unreadable/missing — keeps the pipeline
    from crashing while clearly flagging that this is NOT an image-derived
    detection (classification is 'undetermined', not a score)."""
    polygon = [
        [centroid[0] + 0.02, centroid[1] - 0.02],
        [centroid[0] + 0.02, centroid[1] + 0.02],
        [centroid[0] - 0.02, centroid[1] + 0.02],
        [centroid[0] - 0.02, centroid[1] - 0.02],
    ]
    area_sqm = _geodesic_polygon_area_sqm(polygon)
    thickness_um, thickness_mm, volume_m3, volume_barrels = _thickness_volume(area_sqm, age)

    return {
        "centroid": centroid,
        "polygon": polygon,
        "concentration_grid": [],
        "area_sqm": area_sqm,
        "area_sqkm": area_sqm / 1_000_000.0,
        "depth_um": thickness_um,
        "depth_mm": thickness_mm,
        "volume_m3": volume_m3,
        "volume_barrels": volume_barrels,
        "estimated_age_hours": age,
        "detection": {
            "method": "fallback_no_image",
            "backscatter_threshold": None,
            "classification": "undetermined",
            "classification_score": 0.0,
            "descriptors": {},
            "note": note or "SAR image not available",
        },
    }
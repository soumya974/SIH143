import os
import json
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import rasterio

from eodag import EODataAccessGateway


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "eodag": {
        "config_path": Path.home() / ".config" / "eodag" / "eodag.yml",
        "provider": "cop_dataspace",
        "collection": "S1_SAR_GRD",
    },

    "output": {
        "directory": Path("data"),
        "overview_image": "sample_sar.png",
        "metadata": "spill_metadata.json",
    },

    "processing": {
        "width": 512,
        "height": 512,
        "equalize_histogram": True,
    },
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# EODAG CLIENT (ROBUST INITIALIZATION)
# ============================================================

def create_eodag_client(username=None, password=None):
    """
    Create an EODAG client safely, supporting both default environments
    and custom configuration files.

    If username/password are provided, they are injected as EODAG's
    documented environment-variable credential override
    (EODAG__<PROVIDER>__AUTH__CREDENTIALS__USERNAME/PASSWORD) before the
    gateway is constructed, so they take precedence over — or fill in for —
    any missing entries in the on-disk eodag.yml config.
    """
    config_path = CONFIG["eodag"]["config_path"]
    provider = CONFIG["eodag"]["provider"]

    if username:
        os.environ[f"EODAG__{provider.upper()}__AUTH__CREDENTIALS__USERNAME"] = username
    if password:
        os.environ[f"EODAG__{provider.upper()}__AUTH__CREDENTIALS__PASSWORD"] = password

    if config_path.exists():
        logger.info("Loading EODAG configuration from: %s", config_path)
        dag = EODataAccessGateway(user_conf_file_path=str(config_path))
    else:
        logger.info("Custom config not found at %s. Initializing with default EODAG settings.", config_path)
        dag = EODataAccessGateway()

    dag.set_preferred_provider(provider)
    logger.info("EODAG provider set to: %s", provider)

    return dag


# ============================================================
# SENTINEL-1 SEARCH
# ============================================================

def search_sentinel1(
    dag,
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    start_date,
    end_date,
    items_per_page=1,
):
    collection = CONFIG["eodag"]["collection"]
    provider = CONFIG["eodag"]["provider"]

    geometry = {
        "lonmin": float(min_lon),
        "latmin": float(min_lat),
        "lonmax": float(max_lon),
        "latmax": float(max_lat),
    }

    start = start_date.strftime("%Y-%m-%dT00:00:00")
    end = end_date.strftime("%Y-%m-%dT23:59:59")

    logger.info("Searching Sentinel-1...")
    logger.info("BBox: %.5f, %.5f -> %.5f, %.5f", min_lon, min_lat, max_lon, max_lat)
    logger.info("Start: %s | End: %s", start, end)

    # items_per_page caps how many Sentinel-1 products EODAG returns for this
    # search — configurable from the sidebar (EODAG retrieval settings),
    # defaults to 1.
    items_per_page = max(1, int(items_per_page))
    logger.info("Items per page: %d", items_per_page)

    products = dag.search(
        collection=collection,
        geometry=geometry,
        start=start,
        end=end,
        provider=provider,
        items_per_page=items_per_page,
    )

    logger.info("Found %d Sentinel-1 products", len(products) if products else 0)
    return products


# ============================================================
# DOWNLOAD
# ============================================================

def download_product(dag, products):
    if not products:
        raise ValueError("No Sentinel-1 products found for the specified bounds and date range.")

    product = products[0]
    logger.info("Selected product ID: %s", getattr(product, "id", "Unknown"))
    logger.info("Downloading Sentinel-1 product...")

    downloaded_path = dag.download(product)
    logger.info("Downloaded to: %s", downloaded_path)

    return Path(downloaded_path)


# ============================================================
# FIND VV / VH
# ============================================================

def find_polarization_files(product_path):
    product_path = Path(product_path)
    vv_files = []
    vh_files = []

    for file in product_path.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in {".tif", ".tiff"}:
            continue

        name = file.name.lower()
        if "-vv-" in name or "_vv_" in name:
            vv_files.append(file)
        elif "-vh-" in name or "_vh_" in name:
            vh_files.append(file)

    logger.info("VV files found: %d | VH files found: %d", len(vv_files), len(vh_files))

    return (
        vv_files[0] if vv_files else None,
        vh_files[0] if vh_files else None,
    )


# ============================================================
# READ RASTER & PROCESSING
# ============================================================

def read_raster(path, width, height):
    with rasterio.open(path) as src:
        image = src.read(
            1,
            out_shape=(height, width),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        transform = src.transform

    return image.astype(np.float32), transform


def normalize_image(image):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    minimum, maximum = np.min(image), np.max(image)

    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    if CONFIG["processing"]["equalize_histogram"]:
        normalized = cv2.equalizeHist(normalized)

    return normalized


def create_sar_overview(vv_file, vh_file, output_path):
    width = CONFIG["processing"]["width"]
    height = CONFIG["processing"]["height"]

    vv, _ = read_raster(vv_file, width, height)
    vh, _ = read_raster(vh_file, width, height)

    vv_8 = normalize_image(vv)
    vh_8 = normalize_image(vh)
    ratio = vv - vh
    ratio_8 = normalize_image(ratio)

    composite = cv2.merge([vh_8, ratio_8, vv_8])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), composite)
    if not success:
        raise IOError(f"Failed to save overview image: {output_path}")

    logger.info("SAR overview saved: %s", output_path)
    return output_path


# ============================================================
# METADATA
# ============================================================

def create_metadata(min_lat, max_lat, min_lon, max_lon, start_date, end_date, product, output_path):
    center_lat = (float(min_lat) + float(max_lat)) / 2.0
    center_lon = (float(min_lon) + float(max_lon)) / 2.0
    lat_span = float(max_lat) - float(min_lat)
    lon_span = float(max_lon) - float(min_lon)

    polygon = [
        [round(center_lat + lat_span * 0.1, 5), round(center_lon - lon_span * 0.1, 5)],
        [round(center_lat + lat_span * 0.1, 5), round(center_lon + lon_span * 0.1, 5)],
        [round(center_lat - lat_span * 0.1, 5), round(center_lon + lon_span * 0.1, 5)],
        [round(center_lat - lat_span * 0.1, 5), round(center_lon - lon_span * 0.1, 5)],
    ]

    metadata = {
        "centroid": [round(center_lat, 5), round(center_lon, 5)],
        "polygon": polygon,
        "search": {
            "min_lat": float(min_lat),
            "max_lat": float(max_lat),
            "min_lon": float(min_lon),
            "max_lon": float(max_lon),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "sentinel": {
            "mission": "Sentinel-1",
            "collection": CONFIG["eodag"]["collection"],
            "provider": CONFIG["eodag"]["provider"],
        },
        "product": {
            "id": getattr(product, "id", None),
            "title": getattr(product, "properties", {}).get("title", None),
        },
        "processing": {
            "overview_width": CONFIG["processing"]["width"],
            "overview_height": CONFIG["processing"]["height"],
            "equalize_histogram": CONFIG["processing"]["equalize_histogram"],
        },
        "estimated_age_hours": 12.0,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved: %s", output_path)
    return output_path


# ============================================================
# PIPELINE ENTRYPOINT
# ============================================================

def fetch_and_preprocess_sentinel(min_lat, max_lat, min_lon, max_lon, start_date, end_date,
                                   username=None, password=None, items_per_page=1):
    """
    username/password: optional Copernicus Dataspace credentials. If omitted,
    EODAG falls back to whatever is already configured in eodag.yml or in the
    process environment.
    items_per_page: maximum number of Sentinel-1 products to request from
    EODAG for this search. Defaults to 1.
    """
    output_dir = CONFIG["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_path = output_dir / CONFIG["output"]["overview_image"]
    metadata_path = output_dir / CONFIG["output"]["metadata"]

    dag = create_eodag_client(username=username, password=password)

    products = search_sentinel1(
        dag=dag,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        start_date=start_date,
        end_date=end_date,
        items_per_page=items_per_page,
    )

    if not products:
        raise ValueError("No Sentinel-1 products found for the given bounding box and date range.")

    downloaded_product = download_product(dag, products)

    vv_file, vh_file = find_polarization_files(downloaded_product)
    if not vv_file or not vh_file:
        raise FileNotFoundError("Could not locate Sentinel-1 VV and/or VH measurement rasters.")

    create_sar_overview(vv_file, vh_file, overview_path)
    create_metadata(min_lat, max_lat, min_lon, max_lon, start_date, end_date, products[0], metadata_path)

    return {
        "product": downloaded_product,
        "vv": vv_file,
        "vh": vh_file,
        "overview": overview_path,
        "metadata": metadata_path,
    }
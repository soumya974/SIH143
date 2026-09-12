"""
Sentinel-1 SAR data loader using EODAG with safe fallbacks.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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


def fetch_and_preprocess_sentinel(min_lat, max_lat, min_lon, max_lon, start_date, end_date,
                                   username=None, password=None, items_per_page=1):
    output_dir = CONFIG["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / CONFIG["output"]["overview_image"]
    metadata_path = output_dir / CONFIG["output"]["metadata"]

    try:
        from eodag import EODataAccessGateway
        if username:
            os.environ["EODAG__COP_DATASPACE__AUTH__CREDENTIALS__USERNAME"] = username
        if password:
            os.environ["EODAG__COP_DATASPACE__AUTH__CREDENTIALS__PASSWORD"] = password

        config_path = CONFIG["eodag"]["config_path"]
        if config_path.exists():
            dag = EODataAccessGateway(user_conf_file_path=str(config_path))
        else:
            dag = EODataAccessGateway()
        dag.set_preferred_provider("cop_dataspace")

        geometry = {
            "lonmin": float(min_lon), "latmin": float(min_lat),
            "lonmax": float(max_lon), "latmax": float(max_lat),
        }
        products = dag.search(
            collection="S1_SAR_GRD",
            geometry=geometry,
            start=start_date.strftime("%Y-%m-%dT00:00:00"),
            end=end_date.strftime("%Y-%m-%dT23:59:59"),
            provider="cop_dataspace",
            items_per_page=max(1, int(items_per_page)),
        )
        if products:
            downloaded = dag.download(products[0])
            return {
                "product": downloaded,
                "overview": overview_path,
                "metadata": metadata_path,
            }
    except Exception as exc:
        logger.info("EODAG retrieval fallback: %s", exc)

    return {
        "product": None,
        "overview": overview_path,
        "metadata": metadata_path,
    }
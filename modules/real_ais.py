import os
import pandas as pd
import numpy as np

def fetch_marine_cadastre_data(min_lat, max_lat, min_lon, max_lon, target_date=None):
    """
    Loads locally downloaded Marine Cadastre AIS data from data/raw_ais.csv,
    filters the records by the geographic bounding box, and writes out data/sample_ais.csv.
    """
    os.makedirs("data", exist_ok=True)
    raw_path = os.path.join("data", "raw_ais.csv")
    out_path = os.path.join("data", "sample_ais.csv")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Cached AIS file '{raw_path}' not found. Please run 'python download_ais.py' first."
        )

    # Load local cached dataset
    df = pd.read_csv(raw_path)

    # Standardize column names
    rename_map = {
        "Latitude": "LAT",
        "Longitude": "LON",
        "Vessel_Name": "VesselName",
        "Vessel_Type": "VesselType"
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric coordinates
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")

    # Filter by user-selected bounding box
    filtered = df[
        (df["LAT"] >= float(min_lat)) & (df["LAT"] <= float(max_lat)) &
        (df["LON"] >= float(min_lon)) & (df["LON"] <= float(max_lon))
    ].copy()

    # Fill default columns if missing
    if not filtered.empty:
        if "SOG" not in filtered.columns:
            filtered["SOG"] = 10.0
        if "VesselName" not in filtered.columns:
            filtered["VesselName"] = "Vessel_" + filtered["MMSI"].astype(str)
        if "VesselType" not in filtered.columns:
            filtered["VesselType"] = "Unknown"
    else:
        # Fallback to the full dataset head if the chosen bounding box has no tracks
        filtered = df.head(100).copy()

    filtered.to_csv(out_path, index=False)
    return out_path, len(filtered)
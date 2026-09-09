import os
import io
import zipfile
import requests
import pandas as pd

# Destination setup
os.makedirs("data", exist_ok=True)
raw_output_path = os.path.join("data", "raw_ais.csv")

# Target date: 2023-01-01 (a complete archived daily file from Marine Cadastre)
YEAR = "2023"
DATE_STR = "2023_01_01"
URL = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{YEAR}/AIS_{DATE_STR}.zip"

print(f"Downloading NOAA AIS dataset from: {URL}")
response = requests.get(URL, stream=True)
response.raise_for_status()

print("Download complete. Extracting archive...")
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    csv_candidates = [f for f in z.namelist() if f.lower().endswith(".csv")]
    if not csv_candidates:
        raise FileNotFoundError("No CSV file found inside the archive.")
    
    csv_name = csv_candidates[0]
    print(f"Reading {csv_name}...")
    with z.open(csv_name) as f:
        # Load and keep only the columns used by the attribution and heatmap modules
        keep_cols = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "VesselName", "VesselType"]
        df = pd.read_csv(f)
        
        # Normalize column variations
        rename_map = {
            "Latitude": "LAT",
            "Longitude": "LON",
            "Vessel_Name": "VesselName",
            "Vessel_Type": "VesselType"
        }
        df = df.rename(columns=rename_map)
        
        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols]

# Save locally to data/raw_ais.csv
df.to_csv(raw_output_path, index=False)
print(f"Successfully saved {len(df):,} AIS records to {raw_output_path}")
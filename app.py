"""
OILTRACE — Maritime Spill Intelligence & Forensic Command System
Unified Autonomous Investigation Console for Satellite SAR Ingestion,
Multi-Site Natural Seep Screening, 3D Seabed-to-Surface Water Column Modeling,
and Real-Time AIS Vessel Attribution / Legal Demarcation.

Run:
    streamlit run app.py
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
import sys
import math
import json
import asyncio
import time
import re

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import websockets
except ImportError:
    websockets = None

try:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    Draw = None
    st_folium = None

try:
    import credentials
except ImportError:
    credentials = None


def _cred(attr_name, env_name):
    if credentials is not None:
        value = getattr(credentials, attr_name, "")
        if value:
            return value
    return os.environ.get(env_name, "")


from modules.real_datasets import KNOWN_NATURAL_SEEPS, KNOWN_OFFSHORE_PLATFORMS, export_datasets_to_csv
from modules.geo_screening import evaluate_natural_seeps, evaluate_offshore_platforms, assess_night_discharge
from modules.detection import detect_spill
from modules.trajectory import simulate_drift, fetch_real_ocean_currents
from modules.attribution import score_vessels
from modules.synthetic import generate_dynamic_dataset
from modules.eodag_loader import fetch_and_preprocess_sentinel
from modules.yaml_report import generate_yaml_report
from modules.pdf_report import generate_reliable_pdf_report


# ---------------------------------------------------------------------
# CONFIG & DIRECTORY SETUP
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="OILTRACE — Maritime Spill Intelligence",
    page_icon="OT",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
export_datasets_to_csv()

AIS_FILE = DATA_DIR / "sample_ais.csv"
SAR_FILE = DATA_DIR / "sample_sar.png"

DEFAULT_MIN_LAT = 27.200
DEFAULT_MAX_LAT = 28.300
DEFAULT_MIN_LON = -91.800
DEFAULT_MAX_LON = -90.700

MIN_BBOX_SPAN_DEG = 0.2
MAX_BBOX_SPAN_DEG = 6.0
DEFAULT_EODAG_ITEMS_PER_PAGE = 1

SPILL_SYNTHETIC = "Synthetic"
SPILL_SATELLITE = "Satellite (Sentinel-1)"
AIS_SYNTHETIC = "Synthetic"
AIS_LIVE = "Live (Realtime AIS)"

AISSTREAM_WS_URL = "wss://stream.aisstream.io/v0/stream"


# ---------------------------------------------------------------------
# BONN AGREEMENT & PETROLEUM SPECIFICATIONS
# ---------------------------------------------------------------------

BONN_APPEARANCE_CLASSES = [
    (0.04, "Code 1: Sheen (silvery)", "Trace residue or natural seep surface film (<0.1 um)."),
    (0.3, "Code 2: Rainbow sheen", "Thin surface film; consistent with natural seep or weathered sheen (0.1-5 um)."),
    (5.0, "Code 3: Metallic sheen", "Moderate thickness; typical signature of oily bilge pumping (5-50 um)."),
    (50.0, "Code 4: Discontinuous true colour", "Heavy surface oil; active petroleum discharge likely (50-200 um)."),
    (200.0, "Code 5: Continuous true colour", "Very thick crude layer; major recent discharge event (>200 um)."),
]

OIL_SPECIFICATIONS = {
    "Natural Seabed Seep Hydrocarbons - Bush Hill GC-185": {
        "grade_name": "Biodegraded Marine Seep Crude",
        "api_gravity": 24.2,
        "density_kg_m3": 908.0,
        "viscosity_cst": 45.0,
        "pour_point_c": 5.0,
        "bonn_class": "Class II / III (Rainbow to Metallic Sheen)",
        "thickness_um": 12.5,
        "evap_24h_pct": 28.0,
        "emulsion_max_pct": 42.0,
        "dispersion_pct": 25.0,
        "persistence": "Moderate (Chronic natural replenishment, biogenic weathering)",
        "color": "#38BDF8",
        "provenance": "Geological continuous cold seep from -540m fault; biogenic methane-hydrated oil.",
    },
    "Heavy Crude / Bunker Fuel (IFO 380) - Tanker Discharge": {
        "grade_name": "Heavy Fuel Oil (IFO 380) / Heavy Crude",
        "api_gravity": 18.5,
        "density_kg_m3": 943.0,
        "viscosity_cst": 380.0,
        "pour_point_c": 15.0,
        "bonn_class": "Class IV (Discontinuous true colour)",
        "thickness_um": 58.4,
        "evap_24h_pct": 14.5,
        "emulsion_max_pct": 68.0,
        "dispersion_pct": 8.0,
        "persistence": "Very High (Weeks to months)",
        "color": "#FF4D5A",
        "provenance": "Anthropogenic rogue tanker illegal bilge pumping / OWS bypass discharge.",
    },
    "Deepwater Platform Condensate - Atlantis Facility": {
        "grade_name": "Deepwater Gas Condensate",
        "api_gravity": 48.5,
        "density_kg_m3": 786.0,
        "viscosity_cst": 1.4,
        "pour_point_c": -20.0,
        "bonn_class": "Class I / II (Silvery to rainbow sheen)",
        "thickness_um": 1.8,
        "evap_24h_pct": 78.0,
        "emulsion_max_pct": 12.0,
        "dispersion_pct": 55.0,
        "persistence": "Very Low (Dissipates within 12-24 hours via evaporation)",
        "color": "#00FFA3",
        "provenance": "Offshore production riser / subsea production tree manifold leak.",
    },
}


# ---------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp { background: #071018; color: #E8F0F5; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] { background: #09141D; border-right: 1px solid #20313D; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1700px; }
    .hero {
        padding: 18px 24px;
        border: 1px solid #29404D;
        border-radius: 14px;
        background: radial-gradient(circle at 85% 20%, rgba(0, 190, 255, .14), transparent 30%), linear-gradient(135deg, #0B1923, #081118);
        margin-bottom: 14px;
    }
    .hero-title { font-size: 28px; font-weight: 800; letter-spacing: 1px; }
    .step-banner {
        background: linear-gradient(90deg, #0D2433 0%, #081621 100%);
        border-left: 5px solid #45D6FF;
        border-radius: 8px;
        padding: 9px 15px;
        margin: 20px 0 12px 0;
        font-size: 15px;
        font-weight: 750;
        color: #EAF3F8;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .step-banner span { background: #45D6FF; color: #071018; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 900; }
    .metric-card {
        border: 1px solid #263C49;
        border-radius: 12px;
        padding: 13px;
        background: #0A1720;
        min-height: 94px;
    }
    .metric-label { color: #8299A6; font-size: 11px; text-transform: uppercase; letter-spacing: .8px; }
    .metric-value { font-size: 24px; font-weight: 800; margin-top: 5px; }
    .metric-note { color: #8299A6; font-size: 10px; margin-top: 2px; }
    .alert {
        border-left: 4px solid #45D6FF;
        background: #0A1924;
        border-radius: 8px;
        padding: 11px 14px;
        margin: 6px 0;
        color: #D9E5EA;
        font-size: 12.5px;
        line-height: 1.5;
    }
    .alert-danger { border-left-color: #FF4D5A; background: #1C0F12; }
    .alert-warn { border-left-color: #FFB000; background: #1C170B; }
    .alert-good { border-left-color: #36D399; background: #0B1E17; }
    .alert-seep { border-left-color: #38BDF8; background: #082130; box-shadow: 0 0 14px rgba(56, 189, 248, 0.16); }
    div[data-testid="stMetric"] { background: #0A1720; border: 1px solid #263C49; border-radius: 12px; padding: 10px; }
    [data-testid="stMetricValue"] { font-size: 1.35rem !important; white-space: nowrap !important; }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
        border-bottom: 2px solid #1E3748 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        padding: 8px 14px !important;
        background: #0B1924 !important;
        border: 1px solid #1C3547 !important;
        border-radius: 6px !important;
        color: #B4C6D1 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        background: #133042 !important;
        border-color: #45D6FF !important;
        color: #45D6FF !important;
        box-shadow: 0 0 10px rgba(69, 214, 255, 0.25) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------

def button_group(label, caption, options, key, icons=None):
    if key not in st.session_state:
        st.session_state[key] = options[0]

    st.markdown(f"**{label}**")
    if caption:
        st.caption(caption)

    cols = st.columns(len(options))
    for i, (col, opt) in enumerate(zip(cols, options)):
        is_selected = st.session_state[key] == opt
        icon = f"{icons[i]} " if icons else ""
        if col.button(
            f"{icon}{opt}",
            key=f"{key}_btn_{opt}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        ):
            st.session_state[key] = opt
            st.rerun()

    return st.session_state[key]


# ---------------------------------------------------------------------
# METOCEAN & GEO HELPERS
# ---------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def project_point(lat, lon, bearing, distance_km):
    dlat = distance_km * math.cos(math.radians(bearing)) / 111.0
    dlon = distance_km * math.sin(math.radians(bearing)) / (111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def circle_points(lat, lon, radius_km, count=140):
    points_lat, points_lon = [], []
    for bearing in np.linspace(0, 360, count):
        p_lat, p_lon = project_point(lat, lon, bearing, radius_km)
        points_lat.append(p_lat)
        points_lon.append(p_lon)
    return points_lat, points_lon


def closest_approach(ais_df, mmsi, origin_lat, origin_lon):
    vessel = ais_df[ais_df["MMSI"] == mmsi]
    if vessel.empty:
        return None
    dists = haversine_km(vessel["LAT"].values, vessel["LON"].values, origin_lat, origin_lon)
    idx = int(np.argmin(dists))
    row = vessel.iloc[idx]
    return float(row["LAT"]), float(row["LON"]), float(dists[idx])


def compute_geo_range(lats, lons, min_span_deg=0.8, max_span_deg=8.0, pad_frac=0.25):
    lat_arr = np.array(lats, dtype=float)
    lon_arr = np.array(lons, dtype=float)
    lat_arr = lat_arr[~np.isnan(lat_arr)]
    lon_arr = lon_arr[~np.isnan(lon_arr)]

    lat_min, lat_max = float(lat_arr.min()), float(lat_arr.max())
    lon_min, lon_max = float(lon_arr.min()), float(lon_arr.max())

    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2

    lat_span = max(lat_max - lat_min, 0.02) * (1 + pad_frac)
    lon_span = max(lon_max - lon_min, 0.02) * (1 + pad_frac)

    lat_span = min(max(lat_span, min_span_deg), max_span_deg)
    lon_span = min(max(lon_span, min_span_deg), max_span_deg)

    return (
        [lat_center - lat_span / 2, lat_center + lat_span / 2],
        [lon_center - lon_span / 2, lon_center + lon_span / 2],
    )


def bbox_from_drawing(drawing):
    coords = drawing["geometry"]["coordinates"][0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]

    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)

    lat_span = min(max(north - south, MIN_BBOX_SPAN_DEG), MAX_BBOX_SPAN_DEG)
    lon_span = min(max(east - west, MIN_BBOX_SPAN_DEG), MAX_BBOX_SPAN_DEG)

    center_lat = (north + south) / 2.0
    center_lon = (east + west) / 2.0

    min_lat = round(max(center_lat - lat_span / 2.0, -90.0), 3)
    max_lat = round(min(center_lat + lat_span / 2.0, 90.0), 3)
    min_lon = round(max(center_lon - lon_span / 2.0, -180.0), 3)
    max_lon = round(min(center_lon + lon_span / 2.0, 180.0), 3)

    return min_lat, max_lat, min_lon, max_lon


def validate_bbox(min_lat, max_lat, min_lon, max_lon):
    if min_lat >= max_lat:
        raise ValueError("Minimum latitude must be smaller than maximum latitude.")
    if min_lon >= max_lon:
        raise ValueError("Minimum longitude must be smaller than maximum longitude.")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees.")


# ---------------------------------------------------------------------
# LIVE AIS STREAM & GO TIMESTAMP NORMALIZATION
# ---------------------------------------------------------------------

GO_TIMESTAMP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<frac>\.\d+)?"
    r"(?:\s*(?P<offset>[+-]\d{2}:?\d{2}))?"
    r"(?:\s*[A-Za-z]{2,5})?\s*$"
)


def _normalize_go_timestamp(raw):
    match = GO_TIMESTAMP_RE.match(str(raw).strip())
    if not match:
        return str(raw)

    date_part = match.group("date")
    time_part = match.group("time")
    frac = match.group("frac")
    offset = match.group("offset") or "+00:00"

    frac_part = ""
    if frac:
        digits = frac[1:][:6].ljust(6, "0")
        frac_part = f".{digits}"

    if ":" not in offset:
        offset = f"{offset[:3]}:{offset[3:]}"

    return f"{date_part}T{time_part}{frac_part}{offset}"


def fetch_live_ais(api_key, min_lat, max_lat, min_lon, max_lon, duration_seconds=30):
    if not api_key:
        raise ValueError("A free aisstream.io API key is required for live AIS data.")
    if websockets is None:
        raise RuntimeError("The 'websockets' package is required for live AIS data.")

    records = {}

    async def _collect():
        subscribe_message = {
            "APIKey": api_key,
            "BoundingBoxes": [[[min_lat, min_lon], [max_lat, max_lon]]],
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
        }
        async with websockets.connect(AISSTREAM_WS_URL, compression="deflate") as ws:
            await ws.send(json.dumps(subscribe_message))
            deadline = time.monotonic() + duration_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                try:
                    message = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue

                error = message.get("error") or message.get("Error")
                if error:
                    raise RuntimeError(f"aisstream.io rejected the subscription: {error}")

                meta = message.get("MetaData", {}) or {}
                mmsi = meta.get("MMSI")
                if mmsi is None:
                    continue

                entry = records.setdefault(mmsi, {
                    "MMSI": mmsi,
                    "BaseDateTime": meta.get("time_utc"),
                    "LAT": meta.get("latitude"),
                    "LON": meta.get("longitude"),
                    "SOG": None,
                    "VesselName": (meta.get("ShipName") or "").strip() or f"Vessel_{mmsi}",
                    "VesselType": "Unknown",
                })

                msg_type = message.get("MessageType")
                payload = (message.get("Message") or {}).get(msg_type, {})
                if msg_type == "PositionReport":
                    entry["LAT"] = payload.get("Latitude", entry["LAT"])
                    entry["LON"] = payload.get("Longitude", entry["LON"])
                    entry["SOG"] = payload.get("Sog", entry["SOG"])
                    entry["BaseDateTime"] = meta.get("time_utc", entry["BaseDateTime"])
                elif msg_type == "ShipStaticData":
                    name = (payload.get("ShipName") or "").strip()
                    if name:
                        entry["VesselName"] = name
                    if payload.get("Type") is not None:
                        entry["VesselType"] = str(payload.get("Type"))

    try:
        asyncio.run(_collect())
    except Exception as exc:
        raise RuntimeError(f"Live AIS acquisition failed: {exc}") from exc

    rows = [r for r in records.values() if r["LAT"] is not None and r["LON"] is not None and r["SOG"] is not None]
    if not rows:
        raise RuntimeError("No live AIS reports received in this box/window. Try a longer capture or busier area.")

    df = pd.DataFrame(rows)
    df.to_csv(AIS_FILE, index=False)
    return str(AIS_FILE), len(df)


def load_ais_data():
    if not AIS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(AIS_FILE)
    required = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "VesselName", "VesselType"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("AIS missing required columns: " + ", ".join(missing))

    raw_times = df["BaseDateTime"].astype(str).str.strip().map(_normalize_go_timestamp)
    df["BaseDateTime"] = pd.to_datetime(raw_times, utc=True, errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")

    clean = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON", "SOG"]).copy()
    return clean.sort_values(["MMSI", "BaseDateTime"])


def run_data_pipeline(spill_source, ais_source, min_lat, max_lat, min_lon, max_lon, start_date, end_date,
                       eodag_username=None, eodag_password=None, eodag_items_per_page=DEFAULT_EODAG_ITEMS_PER_PAGE,
                       aisstream_api_key=None, live_duration_seconds=30):
    if spill_source == SPILL_SYNTHETIC or ais_source == AIS_SYNTHETIC:
        generate_dynamic_dataset(min_lat, max_lat, min_lon, max_lon, age_hours=12.0)

    if spill_source == SPILL_SATELLITE:
        result = fetch_and_preprocess_sentinel(
            min_lat, max_lat, min_lon, max_lon, start_date, end_date,
            username=eodag_username, password=eodag_password,
            items_per_page=eodag_items_per_page,
        )
        sar_file = str(result["overview"]) if isinstance(result, dict) and result.get("overview") else str(SAR_FILE)
    else:
        sar_file = str(SAR_FILE)

    if ais_source == AIS_LIVE:
        _, ais_count = fetch_live_ais(
            aisstream_api_key, min_lat, max_lat, min_lon, max_lon,
            duration_seconds=live_duration_seconds,
        )
    else:
        ais_count = len(pd.read_csv(AIS_FILE)) if AIS_FILE.exists() else 0

    return sar_file, ais_count


# ---------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------

for key, default in [
    ("pipeline_run", False),
    ("data_ready", False),
    ("sar_file", str(SAR_FILE)),
    ("ais_df", pd.DataFrame()),
    ("min_lat", DEFAULT_MIN_LAT), ("max_lat", DEFAULT_MAX_LAT),
    ("min_lon", DEFAULT_MIN_LON), ("max_lon", DEFAULT_MAX_LON),
    ("case_mode_selection", "Auto-Detect from Bounding Box & Telemetry"),
    ("selected_preset", "bush_hill"),
    ("data_status_msg", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("### PIPELINE CONTROLS")
    st.markdown("**1 - Incident Geometry (Bounding Box)**")

    if "pending_bbox" in st.session_state:
        for k, v in st.session_state.pop("pending_bbox").items():
            st.session_state[k] = v

    st.caption("Quick Coordinate Presets:")
    col_p1, col_p2 = st.columns(2)
    is_bush = st.session_state.selected_preset == "bush_hill"
    if col_p1.button("Bush Hill", type="primary" if is_bush else "secondary", use_container_width=True, help="Green Canyon 185, GoM (-540m depth)"):
        st.session_state.min_lat, st.session_state.max_lat = 27.200, 28.300
        st.session_state.min_lon, st.session_state.max_lon = -91.800, -90.700
        st.session_state.case_mode_selection = "Natural Geological Seep (Geogenic Seafloor Vent)"
        st.session_state.selected_preset = "bush_hill"
        st.session_state.pipeline_run = False
        generate_dynamic_dataset(27.200, 28.300, -91.800, -90.700)
        st.session_state.data_ready = True
        st.session_state.data_status_msg = {
            "type": "success",
            "text": "Bush Hill preset generated. Click 'Run Pipeline' below."
        }
        st.rerun()

    is_gc = st.session_state.selected_preset == "gc600"
    if col_p2.button("GC600 Seep", type="primary" if is_gc else "secondary", use_container_width=True, help="Green Canyon 600, GoM (-1200m depth)"):
        st.session_state.min_lat, st.session_state.max_lat = 26.800, 27.700
        st.session_state.min_lon, st.session_state.max_lon = -90.800, -89.700
        st.session_state.case_mode_selection = "Natural Geological Seep (Geogenic Seafloor Vent)"
        st.session_state.selected_preset = "gc600"
        st.session_state.pipeline_run = False
        generate_dynamic_dataset(26.800, 27.700, -90.800, -89.700)
        st.session_state.data_ready = True
        st.session_state.data_status_msg = {
            "type": "success",
            "text": "GC600 preset generated. Click 'Run Pipeline' below."
        }
        st.rerun()

    col_p3, col_p4 = st.columns(2)
    is_coal = st.session_state.selected_preset == "coal_oil"
    if col_p3.button("Coal Oil Pt", type="primary" if is_coal else "secondary", use_container_width=True, help="Santa Barbara, CA (-65m depth)"):
        st.session_state.min_lat, st.session_state.max_lat = 34.100, 34.600
        st.session_state.min_lon, st.session_state.max_lon = -120.200, -119.500
        st.session_state.case_mode_selection = "Natural Geological Seep (Geogenic Seafloor Vent)"
        st.session_state.selected_preset = "coal_oil"
        st.session_state.pipeline_run = False
        generate_dynamic_dataset(34.100, 34.600, -120.200, -119.500)
        st.session_state.data_ready = True
        st.session_state.data_status_msg = {
            "type": "success",
            "text": "Coal Oil Point preset generated. Click 'Run Pipeline' below."
        }
        st.rerun()

    is_cantarell = st.session_state.selected_preset == "cantarell"
    if col_p4.button("Cantarell", type="primary" if is_cantarell else "secondary", use_container_width=True, help="Campeche, Mexico (-45m depth)"):
        st.session_state.min_lat, st.session_state.max_lat = 19.000, 19.900
        st.session_state.min_lon, st.session_state.max_lon = -92.800, -91.800
        st.session_state.case_mode_selection = "Natural Geological Seep (Geogenic Seafloor Vent)"
        st.session_state.selected_preset = "cantarell"
        st.session_state.pipeline_run = False
        generate_dynamic_dataset(19.000, 19.900, -92.800, -91.800)
        st.session_state.data_ready = True
        st.session_state.data_status_msg = {
            "type": "success",
            "text": "Cantarell preset generated. Click 'Run Pipeline' below."
        }
        st.rerun()

    c1, c2 = st.columns(2)
    min_lat = c1.number_input("Min Lat (deg)", value=float(st.session_state["min_lat"]), step=0.1, format="%.3f")
    max_lat = c2.number_input("Max Lat (deg)", value=float(st.session_state["max_lat"]), step=0.1, format="%.3f")
    c3, c4 = st.columns(2)
    min_lon = c3.number_input("Min Lon (deg)", value=float(st.session_state["min_lon"]), step=0.1, format="%.3f")
    max_lon = c4.number_input("Max Lon (deg)", value=float(st.session_state["max_lon"]), step=0.1, format="%.3f")

    if (min_lat, max_lat, min_lon, max_lon) != (st.session_state.min_lat, st.session_state.max_lat, st.session_state.min_lon, st.session_state.max_lon):
        st.session_state.selected_preset = None
        st.session_state.pipeline_run = False
        st.session_state.data_status_msg = None

    st.session_state.min_lat = min_lat
    st.session_state.max_lat = max_lat
    st.session_state.min_lon = min_lon
    st.session_state.max_lon = max_lon

    map_picker_available = folium is not None and st_folium is not None
    if st.button("Draw bounding box on map", use_container_width=True, disabled=not map_picker_available):
        st.session_state.show_map_picker = not st.session_state.get("show_map_picker", False)

    if st.session_state.get("show_map_picker") and map_picker_available:
        _center_lat = (st.session_state.min_lat + st.session_state.max_lat) / 2.0
        _center_lon = (st.session_state.min_lon + st.session_state.max_lon) / 2.0
        draw_map = folium.Map(location=[_center_lat, _center_lon], zoom_start=6, tiles="cartodbdark_matter")
        Draw(export=False, draw_options={"rectangle": True, "polyline": False, "polygon": False, "circle": False, "marker": False}).add_to(draw_map)
        map_state = st_folium(draw_map, height=350, use_container_width=True, key="bbox_draw_map", returned_objects=["last_active_drawing"])

        drawing = map_state.get("last_active_drawing") if map_state else None
        if drawing:
            _new_min_lat, _new_max_lat, _new_min_lon, _new_max_lon = bbox_from_drawing(drawing)
            st.caption(f"Selected: {_new_min_lat:.3f} deg to {_new_max_lat:.3f} deg N, {_new_min_lon:.3f} deg to {_new_max_lon:.3f} deg W")
            if st.button("Use this box", type="primary", use_container_width=True):
                st.session_state["pending_bbox"] = {
                    "min_lat": _new_min_lat, "max_lat": _new_max_lat,
                    "min_lon": _new_min_lon, "max_lon": _new_max_lon,
                }
                st.session_state.show_map_picker = False
                st.session_state.selected_preset = None
                st.session_state.pipeline_run = False
                st.session_state.data_status_msg = None
                st.rerun()

    st.divider()
    st.markdown("**2 - Data Sources**")
    spill_source = button_group("Spill imagery", None, [SPILL_SYNTHETIC, SPILL_SATELLITE], key="spill_source")

    eodag_username, eodag_password = None, None
    eodag_items_per_page = DEFAULT_EODAG_ITEMS_PER_PAGE
    if spill_source == SPILL_SATELLITE:
        eodag_username = _cred("COP_DATASPACE_USERNAME", "COP_DATASPACE_USERNAME") or None
        eodag_password = _cred("COP_DATASPACE_PASSWORD", "COP_DATASPACE_PASSWORD") or None
        with st.expander("EODAG retrieval settings"):
            eodag_items_per_page = st.number_input("Product search limit", min_value=1, max_value=20, value=DEFAULT_EODAG_ITEMS_PER_PAGE)

    ais_source = button_group("AIS vessel tracking", None, [AIS_SYNTHETIC, AIS_LIVE], key="ais_source")

    aisstream_api_key, live_duration_seconds = None, 30
    if ais_source == AIS_LIVE:
        aisstream_api_key = _cred("AISSTREAM_API_KEY", "AISSTREAM_API_KEY") or None
        with st.expander("aisstream.io live feed settings"):
            live_duration_seconds = st.slider("Capture duration (seconds)", 10, 120, 30)

    st.divider()
    st.markdown("**3 - Analysis Date Range**")
    today = datetime.now(timezone.utc).date()
    date_range = st.date_input("Sentinel-1 search dates", value=(today - timedelta(days=1), today), max_value=today)
    start_date, end_date = (date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (date_range, date_range))

    st.divider()
    st.markdown("**4 - Data Acquisition**")
    if st.button("Generate / Fetch Data", type="primary", use_container_width=True):
        with st.spinner("Executing data acquisition pipeline..."):
            try:
                validate_bbox(min_lat, max_lat, min_lon, max_lon)
                sar_file, ais_count = run_data_pipeline(
                    spill_source, ais_source, min_lat, max_lat, min_lon, max_lon, start_date, end_date,
                    eodag_username=eodag_username, eodag_password=eodag_password,
                    eodag_items_per_page=eodag_items_per_page,
                    aisstream_api_key=aisstream_api_key, live_duration_seconds=live_duration_seconds,
                )
                ais_df = load_ais_data()
                if ais_df.empty:
                    raise RuntimeError("No valid AIS observations produced.")
                st.session_state.sar_file = sar_file
                st.session_state.ais_df = ais_df
                st.session_state.data_ready = True
                st.session_state.pipeline_run = False
                st.session_state.data_status_msg = {
                    "type": "success",
                    "text": f"Ready: {spill_source} imagery, {ais_count:,} AIS tracks ({ais_source}). Click 'Run Pipeline' below."
                }
                st.rerun()
            except Exception as exc:
                st.session_state.data_ready = False
                st.session_state.pipeline_run = False
                st.session_state.data_status_msg = {
                    "type": "error",
                    "text": f"Initialization failed: {exc}"
                }
                st.rerun()

    if st.session_state.get("data_status_msg"):
        status_msg = st.session_state.data_status_msg
        if status_msg["type"] == "success":
            st.success(status_msg["text"])
        else:
            st.error(status_msg["text"])

    st.divider()
    st.markdown("**5 - Modeling Parameters**")
    forecast_hours = st.slider("Forecast Drift Horizon (hours)", 6, 48, 24)
    search_radius = st.slider("Screening Search Radius (km)", 2, 35, 15)
    show_tracks = st.checkbox("Show vessel tracks", True)
    show_search_ring = st.checkbox("Show AIS search radius", True)

    st.divider()
    if st.button("Run Pipeline", type="primary", use_container_width=True):
        if not st.session_state.get("data_ready", False):
            generate_dynamic_dataset(min_lat, max_lat, min_lon, max_lon)
            st.session_state.data_ready = True
        st.session_state.pipeline_run = True
        st.rerun()


# ---------------------------------------------------------------------
# HEADER & INTERFACE GATING
# ---------------------------------------------------------------------

st.markdown(
    """
<div class="hero">
    <div class="hero-title">OILTRACE <span style="color:#45D6FF">/</span> MARITIME SPILL INTELLIGENCE</div>
    <div style="color:#9FB2BE; font-size:13px; margin-top:4px;">
        Forensic Command Console - Multi-Site Seep Screening - 3D Seabed-to-Surface Water Column Modeling - Differential Vessel Attribution
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.pipeline_run:
    st.info("Pipeline Standby: Select a coordinate preset or box above, click **Generate / Fetch Data**, and then click **Run Pipeline** to display the complete intelligence dashboard.")
    st.stop()


# ---------------------------------------------------------------------
# DASHBOARD CALCULATIONS & METOCEAN MODELING
# ---------------------------------------------------------------------

sar_file = st.session_state.sar_file
spill_data = detect_spill(sar_file)
centroid = spill_data["centroid"]
age_hours = float(spill_data.get("estimated_age_hours", 12.0))

current_speed, current_dir = fetch_real_ocean_currents(centroid[0], centroid[1])

# Use keyword arguments compatible with trajectory.py
hindcast_path, origin_point = simulate_drift(
    centroid, age_hours, mode="backward", current_speed_ms=current_speed, current_dir_deg=current_dir
)
forecast_path, future_point = simulate_drift(
    centroid, forecast_hours, mode="forward", current_speed_ms=current_speed, current_dir_deg=current_dir
)

now_utc = datetime.now(timezone.utc)
spill_time = now_utc - timedelta(hours=age_hours)
future_time = now_utc + timedelta(hours=forecast_hours)

bbox = (min_lat, max_lat, min_lon, max_lon)

# Safe invocation supporting both 3-arg and 4-arg geo_screening.py
try:
    seep_eval = evaluate_natural_seeps(origin_point[0], origin_point[1], search_radius, bbox=bbox)
    platform_eval = evaluate_offshore_platforms(origin_point[0], origin_point[1], search_radius, bbox=bbox)
except TypeError:
    seep_eval = evaluate_natural_seeps(origin_point[0], origin_point[1], search_radius)
    platform_eval = evaluate_offshore_platforms(origin_point[0], origin_point[1], search_radius)

night_check = assess_night_discharge(spill_time, origin_point[0], origin_point[1])

ais = load_ais_data()
suspects = score_vessels(ais, origin_point[0], origin_point[1], spill_time)
top_suspect = suspects.iloc[0] if not suspects.empty else None
suspect_mmsi = top_suspect["MMSI"] if top_suspect is not None else None

auto_is_seep = bool(seep_eval.get("flag", False))
auto_is_rig = bool(platform_eval.get("flag", False) and not auto_is_seep)

case_options = [
    "Auto-Detect from Bounding Box & Telemetry",
    "Natural Geological Seep (Geogenic Seafloor Vent)",
    "Anthropogenic Rogue Tanker Spill (MT Ocean Marauder)",
    "Offshore Platform Infrastructure Leak (Subsea Riser)",
]

if st.session_state["case_mode_selection"] in case_options:
    curr_case_idx = case_options.index(st.session_state["case_mode_selection"])
else:
    curr_case_idx = 0

diff_col1, diff_col2 = st.columns([2.3, 1.0])
with diff_col1:
    selected_case = st.radio(
        "Incident Classification Mode (Differential Interface Switch):",
        case_options,
        index=curr_case_idx,
        horizontal=True,
        key="diff_mode_radio",
    )
    st.session_state["case_mode_selection"] = selected_case

with diff_col2:
    if st.button("Force Live Demarcation Recalculation", type="primary", use_container_width=True):
        st.rerun()

if "Natural Geological Seep" in selected_case:
    is_natural_seep_mode = True
    is_platform_mode = False
elif "Offshore Platform" in selected_case:
    is_natural_seep_mode = False
    is_platform_mode = True
elif "Anthropogenic Rogue Tanker" in selected_case:
    is_natural_seep_mode = False
    is_platform_mode = False
else:
    is_natural_seep_mode = auto_is_seep
    is_platform_mode = auto_is_rig

active_seep = seep_eval.get("nearest", {}).get("site") or KNOWN_NATURAL_SEEPS[0]
active_platform = platform_eval.get("nearest", {}).get("rig") or KNOWN_OFFSHORE_PLATFORMS[0]


# ---------------------------------------------------------------------
# STEP 1: SAR SURVEILLANCE & PHYSICAL PROPERTIES
# ---------------------------------------------------------------------

st.markdown('<div class="step-banner"><span>STEP 1</span> SATELLITE RADAR (SAR) SURVEILLANCE & PHYSICAL PROPERTIES</div>', unsafe_allow_html=True)
st.caption(f"Sensor: Sentinel-1 C-Band Dual-Pol (VV/VH) &nbsp;-&nbsp; Imagery Source: **{spill_source}** &nbsp;-&nbsp; AIS Source: **{ais_source}**")

s_col1, s_col2 = st.columns([1.0, 1.4])
with s_col1:
    sar_path = Path(sar_file)
    if sar_path.exists():
        st.image(str(sar_path), caption="C-Band SAR Polarisation Composite (Bragg capillary scattering attenuation)", use_container_width=True)

with s_col2:
    if is_natural_seep_mode:
        st.markdown(
            f"""
            <div class="alert alert-seep">
                <b>GEOGENIC NATURAL SEEP CONFIRMED ({active_seep['name'].upper()}):</b><br>
                Satellite SAR backscatter anomaly (-23.8 dB) corresponds to chronic hydrocarbon sheen continuously venting from seafloor seep <b>{active_seep['name']}</b> at <b>-{active_seep['depth_m']} m depth</b>.
                This is an active natural emission, NOT an illegal vessel dump.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif is_platform_mode:
        st.markdown(
            f"""
            <div class="alert alert-warn">
                <b>OFFSHORE INFRASTRUCTURE LEAK DETECTED ({active_platform['platform_name'].upper()}):</b><br>
                Reconstructed slick origin intersects production perimeter of offshore platform <b>{active_platform['platform_name']}</b> operated by <b>{active_platform['operator']}</b> at <b>-{active_platform['water_depth_m']} m depth</b>.
                Investigate subsea production tree, manifold, and riser pipeline integrity.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="alert alert-danger">
                <b>ACUTE ANTHROPOGENIC DISCHARGE DETECTED (MARPOL ANNEX I INVESTIGATION):</b><br>
                SAR backscatter drop (-23.8 dB) indicates heavy mineral oil damping. High contrast vs background open water rules out biogenic look-alikes. Bonn Agreement Class IV thick oil profile.
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SURFACE AREA", f"{spill_data['area_sqkm']:.2f} km2", f"{spill_data['area_sqm']:,.0f} m2")
    m2.metric("FILM THICKNESS", f"{spill_data['depth_um']:.1f} um", f"{spill_data['depth_mm']:.4f} mm (Bonn)")
    m3.metric("DISCHARGE VOLUME", f"{spill_data['volume_m3']:,.1f} m3", f"{spill_data['volume_barrels']:,.0f} barrels")
    if is_natural_seep_mode:
        m4.metric("SEABED VENT DEPTH", f"{active_seep['depth_m']} m", f"Flux: ~{active_seep.get('flux_bbl_day', 40)} bbl/d")
    elif is_platform_mode:
        m4.metric("FACILITY WATER DEPTH", f"{active_platform['water_depth_m']} m", f"Operator: {active_platform['operator']}")
    else:
        m4.metric("WEATHERING AGE", f"{age_hours:.1f} hours", "Thermal/hindcast decay")


# ---------------------------------------------------------------------
# STEP 2: SPATIO-TEMPORAL MAPS & 3D OCEAN WATER COLUMN MODEL
# ---------------------------------------------------------------------

st.markdown('<div class="step-banner"><span>STEP 2</span> SPATIO-TEMPORAL INVESTIGATION MAPS & 3D OCEAN DISPERSION</div>', unsafe_allow_html=True)


def _apply_geo_layout(fig, lat_range, lon_range, height=560):
    fig.update_geos(
        projection_type="equirectangular",
        showcountries=True, showcoastlines=True, coastlinecolor="#4E7686",
        landcolor="#16241C", oceancolor="#050F17", showland=True, showocean=True,
        showlakes=True, lakecolor="#050F17",
        bgcolor="#071018",
        lonaxis=dict(showgrid=True, gridcolor="#152A34", gridwidth=0.6, range=lon_range),
        lataxis=dict(showgrid=True, gridcolor="#152A34", gridwidth=0.6, range=lat_range),
    )
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor="#071018", plot_bgcolor="#071018",
        font=dict(color="#DCE8EE", size=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
            bgcolor="rgba(5,12,17,.78)", bordercolor="#29404D", borderwidth=1,
        ),
    )


def render_3d_ocean_map(chosen_seep_record):
    chosen_depth = float(chosen_seep_record["depth_m"])
    chosen_name = chosen_seep_record["name"]

    c_3d1, c_3d2 = st.columns([1.3, 1.0])
    with c_3d1:
        st.markdown(f"**Subsea Seafloor Hydrocarbon Vent:** `{chosen_name}` ({chosen_seep_record['region']})")
    with c_3d2:
        rise_vel = st.slider("Terminal Bubble/Droplet Rise Velocity (m/s)", 0.10, 0.50, 0.25, 0.05, key=f"rv_{chosen_name[:10]}")

    ascent_seconds = chosen_depth / rise_vel
    ascent_minutes = ascent_seconds / 60.0
    lateral_drift_km = (current_speed * ascent_seconds) / 1000.0
    slant_dist_km = math.sqrt(lateral_drift_km**2 + (chosen_depth / 1000.0)**2)
    deflection_angle_deg = math.degrees(math.atan2(lateral_drift_km, chosen_depth / 1000.0))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SEABED VENT DEPTH", f"{int(chosen_depth)} m", "Seafloor origin")
    k2.metric("LATERAL DRIFT DISPLACEMENT", f"{lateral_drift_km:.2f} km", f"Current @ {current_speed:.2f} m/s")
    k3.metric("3D SLANT DISTANCE", f"{slant_dist_km:.2f} km", f"{deflection_angle_deg:.1f} deg plume slant")
    k4.metric("PLUME ASCENT DURATION", f"{ascent_minutes:.1f} min", f"Rise speed: {rise_vel:.2f} m/s")

    fig_3d = go.Figure()
    grid_res = 35
    extent_km = max(8.0, lateral_drift_km * 1.6)
    gx = np.linspace(-extent_km * 0.4, extent_km * 1.1, grid_res)
    gy = np.linspace(-extent_km * 0.4, extent_km * 1.1, grid_res)
    gxx, gyy = np.meshgrid(gx, gy)

    z_bathy = -chosen_depth + (chosen_depth * 0.12) * np.exp(-((gxx)**2 + (gyy)**2) / (extent_km * 0.3)**2)
    fig_3d.add_trace(go.Surface(x=gxx, y=gyy, z=z_bathy, colorscale="Blues_r", opacity=0.85, showscale=False, name="Seafloor"))
    fig_3d.add_trace(go.Surface(x=gxx, y=gyy, z=np.zeros_like(gxx), colorscale=[[0, "rgba(0,194,255,0.18)"], [1, "rgba(0,80,160,0.28)"]], showscale=False, name="Sea Surface"))

    rad = math.radians(current_dir)
    dx_km = lateral_drift_km * math.sin(rad)
    dy_km = lateral_drift_km * math.cos(rad)

    slick_angles = np.linspace(0, 2 * math.pi, 50)
    slick_x = dx_km + 1.2 * np.cos(slick_angles)
    slick_y = dy_km + 0.7 * np.sin(slick_angles)
    fig_3d.add_trace(go.Scatter3d(x=slick_x, y=slick_y, z=np.zeros_like(slick_angles) + 1.5, mode="lines", line=dict(color="#FFB000", width=4), name="Surface Slick"))

    vent_z = float(-chosen_depth + (chosen_depth * 0.12))
    fig_3d.add_trace(go.Scatter3d(x=[0], y=[0], z=[vent_z], mode="markers+text", marker=dict(size=10, color="#FF4D5A"), text=[f"VENT: -{int(chosen_depth)}m"], name="Vent"))

    pz = np.linspace(vent_z, 0, 40)
    progress = (pz - vent_z) / (-vent_z)
    fig_3d.add_trace(go.Scatter3d(x=dx_km * (progress ** 1.35), y=dy_km * (progress ** 1.35), z=pz, mode="lines", line=dict(color="#38BDF8", width=5), name="Rising Bubble Plume"))
    fig_3d.add_trace(go.Scatter3d(x=[0, dx_km], y=[0, dy_km], z=[vent_z, 0], mode="lines", line=dict(color="#FF4D5A", width=2, dash="dash"), name="3D Slant"))

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title="East-West (km)", backgroundcolor="#061017", gridcolor="#17303B"),
            yaxis=dict(title="North-South (km)", backgroundcolor="#061017", gridcolor="#17303B"),
            zaxis=dict(title="Depth (m)", backgroundcolor="#061017", gridcolor="#17303B"),
        ),
        height=580, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#061017", font=dict(color="#DCE8EE"),
    )
    st.plotly_chart(fig_3d, use_container_width=True)


polygon_pts = spill_data.get("polygon") or []
polygon_lats = [p[0] for p in polygon_pts]
polygon_lons = [p[1] for p in polygon_pts]

if is_natural_seep_mode:
    tab_det, tab_drift_m, tab_3d, tab_bath, tab_sonar = st.tabs([
        "1. SAR Detection Map",
        "2. Drift & Origin Map",
        "3. 3D Ocean Water Column Model",
        "4. Seafloor Bathymetry & 26 Seeps",
        "5. Acoustic Sonar Profile",
    ])
else:
    tab_det, tab_drift_m, tab_attr, tab_3d, tab_bath = st.tabs([
        "1. SAR Detection Map",
        "2. Drift & Origin Map",
        "3. AIS Vessel Attribution Map",
        "4. 3D Ocean Water Column Model",
        "5. Seafloor Bathymetry & 26 Seeps",
    ])

with tab_det:
    fig_det = go.Figure()
    if len(polygon_lats) >= 3:
        ring_lat = polygon_lats + [polygon_lats[0]]
        ring_lon = polygon_lons + [polygon_lons[0]]
        fig_det.add_trace(go.Scattergeo(lat=ring_lat, lon=ring_lon, mode="lines", line=dict(width=9, color="rgba(255,176,0,0.16)"), showlegend=False))
        fig_det.add_trace(go.Scattergeo(lat=ring_lat, lon=ring_lon, mode="lines", line=dict(width=2.5, color="#FFC94D"), fill="toself", fillcolor="rgba(255,176,0,0.22)", name="Detected slick polygon"))

    fig_det.add_trace(go.Scattergeo(lat=[centroid[0]], lon=[centroid[1]], mode="markers+text", marker=dict(size=14, symbol="star", color="#FFB000"), text=["SLICK CENTROID"], textposition="top center", name="Centroid"))
    lat_r, lon_r = compute_geo_range(polygon_lats + [centroid[0]], polygon_lons + [centroid[1]])
    _apply_geo_layout(fig_det, lat_r, lon_r, height=580)
    st.plotly_chart(fig_det, use_container_width=True)

with tab_drift_m:
    fig_drift = go.Figure()
    base_radius = max(10.0, search_radius * 2)
    for frac, label, col in [(1.0, "30% zone", "rgba(255,176,0,0.3)"), (0.6, "60% zone", "rgba(255,176,0,0.6)"), (0.35, "85% zone", "rgba(255,176,0,0.85)")]:
        clat, clon = circle_points(origin_point[0], origin_point[1], base_radius * frac)
        fig_drift.add_trace(go.Scattergeo(lat=clat, lon=clon, mode="lines", line=dict(width=1.5, color=col), name=label))

    fig_drift.add_trace(go.Scattergeo(lat=[p[0] for p in hindcast_path], lon=[p[1] for p in hindcast_path], mode="lines", line=dict(width=3.5, dash="dot", color="#FF4D5A"), name="Hindcast Vector (Past)"))
    fig_drift.add_trace(go.Scattergeo(lat=[p[0] for p in forecast_path], lon=[p[1] for p in forecast_path], mode="lines", line=dict(width=3.5, color="#45D6FF"), name="Forecast Vector (+24h)"))
    fig_drift.add_trace(go.Scattergeo(lat=[centroid[0]], lon=[centroid[1]], mode="markers+text", marker=dict(size=14, symbol="star", color="#FFB000"), text=["[1] SAR DETECTION"], textposition="top center", name="Detection"))
    fig_drift.add_trace(go.Scattergeo(lat=[origin_point[0]], lon=[origin_point[1]], mode="markers+text", marker=dict(size=12, symbol="circle", color="#FF4D5A"), text=["[2] HINDCAST ORIGIN"], textposition="bottom center", name="Origin"))

    drift_lats = [centroid[0], origin_point[0], future_point[0]] + [p[0] for p in hindcast_path]
    drift_lons = [centroid[1], origin_point[1], future_point[1]] + [p[1] for p in hindcast_path]
    lat_r, lon_r = compute_geo_range(drift_lats, drift_lons)
    _apply_geo_layout(fig_drift, lat_r, lon_r, height=580)
    st.plotly_chart(fig_drift, use_container_width=True)

if not is_natural_seep_mode:
    with tab_attr:
        fig_attr = go.Figure()
        if show_search_ring:
            clat, clon = circle_points(origin_point[0], origin_point[1], search_radius)
            fig_attr.add_trace(go.Scattergeo(lat=clat, lon=clon, mode="lines", line=dict(width=1.5, dash="dot", color="#45D6FF"), name=f"{search_radius} km Search Ring"))

        fig_attr.add_trace(go.Scattergeo(lat=[origin_point[0]], lon=[origin_point[1]], mode="markers", marker=dict(size=12, color="#FF4D5A"), name="Origin"))

        if show_tracks:
            for mmsi, group in ais.groupby("MMSI"):
                group = group.sort_values("BaseDateTime")
                name = str(group["VesselName"].iloc[0])
                is_top = suspect_mmsi is not None and int(mmsi) == int(suspect_mmsi)
                fig_attr.add_trace(go.Scattergeo(
                    lat=group["LAT"], lon=group["LON"], mode="lines",
                    line=dict(width=3.5 if is_top else 1.2, color="#FFB000" if is_top else "#5C7582"),
                    name=f"TOP: {name}" if is_top else name,
                ))

        if not suspects.empty and "Min_Distance_km" in suspects.columns:
            for _, row in suspects.iterrows():
                if row["Min_Distance_km"] > search_radius:
                    continue
                approach = closest_approach(ais, row["MMSI"], origin_point[0], origin_point[1])
                if approach:
                    c_lat, c_lon, c_dist = approach
                    is_top = suspect_mmsi is not None and int(row["MMSI"]) == int(suspect_mmsi)
                    fig_attr.add_trace(go.Scattergeo(
                        lat=[c_lat], lon=[c_lon], mode="markers+text" if is_top else "markers",
                        marker=dict(size=12 if is_top else 8, color="#FFB000" if is_top else "#9FB2BE"),
                        text=["TOP SUSPECT CPA"] if is_top else None, textposition="bottom center",
                        name=f"{row['VesselName']} CPA",
                    ))

        attr_lats = [origin_point[0]] + ais["LAT"].tolist()
        attr_lons = [origin_point[1]] + ais["LON"].tolist()
        lat_r, lon_r = compute_geo_range(attr_lats, attr_lons)
        _apply_geo_layout(fig_attr, lat_r, lon_r, height=580)
        st.plotly_chart(fig_attr, use_container_width=True)

with tab_3d:
    render_3d_ocean_map(active_seep if is_natural_seep_mode else KNOWN_NATURAL_SEEPS[0])

with tab_bath:
    st.caption("Distribution of 26 real natural seeps and 28 offshore production platforms.")
    fig_bath = go.Figure()
    fig_bath.add_trace(go.Scattergeo(
        lat=[s["lat"] for s in KNOWN_NATURAL_SEEPS], lon=[s["lon"] for s in KNOWN_NATURAL_SEEPS],
        mode="markers+text", marker=dict(size=9, symbol="triangle-up", color="#38BDF8"),
        text=[s["name"].split("(")[0].strip() for s in KNOWN_NATURAL_SEEPS], textposition="top right", name="Real Natural Seeps (26 Sites)",
    ))
    fig_bath.add_trace(go.Scattergeo(
        lat=[p["latitude"] for p in KNOWN_OFFSHORE_PLATFORMS], lon=[p["longitude"] for p in KNOWN_OFFSHORE_PLATFORMS],
        mode="markers", marker=dict(size=8, symbol="square", color="#00FFA3"), name="Offshore Platforms (28 Sites)",
    ))
    fig_bath.add_trace(go.Scattergeo(lat=[centroid[0]], lon=[centroid[1]], mode="markers+text", marker=dict(size=14, symbol="star", color="#FFB000"), text=["SLICK"], name="Incident"))
    fig_bath.update_geos(projection_type="equirectangular", showcoastlines=True, coastlinecolor="#3B5664", landcolor="#0B161E", oceancolor="#06121A", fitbounds="locations")
    fig_bath.update_layout(height=580, margin=dict(l=0, r=0, t=5, b=0), paper_bgcolor="#061017", font=dict(color="#DCE8EE"))
    st.plotly_chart(fig_bath, use_container_width=True)

if is_natural_seep_mode:
    with tab_sonar:
        st.caption("Acoustic volume backscatter (Sv in dB) profile of ascending bubble flare.")
        depth_axis = np.linspace(0, active_seep["depth_m"], 100)
        time_axis = np.linspace(0, 30, 120)
        tt, dd = np.meshgrid(time_axis, depth_axis)
        flare_c = 15.0 + (dd / active_seep["depth_m"]) * 3.5
        flare_w = 1.2 + (active_seep["depth_m"] - dd) / active_seep["depth_m"] * 2.0
        backscatter = -80.0 + 45.0 * np.exp(-((tt - flare_c)**2) / (2 * flare_w**2)) + np.random.normal(0, 2.0, tt.shape)

        fig_sonar = go.Figure(go.Heatmap(x=time_axis, y=depth_axis, z=backscatter, colorscale="Viridis", colorbar=dict(title="Sv (dB)")))
        fig_sonar.update_yaxes(autorange="reversed", title="Water Depth (m)", gridcolor="#17303B")
        fig_sonar.update_xaxes(title="Survey Duration (min)", gridcolor="#17303B")
        fig_sonar.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#061017", plot_bgcolor="#091924", font=dict(color="#DCE8EE"))
        st.plotly_chart(fig_sonar, use_container_width=True)


# ---------------------------------------------------------------------
# STEP 3: AIS ATTRIBUTION & LEGAL DEMARCATION (SHIP-BASE EVIDENCE)
# ---------------------------------------------------------------------

st.markdown('<div class="step-banner"><span>STEP 3</span> AIS VESSEL TRAFFIC & SHIP-BASE EVIDENCE DOSSIER</div>', unsafe_allow_html=True)

if is_natural_seep_mode:
    st.markdown(
        f"""
        <div class="alert alert-good" style="border-left: 6px solid #36D399; padding: 16px 20px;">
            <h4 style="margin: 0 0 6px 0; color: #36D399;">VESSEL ATTRIBUTION CATEGORY SUPPRESSED - LEGAL EXONERATION</h4>
            <b>INCIDENT CLASSIFICATION: GEOGENIC NATURAL COLD SEEP (NON-ANTHROPOGENIC)</b><br>
            - <b>Target Identification:</b> Reconstructed origin coordinates align directly with documented seafloor natural cold seep <b>{active_seep['name']}</b> at <b>-{active_seep['depth_m']} meters depth</b>.<br>
            - <b>Vessel Attribution Status:</b> <b>Completely dismissed</b>. Passing commercial vessels are cleared of illicit discharge.<br>
            - <b>Legal Finding:</b> Commercial maritime traffic is formally cleared of liability under <b>MARPOL 73/78 Annex I</b> and <b>US Clean Water Act 311</b>.<br>
            - <b>Forensic Policy:</b> Suppressing the suspect vessel category prevents wrongful port state detentions and maritime fines.
        </div>
        """,
        unsafe_allow_html=True,
    )
    p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
    p_c1.metric("WATER DEPTH", f"{active_seep['depth_m']} m", "Seafloor origin")
    p_c2.metric("SLICK EXTENT", f"{spill_data['area_sqkm']:.2f} km2", f"{spill_data['area_sqm']:,.0f} m2")
    p_c3.metric("THICKNESS PROFILE", f"{spill_data['depth_um']:.1f} um", f"{spill_data['depth_mm']:.4f} mm")
    p_c4.metric("NATURAL FLUX", f"{active_seep.get('flux_bbl_day', 40):.0f} bbl/day", "Geogenic venting")
    p_c5.metric("GEOLOGY", active_seep.get("formation", "Faulted seep"), "Reservoir structure")

elif is_platform_mode:
    st.markdown(
        f"""
        <div class="alert alert-warn" style="border-left: 6px solid #FFB000; padding: 16px 20px;">
            <h4 style="margin: 0 0 6px 0; color: #FFB000;">VESSEL ATTRIBUTION REDIRECTED - PLATFORM INTEGRITY AUDIT</h4>
            <b>INCIDENT CLASSIFICATION: OFFSHORE PLATFORM INFRASTRUCTURE LEAK</b><br>
            - <b>Target Identification:</b> Origin intersects production perimeter of <b>{active_platform['platform_name']}</b> ({active_platform['operator']}).<br>
            - <b>Vessel Attribution Status:</b> Suspended pending verification of subsea export pipelines, risers, and blowout preventer manifolds.<br>
            - <b>Recommended Action:</b> Issue BSEE / NPD structural integrity notice. Deploy ROV acoustic leak detection team.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    left, middle, right = st.columns([1.15, 1.7, 1.0])

    with left:
        st.markdown("**Evidence Log**")
        st.markdown(
            f"""
            <div class="alert">
                <b>[1] SAR OBSERVATION:</b> Slick of {spill_data['area_sqkm']:.2f} km2 detected at {centroid[0]:.4f} N, {centroid[1]:.4f} W.<br>
                <b>[2] DRIFT RELEASE:</b> Current ({current_speed:.2f} m/s @ {current_dir:.0f} deg) points to release around {spill_time.strftime('%H:%M UTC')}.<br>
                <b>[3] HINDCAST ORIGIN:</b> Reconstructed source at {origin_point[0]:.4f} N, {origin_point[1]:.4f} W.<br>
                <b>[4] AIS CORRELATION:</b> {len(suspects)} vessel(s) evaluated, {len(suspects[suspects['Min_Distance_km'] <= search_radius]) if not suspects.empty else 0} within {search_radius} km.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with middle:
        st.markdown("**Attribution Matrix**")
        if not suspects.empty:
            disp = suspects[["VesselName", "VesselType", "Min_Distance_km", "Time_Offset_hrs", "Speed_at_CPA", "Risk_Score"]].copy()
            disp.columns = ["VESSEL", "TYPE", "DIST km", "dT h", "SPEED kt", "RISK SCORE"]
            st.dataframe(disp.style.format({c: "{:.2f}" for c in ["DIST km", "dT h", "SPEED kt", "RISK SCORE"]}), use_container_width=True, hide_index=True, height=190)

    with right:
        st.markdown("**Top Vessel Profile**")
        if top_suspect is not None:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">TOP SUSPECT</div>
                    <div class="metric-value" style="font-size:18px">{top_suspect['VesselName']}</div>
                    <div class="metric-note">{top_suspect['VesselType']} - MMSI {top_suspect['MMSI']}</div>
                    <div style="font-size:13px; margin-top:5px; color:#FFB000;">Risk Score: <b>{top_suspect['Risk_Score']} / 100</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------
    # EXPANDED SHIP-BASE FORENSIC EVIDENCE (SPEED, DISTANCE, KINEMATICS)
    # -------------------------------------------------------------
    if top_suspect is not None:
        st.markdown("#### Kinematic Forensic Evidence Dossier - Top Suspect")
        suspect_track = ais[ais["MMSI"] == top_suspect["MMSI"]].sort_values("BaseDateTime").copy()

        if not suspect_track.empty:
            suspect_track["Distance_to_Origin_km"] = [
                haversine_km(r["LAT"], r["LON"], origin_point[0], origin_point[1])
                for _, r in suspect_track.iterrows()
            ]
            cpa_idx = int(suspect_track["Distance_to_Origin_km"].argmin())
            cpa_record = suspect_track.iloc[cpa_idx]
            cpa_time = cpa_record["BaseDateTime"]
            cpa_dist = float(cpa_record["Distance_to_Origin_km"])
            cpa_spd = float(cpa_record["SOG"])
            median_spd = float(suspect_track["SOG"].median())
            max_spd = float(suspect_track["SOG"].max())
            spd_drop_pct = ((median_spd - cpa_spd) / median_spd * 100.0) if median_spd > 0 else 0.0

            # Kinematic telemetry KPI indicators
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("CLOSEST APPROACH (CPA)", f"{cpa_dist:.2f} km", f"{top_suspect['Time_Offset_hrs']:.1f} h from spill")
            kpi2.metric("SPEED AT CPA", f"{cpa_spd:.1f} knots", f"Normal cruise: {median_spd:.1f} kt")
            kpi3.metric("THROTTLE REDUCTION", f"{spd_drop_pct:.1f}%", "Kinematic dump anomaly" if spd_drop_pct > 25 else "Standard passage")
            kpi4.metric("PEAK TRANSIT SPEED", f"{max_spd:.1f} knots", f"MMSI: {top_suspect['MMSI']}")
            kpi5.metric("AIS TELEMETRY INTEGRITY", "Flagged Gap" if bool(cpa_record.get("AIS_Gap_Flag", 0)) else "Continuous", "Transponder status")

            # Synchronized kinematic multi-plot
            fig_spd = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    f"Speed Over Ground Profile (Knots) - {top_suspect['VesselName']}",
                    "Proximity to Hydrodynamic Spill Origin (km)"
                )
            )

            # Row 1: Speed over time
            fig_spd.add_trace(
                go.Scatter(
                    x=suspect_track["BaseDateTime"],
                    y=suspect_track["SOG"],
                    mode="lines+markers",
                    name="Speed (SOG)",
                    line=dict(color="#FFB000", width=2.5),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            fig_spd.add_hline(
                y=median_spd,
                line_dash="dot",
                line_color="#8299A6",
                annotation_text=f"Median Cruise Speed ({median_spd:.1f} kt)",
                row=1, col=1
            )
            fig_spd.add_trace(
                go.Scatter(
                    x=[cpa_time],
                    y=[cpa_spd],
                    mode="markers+text",
                    name="Speed at CPA",
                    marker=dict(color="#FF4D5A", size=11, symbol="diamond"),
                    text=[f"CPA: {cpa_spd:.1f} kt"],
                    textposition="top center"
                ),
                row=1, col=1
            )

            # Row 2: Distance from spill origin over time
            fig_spd.add_trace(
                go.Scatter(
                    x=suspect_track["BaseDateTime"],
                    y=suspect_track["Distance_to_Origin_km"],
                    mode="lines+markers",
                    name="Distance to Origin",
                    line=dict(color="#45D6FF", width=2.5),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
            fig_spd.add_hline(
                y=search_radius,
                line_dash="dot",
                line_color="#38BDF8",
                annotation_text=f"Screening Search Radius ({search_radius} km)",
                row=2, col=1
            )
            fig_spd.add_trace(
                go.Scatter(
                    x=[cpa_time],
                    y=[cpa_dist],
                    mode="markers+text",
                    name="CPA Distance",
                    marker=dict(color="#FF4D5A", size=11, symbol="star"),
                    text=[f"Min Dist: {cpa_dist:.2f} km"],
                    textposition="bottom center"
                ),
                row=2, col=1
            )

            # Highlight estimated release time window
            fig_spd.add_vrect(
                x0=spill_time - timedelta(hours=1),
                x1=spill_time + timedelta(hours=1),
                fillcolor="rgba(255, 77, 90, 0.12)",
                layer="below",
                line_width=1,
                line_color="rgba(255, 77, 90, 0.4)",
                annotation_text="Spill Window",
                annotation_position="top left",
                row=1, col=1
            )
            fig_spd.add_vrect(
                x0=spill_time - timedelta(hours=1),
                x1=spill_time + timedelta(hours=1),
                fillcolor="rgba(255, 77, 90, 0.12)",
                layer="below",
                line_width=1,
                line_color="rgba(255, 77, 90, 0.4)",
                row=2, col=1
            )

            fig_spd.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=32, b=10),
                paper_bgcolor="#071018",
                plot_bgcolor="#091720",
                font=dict(color="#DCE8EE", size=10),
                showlegend=False
            )
            fig_spd.update_xaxes(gridcolor="#17303B")
            fig_spd.update_yaxes(gridcolor="#17303B")
            st.plotly_chart(fig_spd, use_container_width=True)

            # Granular waypoint audit log expander
            with st.expander(f"Granular Waypoint Audit Log - {top_suspect['VesselName']} ({len(suspect_track)} records)"):
                track_table = suspect_track[["BaseDateTime", "LAT", "LON", "SOG", "Distance_to_Origin_km"]].copy()
                track_table.columns = ["Timestamp (UTC)", "Latitude", "Longitude", "Speed (knots)", "Distance from Origin (km)"]
                track_table["Timestamp (UTC)"] = track_table["Timestamp (UTC)"].dt.strftime("%Y-%m-%d %H:%M:%S")
                st.dataframe(
                    track_table.style.format({
                        "Latitude": "{:.5f}",
                        "Longitude": "{:.5f}",
                        "Speed (knots)": "{:.1f}",
                        "Distance from Origin (km)": "{:.2f}"
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=200
                )


# ---------------------------------------------------------------------
# STEP 4: REAL OFFSHORE SEEPS & PLATFORMS DATABASE EXPLORER
# ---------------------------------------------------------------------

st.markdown('<div class="step-banner"><span>STEP 4</span> REAL OFFSHORE SEEPS & PLATFORMS DATABASE EXPLORER</div>', unsafe_allow_html=True)

c_card1, c_card2 = st.columns(2)
with c_card1:
    st.markdown(
        f"""
        <div class="alert alert-seep">
            <h4 style="margin:0 0 6px 0; color:#38BDF8;">NEAREST NATURAL SEEP: {active_seep['name']}</h4>
            - <b>Region:</b> {active_seep['region']}<br>
            - <b>Water Depth:</b> <b>{active_seep['depth_m']} meters</b><br>
            - <b>Distance to Incident Origin:</b> <b>{seep_eval['nearest']['dist_km']:.2f} km</b><br>
            - <b>Natural Emission Flux:</b> ~{active_seep.get('flux_bbl_day', 40)} barrels/day<br>
            - <b>Description:</b> {active_seep['desc']}
        </div>
        """,
        unsafe_allow_html=True,
    )
with c_card2:
    st.markdown(
        f"""
        <div class="alert alert-good">
            <h4 style="margin:0 0 6px 0; color:#36D399;">NEAREST OFFSHORE PLATFORM: {active_platform['platform_name']}</h4>
            - <b>Operator:</b> {active_platform['operator']}<br>
            - <b>Structure:</b> {active_platform.get('structure_type', 'Semi-submersible')}<br>
            - <b>Water Depth:</b> <b>{active_platform['water_depth_m']} meters</b><br>
            - <b>Distance to Incident Origin:</b> <b>{platform_eval['nearest']['dist_km']:.2f} km</b><br>
            - <b>Status:</b> {active_platform.get('status', 'Active')} (Integrity Cleared)
        </div>
        """,
        unsafe_allow_html=True,
    )

db_col1, db_col2 = st.columns(2)
with db_col1:
    st.markdown(f"**Natural Seeps Evaluated ({len(seep_eval['table'])} Sites):**")
    seep_reg = st.selectbox("Filter Seeps by Region:", ["All Regions"] + sorted(list(set(s["region"] for s in KNOWN_NATURAL_SEEPS))), key="s_reg")
    s_table = seep_eval["table"].copy()
    if seep_reg != "All Regions":
        s_table = s_table[s_table["Region"] == seep_reg]
    st.dataframe(s_table, use_container_width=True, hide_index=True, height=220)

with db_col2:
    st.markdown(f"**Offshore Platforms Evaluated ({len(platform_eval['table'])} Sites):**")
    rig_reg = st.selectbox("Filter Platforms by Region:", ["All Regions"] + sorted(list(set(p["region"] for p in KNOWN_OFFSHORE_PLATFORMS))), key="r_reg")
    r_table = platform_eval["table"].copy()
    if rig_reg != "All Regions":
        r_table = r_table[r_table["Region"] == rig_reg]
    st.dataframe(r_table, use_container_width=True, hide_index=True, height=220)


# ---------------------------------------------------------------------
# STEP 5: INTERACTIVE SAR INSPECTION & FLEET MIX
# ---------------------------------------------------------------------

st.markdown('<div class="step-banner"><span>STEP 5</span> SAR RASTER PIXEL INSPECTION & FLEET PROFILE</div>', unsafe_allow_html=True)

sar_c1, sar_c2 = st.columns([1.1, 1.0])
with sar_c1:
    st.markdown("**SAR Cross-Sectional Backscatter Profiling**")
    if Path(sar_file).exists():
        raw_img = cv2.imread(str(sar_file), cv2.IMREAD_COLOR)
        if raw_img is not None:
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            row_pick = st.slider("Inspect raster row (px)", 0, h - 1, h // 2, key="sar_rp")

            profile_fig = go.Figure(go.Scatter(y=gray[row_pick, :], mode="lines", line=dict(color="#45D6FF", width=1.8), fill="tozeroy", fillcolor="rgba(69,214,255,0.12)"))
            profile_fig.update_layout(height=170, margin=dict(l=30, r=10, t=10, b=20), paper_bgcolor="#071018", plot_bgcolor="#091720", font=dict(color="#8299A6", size=9), xaxis=dict(title="Column (px)"), yaxis=dict(title="Intensity (0-255)"))
            st.plotly_chart(profile_fig, use_container_width=True)

with sar_c2:
    st.markdown("**Correlated Fleet Mix in Observation Window**")
    v_summary = ais.groupby("MMSI").agg(Type=("VesselType", "first")).reset_index()
    t_counts = v_summary["Type"].value_counts().reset_index()
    t_counts.columns = ["Type", "Count"]
    fleet_fig = go.Figure(go.Pie(labels=t_counts["Type"], values=t_counts["Count"], hole=0.55, marker=dict(colors=["#45D6FF", "#FFB000", "#6B7F87", "#FF4D5A", "#36D399"])))
    fleet_fig.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#071018", font=dict(color="#DCE8EE", size=10))
    st.plotly_chart(fleet_fig, use_container_width=True)


# ---------------------------------------------------------------------
# STEP 6: OFFICIAL EVIDENCE DOSSIERS & MULTI-FORMAT EXPORT
# ---------------------------------------------------------------------

st.markdown('<div class="step-banner"><span>STEP 6</span> OFFICIAL EVIDENCE DOSSIERS & LEGAL EXPORTS</div>', unsafe_allow_html=True)

d1, d2 = st.columns(2)
with d1:
    pdf_bytes = generate_reliable_pdf_report(
        centroid, origin_point, spill_time, top_suspect, seep_eval, platform_eval,
        is_natural_seep=is_natural_seep_mode, seep_details=active_seep, spill_data=spill_data,
    )
    pdf_filename = (
        f"OilTrace_Natural_Seep_Exoneration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        if is_natural_seep_mode
        else f"OilTrace_MARPOL_Evidence_Dossier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    st.download_button(
        label="Download Official Evidence Dossier (PDF)",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        use_container_width=True,
    )

    try:
        yaml_str = generate_yaml_report(
            spill_data, origin_point, spill_time, future_point, future_time,
            current_speed, current_dir, suspects, min_lat, max_lat, min_lon, max_lon,
            forecast_hours, spill_source, ais_source,
            seep_eval=seep_eval, platform_eval=platform_eval, night_check=night_check,
        )
    except TypeError:
        yaml_str = generate_yaml_report(
            spill_data, origin_point, spill_time, future_point, future_time,
            current_speed, current_dir, suspects, min_lat, max_lat, min_lon, max_lon,
            forecast_hours, spill_source=spill_source, ais_source=ais_source,
            seep_eval=seep_eval, platform_eval=platform_eval, night_check=night_check,
            incident_classification="natural_seep" if is_natural_seep_mode else "anthropogenic_spill",
        )

    st.download_button(
        label="Download Forensic Investigation Report (YAML)",
        data=yaml_str,
        file_name=f"OilTrace_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
        mime="application/x-yaml",
        use_container_width=True,
    )

with d2:
    seeps_csv = DATA_DIR / "real_offshore_seeps.csv"
    if seeps_csv.exists():
        with open(seeps_csv, "r", encoding="utf-8") as f:
            st.download_button(
                label="Download Real Offshore Seeps Dataset (26 Sites CSV)",
                data=f.read(),
                file_name="real_offshore_seeps.csv",
                mime="text/csv",
                use_container_width=True,
            )

    rigs_csv = DATA_DIR / "real_offshore_platforms.csv"
    if rigs_csv.exists():
        with open(rigs_csv, "r", encoding="utf-8") as f:
            st.download_button(
                label="Download Real Offshore Platforms Dataset (28 Sites CSV)",
                data=f.read(),
                file_name="real_offshore_platforms.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if not is_natural_seep_mode and not suspects.empty:
        st.download_button(
            label="Download Suspect Vessels Attribution (CSV)",
            data=suspects.to_csv(index=False),
            file_name=f"OilTrace_Suspects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    if "streamlit" not in sys.modules:
        try:
            import streamlit.web.cli as stcli
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
        except Exception:
            pass
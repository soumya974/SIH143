"""
OILTRACE — Maritime Spill Intelligence Dashboard

SAR-based spill detection, Monte Carlo drift reconstruction and AIS vessel
attribution. Spill imagery can come from a synthetic simulation or a real
Sentinel-1 (EODAG) retrieval; AIS vessel traffic can come from a synthetic
simulation or a live real-time feed — chosen independently from the sidebar.

Run:
    streamlit run app.py
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
import math
import json
import asyncio
import time

import numpy as np
import pandas as pd
import re
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
    import credentials  # local, gitignored file — see credentials.py
except ImportError:
    credentials = None


def _cred(attr_name, env_name):
    """Looks up a credential from credentials.py first, then the environment."""
    if credentials is not None:
        value = getattr(credentials, attr_name, "")
        if value:
            return value
    return os.environ.get(env_name, "")


from modules.eodag_loader import fetch_and_preprocess_sentinel
from modules.synthetic import generate_dynamic_dataset
from modules.detection import detect_spill
from modules.trajectory import simulate_drift, fetch_real_ocean_currents
from modules.attribution import score_vessels
from modules.geo_screening import evaluate_natural_seeps, evaluate_offshore_platforms, assess_night_discharge
from modules.yaml_report import generate_yaml_report


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Sagar Oil Sentinel",
    page_icon="OT",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

AIS_FILE = DATA_DIR / "sample_ais.csv"
SAR_FILE = DATA_DIR / "sample_sar.png"

# Default test area — Gulf of Mexico.
DEFAULT_MIN_LAT = 28.000
DEFAULT_MAX_LAT = 29.500
DEFAULT_MIN_LON = -94.000
DEFAULT_MAX_LON = -92.000

# Fixed min/max edge length (degrees) allowed for a box drawn on the map —
# a drawn rectangle is clamped into this range on both axes before use.
MIN_BBOX_SPAN_DEG = 0.2
MAX_BBOX_SPAN_DEG = 6.0

# Default EODAG search result limit (items_per_page) — how many Sentinel-1
# products a satellite search will return.
DEFAULT_EODAG_ITEMS_PER_PAGE = 1

SPILL_SYNTHETIC = "Synthetic"
SPILL_SATELLITE = "Satellite (Sentinel-1)"
AIS_SYNTHETIC = "Synthetic"
AIS_LIVE = "Live (Realtime AIS)"

# Free real-time AIS feed (aisstream.io) — requires a free API key from
# https://aisstream.io. All AIS/SAR outputs are written to ./data on
# whatever machine runs this Streamlit app (your local system, or wherever
# you deploy it) — nothing is uploaded elsewhere by this app.
AISSTREAM_WS_URL = "wss://stream.aisstream.io/v0/stream"


# ---------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp {
        background: #071018;
        color: #E8F0F5;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: #09141D;
        border-right: 1px solid #20313D;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1700px;
    }

    .hero {
        padding: 18px 22px;
        border: 1px solid #29404D;
        border-radius: 14px;
        background:
            radial-gradient(circle at 85% 20%, rgba(0, 190, 255, .13), transparent 28%),
            linear-gradient(135deg, #0B1923, #081118);
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .hero-sub {
        color: #9FB2BE;
        font-size: 13px;
        margin-top: 4px;
    }

    .pill {
        display: inline-block;
        border: 1px solid #31515F;
        border-radius: 999px;
        padding: 5px 10px;
        margin-right: 6px;
        margin-top: 6px;
        font-size: 11px;
        color: #B9D5E1;
        background: #0B1C26;
    }

    .section-title {
        font-size: 17px;
        font-weight: 750;
        margin: 9px 0 8px 0;
        color: #F2F7FA;
    }

    .metric-card {
        border: 1px solid #263C49;
        border-radius: 12px;
        padding: 13px;
        background: #0A1720;
        min-height: 94px;
    }

    .metric-label {
        color: #8299A6;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .8px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 800;
        margin-top: 6px;
    }

    .metric-note {
        color: #8299A6;
        font-size: 10px;
        margin-top: 2px;
    }

    .alert {
        border-left: 4px solid #FFB000;
        background: #17160D;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 5px 0;
        color: #D9E5EA;
        font-size: 12px;
    }

    .danger { border-left-color: #FF4D5A; }
    .good { border-left-color: #36D399; }

    .timeline {
        border: 1px solid #263C49;
        border-radius: 12px;
        padding: 12px 14px;
        background: #0A1720;
    }

    .timeline-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 8px 0;
        font-size: 12px;
    }

    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #45D6FF;
        box-shadow: 0 0 10px rgba(69,214,255,.55);
        flex-shrink: 0;
    }

    .dot-red {
        background: #FF5964;
        box-shadow: 0 0 10px rgba(255,89,100,.5);
    }

    .small { color: #8299A6; font-size: 10px; }

    div[data-testid="stMetric"] {
        background: #0A1720;
        border: 1px solid #263C49;
        border-radius: 12px;
        padding: 10px;
    }

    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #263C49;
    }

    .src-caption {
        color: #7E96A3;
        font-size: 11px;
        margin: 2px 0 10px 0;
    }

    /* Investigation-map tab bar — custom segmented-control look */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 6px;
        background: #0A1720;
        border: 1px solid #20313D;
        border-radius: 12px;
        padding: 5px;
        width: fit-content;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        height: 38px;
        padding: 0 20px;
        background: transparent;
        border: none;
        border-radius: 8px;
        color: #8299A6;
        font-size: 13px;
        font-weight: 650;
        letter-spacing: .3px;
        transition: all .15s ease;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
        background: #10222E;
        color: #DCE8EE;
    }
    div[data-testid="stTabs"] [aria-selected="true"] {
        background: #45D6FF !important;
        color: #071018 !important;
        font-weight: 750;
    }
    div[data-testid="stTabs"] [aria-selected="true"]:hover {
        background: #45D6FF !important;
        color: #071018 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------

def button_group(label, caption, options, key, icons=None):
    """Row of toggle-style buttons; returns the currently selected option."""
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
# DATA HELPERS — SYNTHETIC / REAL AIS
# ---------------------------------------------------------------------

def fetch_live_ais(api_key, min_lat, max_lat, min_lon, max_lon, duration_seconds=30):
    """
    Connects to the free aisstream.io real-time AIS websocket feed for
    duration_seconds, collecting position/static reports inside the bounding
    box, and writes them to data/sample_ais.csv on the local machine running
    this app — the same file the synthetic AIS path uses.

    Requires a free API key from https://aisstream.io (no cost, sign-up only)
    and the 'websockets' package (pip install websockets).
    """
    if not api_key:
        raise ValueError(
            "A free aisstream.io API key is required for live AIS data. "
            "Sign up at https://aisstream.io to get one, then paste it into the sidebar."
        )
    if websockets is None:
        raise RuntimeError(
            "The 'websockets' package is required for live AIS data. Install it with: pip install websockets"
        )

    records = {}

    async def _collect():
        subscribe_message = {
            "APIKey": api_key,
            "BoundingBoxes": [[[min_lat, min_lon], [max_lat, max_lon]]],
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
        }

        # aisstream.io requires deflate compression on the socket — omitting it
        # is the usual cause of a connection that "just doesn't work".
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

                # aisstream.io reports subscription problems (bad key, malformed
                # bounding box, etc.) as a JSON error payload rather than closing
                # the socket — surface it clearly instead of silently ignoring it.
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
    except websockets.exceptions.ConnectionClosed as exc:
        raise RuntimeError(
            f"The aisstream.io connection closed unexpectedly ({exc}). "
            "Double-check the API key, and that the bounding box isn't degenerate "
            "(min/max lat and lon must differ)."
        ) from exc
    except (websockets.exceptions.InvalidHandshake, OSError) as exc:
        raise RuntimeError(
            f"Could not reach aisstream.io ({exc}). Check your internet connection "
            "and that outbound access to wss://stream.aisstream.io is allowed."
        ) from exc

    rows = [r for r in records.values() if r["LAT"] is not None and r["LON"] is not None and r["SOG"] is not None]
    if not rows:
        raise RuntimeError(
            "No live AIS position reports were received in that time window / bounding box. "
            "Try a busier area or a longer capture duration."
        )

    df = pd.DataFrame(rows)
    df.to_csv(AIS_FILE, index=False)
    return str(AIS_FILE), len(df)


def run_data_pipeline(spill_source, ais_source, min_lat, max_lat, min_lon, max_lon, start_date, end_date,
                       eodag_username=None, eodag_password=None, eodag_items_per_page=DEFAULT_EODAG_ITEMS_PER_PAGE,
                       aisstream_api_key=None, live_duration_seconds=30):
    """
    Populates data/sample_sar.png (+ spill_metadata.json) and data/sample_ais.csv
    according to the chosen sources, without letting one source's output clobber
    the other's. Returns (sar_file_path, ais_record_count).
    """
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


GO_TIMESTAMP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<frac>\.\d+)?"
    r"(?:\s*(?P<offset>[+-]\d{2}:?\d{2}))?"
    r"(?:\s*[A-Za-z]{2,5})?\s*$"
)


def _normalize_go_timestamp(raw):
    """
    aisstream.io (written in Go) serializes timestamps using Go's
    time.Time string format, e.g. '2026-09-10 11:02:31.555441947 +0000 UTC'
    — nanosecond precision (Python's datetime only holds microseconds) plus a
    trailing zone name ("UTC") that no strptime format recognizes. Rewrite
    that into standard ISO-8601 microsecond precision so it parses cleanly;
    strings that don't match this pattern are returned unchanged.
    """
    match = GO_TIMESTAMP_RE.match(raw.strip())
    if not match:
        return raw

    date_part = match.group("date")
    time_part = match.group("time")
    frac = match.group("frac")
    offset = match.group("offset") or "+00:00"

    frac_part = ""
    if frac:
        digits = frac[1:][:6].ljust(6, "0")  # truncate/pad nanoseconds -> microseconds
        frac_part = f".{digits}"

    if ":" not in offset:
        offset = f"{offset[:3]}:{offset[3:]}"

    return f"{date_part}T{time_part}{frac_part}{offset}"


def load_ais_data():
    """Loads data/sample_ais.csv, whichever source produced it, with consistent typing."""
    if not AIS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(AIS_FILE)
    raw_row_count = len(df)

    required = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "VesselName", "VesselType"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("AIS file is missing required columns: " + ", ".join(missing))

    raw_times = df["BaseDateTime"].astype(str).str.strip().map(_normalize_go_timestamp)
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        still_missing = parsed.isna()
        if not still_missing.any():
            break
        parsed.loc[still_missing] = pd.to_datetime(
            raw_times[still_missing], format=fmt, utc=True, errors="coerce"
        )

    still_missing = parsed.isna()
    if still_missing.any():
        parsed.loc[still_missing] = pd.to_datetime(
            raw_times[still_missing], utc=True, errors="coerce"
        )

    df["BaseDateTime"] = parsed
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")

    clean = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON", "SOG"]).copy()

    if clean.empty and raw_row_count > 0:
        bad_time = int(df["BaseDateTime"].isna().sum())
        bad_lat = int(df["LAT"].isna().sum())
        bad_lon = int(df["LON"].isna().sum())
        bad_sog = int(df["SOG"].isna().sum())
        sample = raw_times.iloc[0] if raw_row_count else "n/a"
        raise ValueError(
            f"{AIS_FILE} had {raw_row_count} row(s), but none survived cleaning "
            f"(unparseable BaseDateTime: {bad_time}, LAT: {bad_lat}, LON: {bad_lon}, "
            f"SOG: {bad_sog}). Example raw timestamp: '{sample}'."
        )

    clean = clean.sort_values(["MMSI", "BaseDateTime"])
    return clean


def validate_bbox(min_lat, max_lat, min_lon, max_lon):
    if min_lat >= max_lat:
        raise ValueError("Minimum latitude must be smaller than maximum latitude.")
    if min_lon >= max_lon:
        raise ValueError("Minimum longitude must be smaller than maximum longitude.")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees.")


def bbox_from_drawing(drawing):
    """
    Turns a single GeoJSON rectangle feature into a (min_lat, max_lat, min_lon, max_lon) bounding box.
    """
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


# ---------------------------------------------------------------------
# GEO HELPERS
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


def circle_points(lat, lon, radius_km, count=160):
    points_lat, points_lon = [], []
    for bearing in np.linspace(0, 360, count):
        p_lat, p_lon = project_point(lat, lon, bearing, radius_km)
        points_lat.append(p_lat)
        points_lon.append(p_lon)
    return points_lat, points_lon


def closest_approach(ais_df, mmsi, origin_lat, origin_lon):
    """Real observation (from the loaded AIS track) nearest the reconstructed origin."""
    vessel = ais_df[ais_df["MMSI"] == mmsi]
    if vessel.empty:
        return None
    dists = haversine_km(vessel["LAT"].values, vessel["LON"].values, origin_lat, origin_lon)
    idx = int(np.argmin(dists))
    row = vessel.iloc[idx]
    return float(row["LAT"]), float(row["LON"]), float(dists[idx])


def evidence_strength(top_suspect, candidate_count):
    risk = float(top_suspect["Risk_Score"])
    dist = float(top_suspect["Min_Distance_km"])

    if risk >= 70 and dist <= 5:
        return "HIGH", "danger"
    elif risk >= 40 or dist <= 10:
        return "MODERATE", ""
    else:
        return "LOW", "good"


def compute_geo_range(lats, lons, min_span_deg=1.0, max_span_deg=10.0, pad_frac=0.2):
    lat_arr = np.array(lats, dtype=float)
    lon_arr = np.array(lons, dtype=float)
    lat_arr = lat_arr[~np.isnan(lat_arr)]
    lon_arr = lon_arr[~np.isnan(lon_arr)]

    lat_min, lat_max = float(lat_arr.min()), float(lat_arr.max())
    lon_min, lon_max = float(lon_arr.min()), float(lon_arr.max())

    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2

    lat_span = max(lat_max - lat_min, 0.01) * (1 + pad_frac)
    lon_span = max(lon_max - lon_min, 0.01) * (1 + pad_frac)

    lat_span = min(max(lat_span, min_span_deg), max_span_deg)
    lon_span = min(max(lon_span, min_span_deg), max_span_deg)

    return (
        [lat_center - lat_span / 2, lat_center + lat_span / 2],
        [lon_center - lon_span / 2, lon_center + lon_span / 2],
    )


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------

for key, default in [
    ("pipeline_run", False),
    ("data_ready", False),
    ("sar_file", None),
    ("ais_df", pd.DataFrame()),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

st.markdown(
    """
<div class="hero">
    <div class="hero-title">
        OILTRACE <span style="color:#45D6FF">/</span>
        MARITIME SPILL INTELLIGENCE
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("### PIPELINE CONTROLS")

    st.markdown("**1 · Incident geometry**")

    # Apply staged bounding box values if updated by the map drawing picker
    if "pending_bbox" in st.session_state:
        for k, v in st.session_state.pop("pending_bbox").items():
            st.session_state[k] = v

    for _key, _default in (
        ("min_lat", DEFAULT_MIN_LAT), ("max_lat", DEFAULT_MAX_LAT),
        ("min_lon", DEFAULT_MIN_LON), ("max_lon", DEFAULT_MAX_LON),
    ):
        if _key not in st.session_state:
            st.session_state[_key] = _default

    c1, c2 = st.columns(2)
    min_lat = c1.number_input("Min Lat (°)", value=float(st.session_state["min_lat"]), step=0.1, format="%.3f", key="input_min_lat")
    max_lat = c2.number_input("Max Lat (°)", value=float(st.session_state["max_lat"]), step=0.1, format="%.3f", key="input_max_lat")
    c3, c4 = st.columns(2)
    min_lon = c3.number_input("Min Lon (°)", value=float(st.session_state["min_lon"]), step=0.1, format="%.3f", key="input_min_lon")
    max_lon = c4.number_input("Max Lon (°)", value=float(st.session_state["max_lon"]), step=0.1, format="%.3f", key="input_max_lon")

    st.session_state.min_lat = min_lat
    st.session_state.max_lat = max_lat
    st.session_state.min_lon = min_lon
    st.session_state.max_lon = max_lon

    map_picker_available = folium is not None and st_folium is not None

    if st.button("Draw bounding box on map", use_container_width=True, disabled=not map_picker_available):
        st.session_state.show_map_picker = not st.session_state.get("show_map_picker", False)

    if not map_picker_available:
        st.caption("Install `folium` and `streamlit-folium` to draw the box on a map: `pip install folium streamlit-folium`.")

    if st.session_state.get("show_map_picker") and map_picker_available:
        st.caption(
            f"Draw a rectangle on the map. The box is clamped between "
            f"{MIN_BBOX_SPAN_DEG:.1f}° and {MAX_BBOX_SPAN_DEG:.1f}° on each side."
        )
        _center_lat = (st.session_state.min_lat + st.session_state.max_lat) / 2.0
        _center_lon = (st.session_state.min_lon + st.session_state.max_lon) / 2.0

        draw_map = folium.Map(
            location=[_center_lat, _center_lon], zoom_start=6, tiles="cartodbdark_matter",
        )
        Draw(
            export=False,
            draw_options={
                "rectangle": True,
                "polyline": False,
                "polygon": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(draw_map)

        map_state = st_folium(
            draw_map, height=380, use_container_width=True, key="bbox_draw_map",
            returned_objects=["last_active_drawing"],
        )

        drawing = map_state.get("last_active_drawing") if map_state else None
        if drawing:
            _new_min_lat, _new_max_lat, _new_min_lon, _new_max_lon = bbox_from_drawing(drawing)
            st.caption(
                f"Selected box: {_new_min_lat:.3f}° to {_new_max_lat:.3f}° lat, "
                f"{_new_min_lon:.3f}° to {_new_max_lon:.3f}° lon."
            )
            if st.button("Use this box", type="primary", use_container_width=True, key="use_drawn_box"):
                st.session_state["pending_bbox"] = {
                    "min_lat": _new_min_lat,
                    "max_lat": _new_max_lat,
                    "min_lon": _new_min_lon,
                    "max_lon": _new_max_lon,
                    "input_min_lat": _new_min_lat,
                    "input_max_lat": _new_max_lat,
                    "input_min_lon": _new_min_lon,
                    "input_max_lon": _new_max_lon,
                }
                st.session_state.show_map_picker = False
                st.rerun()
        else:
            st.caption("No rectangle drawn yet.")

    st.divider()
    st.markdown("**2 · Data sources**")
    st.caption("Choose the spill imagery source and the AIS tracking source independently.")

    spill_source = button_group(
        "Spill imagery", None,
        [SPILL_SYNTHETIC, SPILL_SATELLITE],
        key="spill_source",
    )

    eodag_username, eodag_password = None, None
    eodag_items_per_page = DEFAULT_EODAG_ITEMS_PER_PAGE
    if spill_source == SPILL_SATELLITE:
        eodag_username = _cred("COP_DATASPACE_USERNAME", "COP_DATASPACE_USERNAME") or None
        eodag_password = _cred("COP_DATASPACE_PASSWORD", "COP_DATASPACE_PASSWORD") or None
        with st.expander("EODAG retrieval settings", expanded=False):
            st.caption(
                "Copernicus Dataspace credentials are read automatically — from "
                "credentials.py, then environment variables, then an existing "
                "~/.config/eodag/eodag.yml on this machine."
            )
            eodag_items_per_page = st.number_input(
                "Product search limit (items per page)", min_value=1, max_value=20,
                value=DEFAULT_EODAG_ITEMS_PER_PAGE, step=1, key="eodag_items_per_page",
                help="Maximum number of Sentinel-1 products EODAG returns for this search.",
            )

    st.write("")
    ais_source = button_group(
        "AIS vessel tracking", None,
        [AIS_SYNTHETIC, AIS_LIVE],
        key="ais_source",
    )

    aisstream_api_key, live_duration_seconds = None, 30
    if ais_source == AIS_LIVE:
        aisstream_api_key = _cred("AISSTREAM_API_KEY", "AISSTREAM_API_KEY") or None
        with st.expander("aisstream.io live feed settings", expanded=False):
            st.caption("API key is read automatically from credentials.py or the AISSTREAM_API_KEY env var.")
            live_duration_seconds = st.slider(
                "Capture duration (seconds)", 10, 120, 30, key="live_duration_seconds",
                help="How long to listen to the live feed before stopping and saving what was received.",
            )

    st.divider()
    st.markdown("**3 · Analysis date range**")
    today = datetime.now(timezone.utc).date()
    date_range = st.date_input(
        "Sentinel-1 search dates",
        value=(today - timedelta(days=1), today),
        max_value=today,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    st.markdown("**4 · Data acquisition**")
    st.caption(f"Imagery: **{spill_source}**  ·  AIS: **{ais_source}**")

    if st.button("Fetch Data and Initialize Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running the data pipeline..."):
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
                    raise RuntimeError("No valid AIS observations were produced by the pipeline.")

                st.session_state.sar_file = sar_file
                st.session_state.ais_df = ais_df
                st.session_state.data_ready = True
                st.session_state.pipeline_run = False

                st.success(f"Pipeline ready — {spill_source} imagery, {ais_count:,} AIS records ({ais_source}).")

            except Exception as exc:
                st.session_state.data_ready = False
                st.session_state.pipeline_run = False
                st.error(f"Error initializing data pipeline: {exc}")

    st.divider()
    st.markdown("**5 · Forecast horizon**")
    forecast_hours = st.slider("Future forecast (hours ahead)", 6, 48, 24)

    st.markdown("**6 · Visualization**")
    show_tracks = st.checkbox("Show vessel tracks", True)
    show_probability = st.checkbox("Show probability zone", True)
    show_search_ring = st.checkbox("Show AIS investigation ring", True)
    show_uncertainty_ring = st.checkbox("Show uncertainty ring", True)
    search_radius = st.slider("AIS search radius (km)", 1, 20, 5)

    st.caption("Map zoom range — how close/far the investigation map can start zoomed.")
    zoom_min_deg, zoom_max_deg = st.slider(
        "Map zoom range (°)", 0.5, 20.0, (1.0, 8.0), step=0.5,
    )

    st.divider()
    if st.button("Run Full Analysis Pipeline", use_container_width=True, disabled=not st.session_state.data_ready):
        st.session_state.pipeline_run = True


# ---------------------------------------------------------------------
# DATA STATUS
# ---------------------------------------------------------------------

if not st.session_state.data_ready:
    st.info(
        "Set the region, date range and data sources in the sidebar, then click "
        "**Fetch Data and Initialize Pipeline**."
    )
    st.stop()

ais = st.session_state.ais_df

if ais.empty:
    st.error("No AIS observations are available.")
    st.stop()


# ---------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------

if st.session_state.pipeline_run:
    with st.spinner("Running SAR detection, drift modelling and AIS attribution..."):
        try:
            sar_file = st.session_state.sar_file
            spill_data = detect_spill(sar_file)
            centroid = spill_data["centroid"]
            age_hours = float(spill_data["estimated_age_hours"])

            current_speed, current_dir = fetch_real_ocean_currents(centroid[0], centroid[1])

            hindcast_path, origin_point = simulate_drift(centroid, age_hours, mode="backward")
            forecast_path, future_point = simulate_drift(centroid, forecast_hours, mode="forward")

            now_utc = datetime.now(timezone.utc)
            spill_time = now_utc - timedelta(hours=age_hours)
            future_time = now_utc + timedelta(hours=forecast_hours)

            suspects = score_vessels(ais, origin_point[0], origin_point[1], spill_time)
            if suspects is None:
                suspects = pd.DataFrame()

            top_suspect = suspects.iloc[0] if not suspects.empty else None

            seep_eval = evaluate_natural_seeps(origin_point[0], origin_point[1], search_radius)
            platform_eval = evaluate_offshore_platforms(origin_point[0], origin_point[1], search_radius)
            night_check = assess_night_discharge(spill_time, origin_point[0], origin_point[1])

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()
else:
    st.info("Data has been fetched. Click **Run Full Analysis Pipeline** in the sidebar to calculate the spill trajectory and vessel attribution.")
    st.stop()


# ---------------------------------------------------------------------
# DERIVED VISUALIZATION DATA
# ---------------------------------------------------------------------

suspect_mmsi = top_suspect["MMSI"] if top_suspect is not None else None

candidate_count = 0
if not suspects.empty and "Min_Distance_km" in suspects.columns:
    candidate_count = int((suspects["Min_Distance_km"] <= search_radius).sum())

future_lat, future_lon = future_point


# ---------------------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------------------

st.markdown('<div class="section-title">OIL SLICK PHYSICAL PROPERTIES</div>', unsafe_allow_html=True)
st.markdown(f'<div class="src-caption">Imagery: <b>{spill_source}</b> &nbsp;·&nbsp; AIS: <b>{ais_source}</b></div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("SURFACE AREA", f"{spill_data['area_sqkm']:.2f} km²", f"{spill_data['area_sqm']:,.0f} m²")
k2.metric("ESTIMATED THICKNESS", f"{spill_data['depth_mm']:.3f} mm", f"{spill_data['depth_um']:.1f} µm")
k3.metric("DISCHARGE VOLUME", f"{spill_data['volume_m3']:,.1f} m³", f"{spill_data['volume_barrels']:,.0f} barrels")
k4.metric("ESTIMATED AGE", f"{age_hours:.1f} h", "SAR detection")
k5.metric("AIS CANDIDATES", f"{candidate_count}", f"within {search_radius} km")

st.divider()


# ---------------------------------------------------------------------
# SEPARATE SPATIO-TEMPORAL MAPS (detection / drift / attribution)
# ---------------------------------------------------------------------

st.markdown('<div class="section-title">SPATIO-TEMPORAL INVESTIGATION MAPS</div>', unsafe_allow_html=True)
st.caption(
    "Each evidence layer gets its own map detected slick geometry, drift/origin reconstruction, "
    "and AIS vessel attribution instead of one overlapping map."
)


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


def _hud_annotation(fig, title, lines, accent="#45D6FF"):
    """Small translucent HUD-style info card pinned to the map's top-left corner."""
    body = "<br>".join(lines)
    fig.add_annotation(
        xref="paper", yref="paper", x=0.012, y=0.98, xanchor="left", yanchor="top",
        align="left", showarrow=False, bordercolor=accent, borderwidth=1, borderpad=8,
        bgcolor="rgba(6,14,20,0.82)",
        text=f"<b style='color:{accent}'>{title}</b><br>{body}",
        font=dict(size=11, color="#DCE8EE"),
    )


polygon_pts = spill_data.get("polygon") or []
polygon_lats = [p[0] for p in polygon_pts]
polygon_lons = [p[1] for p in polygon_pts]
detection_meta = spill_data.get("detection", {}) or {}

tab_detection, tab_drift, tab_attribution = st.tabs([
    "Detection Map", "Drift and Origin Map", "Vessel Attribution Map",
])

# --- TAB 1: DETECTION MAP -------
with tab_detection:
    fig_det = go.Figure()

    if len(polygon_lats) >= 3:
        ring_lat = polygon_lats + [polygon_lats[0]]
        ring_lon = polygon_lons + [polygon_lons[0]]

        # Soft outer glow pass, then a crisp inner boundary — reads much
        # better against the dark basemap than a single flat line.
        fig_det.add_trace(go.Scattergeo(
            lat=ring_lat, lon=ring_lon, mode="lines",
            line=dict(width=9, color="rgba(255,176,0,0.16)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig_det.add_trace(go.Scattergeo(
            lat=ring_lat, lon=ring_lon, mode="lines",
            line=dict(width=2.25, color="#FFC94D"),
            fill="toself", fillcolor="rgba(255,176,0,0.22)",
            name="Detected slick polygon",
            hovertemplate="Detected slick boundary<extra></extra>",
        ))
    else:
        st.warning("No detected polygon available for this run.")

    concentration = spill_data.get("concentration_grid") or []
    if concentration:
        conc_lat = [p[0] for p in concentration]
        conc_lon = [p[1] for p in concentration]
        # Colour samples by distance from centroid so the interior texture
        # of the slick reads visually instead of one flat dot colour.
        conc_dist = [
            haversine_km(la, lo, centroid[0], centroid[1]) for la, lo in zip(conc_lat, conc_lon)
        ]
        fig_det.add_trace(go.Scattergeo(
            lat=conc_lat, lon=conc_lon, mode="markers",
            marker=dict(
                size=5, color=conc_dist, colorscale=[[0, "#FFF3D6"], [0.5, "#FFC94D"], [1, "#B9720C"]],
                opacity=0.65, line=dict(width=0),
                colorbar=dict(title="dist. from<br>centroid (km)", thickness=10, len=0.4, x=1.0, y=0.18),
            ),
            name="Detected dark-spot samples", hoverinfo="skip",
        ))

    # Layered "pulse" halo behind the centroid marker for visual weight.
    for size, opacity in [(34, 0.10), (26, 0.16), (19, 0.22)]:
        fig_det.add_trace(go.Scattergeo(
            lat=[centroid[0]], lon=[centroid[1]], mode="markers",
            marker=dict(size=size, symbol="star", color=f"rgba(255,176,0,{opacity})"),
            hoverinfo="skip", showlegend=False,
        ))
    fig_det.add_trace(go.Scattergeo(
        lat=[centroid[0]], lon=[centroid[1]], mode="markers+text",
        marker=dict(size=13, symbol="star", color="#FFB000", line=dict(width=1.5, color="#071018")),
        text=["SLICK CENTROID"], textposition="top center", textfont=dict(color="#FFE8B0", size=11),
        name="Slick centroid",
        hovertemplate=(
            f"<b>Detection — {detection_meta.get('classification', 'n/a')}</b><br>"
            f"Confidence score: {detection_meta.get('classification_score', 'n/a')}<br>"
            f"Area: {spill_data['area_sqkm']:.2f} km²<br>"
            f"Est. volume: {spill_data['volume_barrels']:,.0f} barrels<br>"
            f"Est. age: {age_hours:.1f} h<br>"
            "Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>"
        ),
    ))

    det_lats = polygon_lats + [centroid[0]]
    det_lons = polygon_lons + [centroid[1]]
    lat_range, lon_range = compute_geo_range(
        det_lats, det_lons, min_span_deg=max(zoom_min_deg * 0.3, 0.2), max_span_deg=zoom_max_deg,
    )
    _apply_geo_layout(fig_det, lat_range, lon_range, height=600)

    classification = detection_meta.get("classification", "n/a")
    conf_score = detection_meta.get("classification_score", None)
    accent_color = (
        "#FF5964" if classification == "likely_lookalike"
        else "#36D399" if classification == "probable_oil_slick"
        else "#45D6FF"
    )
    _hud_annotation(
        fig_det, "SAR DETECTION",
        [
            f"Classification: <b>{classification}</b>",
            f"Confidence: <b>{conf_score if conf_score is not None else 'n/a'}</b>",
            f"Area: <b>{spill_data['area_sqkm']:.2f} km²</b>",
            f"Volume: <b>{spill_data['volume_barrels']:,.0f} bbl</b>",
            f"Age: <b>{age_hours:.1f} h</b>",
        ],
        accent=accent_color,
    )

    st.plotly_chart(fig_det, use_container_width=True, config={"displayModeBar": True})

    badge_class = (
        "danger" if classification == "likely_lookalike"
        else "good" if classification == "probable_oil_slick"
        else ""
    )
    st.markdown(
        f"""
        <div class="alert {badge_class}">
            <b>DETECTION EVIDENCE</b><br>
            Method: <b>{detection_meta.get('method', 'n/a')}</b> ·
            Classification: <b>{classification}</b> ·
            Confidence score: <b>{detection_meta.get('classification_score', 'n/a')}</b><br>
            Backscatter threshold: {detection_meta.get('backscatter_threshold', 'n/a')} ·
            Polygon vertices: {len(polygon_pts)} ·
            Area: {spill_data['area_sqkm']:.2f} km² ·
            Est. volume: {spill_data['volume_barrels']:,.0f} barrels
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Classification descriptors (raw)"):
        st.json(detection_meta.get("descriptors", {}))

# --- TAB 2: DRIFT / ORIGIN RECONSTRUCTION MAP ------------------------------
with tab_drift:
    fig_drift = go.Figure()

    if show_probability:
        base_radius = max(10.0, search_radius * 2)
        contour_rings = [
            (1.00, "30% contour", "rgba(255,176,0,0.30)"),
            (0.65, "60% contour", "rgba(255,176,0,0.55)"),
            (0.35, "85% contour", "rgba(255,176,0,0.85)"),
        ]
        for fraction, label, color in contour_rings:
            clat, clon = circle_points(origin_point[0], origin_point[1], base_radius * fraction)
            fig_drift.add_trace(go.Scattergeo(
                lat=clat, lon=clon, mode="lines", line=dict(width=1.5, color=color),
                name=label, hovertemplate=f"Origin probability — {label}<extra></extra>",
            ))

    if show_uncertainty_ring:
        clat, clon = circle_points(origin_point[0], origin_point[1], 10)
        fig_drift.add_trace(go.Scattergeo(
            lat=clat, lon=clon, mode="lines", line=dict(width=1, dash="dash", color="#5C7A88"),
            name="10 km uncertainty", hoverinfo="skip",
        ))

    if hindcast_path:
        fig_drift.add_trace(go.Scattergeo(
            lat=[p[0] for p in hindcast_path], lon=[p[1] for p in hindcast_path],
            mode="lines", line=dict(width=4, dash="dot", color="#FF4D5A"),
            name="Hindcast / probable source", hovertemplate="Hindcast path<extra></extra>",
        ))

    if forecast_path:
        fig_drift.add_trace(go.Scattergeo(
            lat=[p[0] for p in forecast_path], lon=[p[1] for p in forecast_path],
            mode="lines", line=dict(width=4, color="#45D6FF"),
            name="Future oil drift", hovertemplate="Projected drift<extra></extra>",
        ))

    if len(polygon_lats) >= 3:
        ring_lat = polygon_lats + [polygon_lats[0]]
        ring_lon = polygon_lons + [polygon_lons[0]]
        fig_drift.add_trace(go.Scattergeo(
            lat=ring_lat, lon=ring_lon, mode="lines",
            line=dict(width=1, color="rgba(255,176,0,0.6)"),
            fill="toself", fillcolor="rgba(255,176,0,0.10)",
            name="Detected slick (context)", hoverinfo="skip",
        ))

    fig_drift.add_trace(go.Scattergeo(
        lat=[centroid[0]], lon=[centroid[1]], mode="markers+text",
        marker=dict(size=13, symbol="star", color="#FFB000", line=dict(width=1.5, color="#071018")),
        text=["① SAR DETECTION"], textposition="top center", textfont=dict(color="#FFE8B0", size=11),
        name="1 · Observed slick",
        hovertemplate=(
            f"<b>Evidence 1 — SAR detection</b><br>Area: {spill_data['area_sqkm']:.2f} km²<br>"
            "Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>"
        ),
    ))
    fig_drift.add_trace(go.Scattergeo(
        lat=[origin_point[0]], lon=[origin_point[1]], mode="markers",
        marker=dict(size=26, symbol="circle", color="rgba(255,77,90,0.22)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig_drift.add_trace(go.Scattergeo(
        lat=[origin_point[0]], lon=[origin_point[1]], mode="markers+text",
        marker=dict(size=13, symbol="circle", color="#FF4D5A", line=dict(width=1.5, color="#071018")),
        text=["② HINDCAST ORIGIN"], textposition="bottom center", textfont=dict(color="#FFC2C7", size=11),
        name="2 · Hindcast origin",
        hovertemplate=(
            f"<b>Evidence 2 — Reconstructed origin</b><br>"
            f"Release window: {spill_time.strftime('%Y-%m-%d %H:%M UTC')}<br>"
            f"Current used: {current_speed:.2f} m/s @ {current_dir:.0f}°<br>"
            "Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>"
        ),
    ))
    fig_drift.add_trace(go.Scattergeo(
        lat=[future_lat], lon=[future_lon], mode="markers+text",
        marker=dict(size=11, symbol="diamond", color="#45D6FF", line=dict(width=1.5, color="#071018")),
        text=["③ FORECAST"], textposition="top center", textfont=dict(color="#C4EEFF", size=11),
        name="3 · Forecast position",
        hovertemplate=(
            f"<b>Projection — +{forecast_hours} h</b><br>"
            f"Target time: {future_time.strftime('%Y-%m-%d %H:%M UTC')}<br>"
            "Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>"
        ),
    ))

    drift_lats = [centroid[0], origin_point[0], future_lat] + [p[0] for p in hindcast_path] + [p[0] for p in forecast_path]
    drift_lons = [centroid[1], origin_point[1], future_lon] + [p[1] for p in hindcast_path] + [p[1] for p in forecast_path]
    lat_range, lon_range = compute_geo_range(
        drift_lats, drift_lons, min_span_deg=zoom_min_deg, max_span_deg=zoom_max_deg,
    )
    _apply_geo_layout(fig_drift, lat_range, lon_range)
    st.plotly_chart(fig_drift, use_container_width=True, config={"displayModeBar": True})

    st.markdown(
        f"""
        <div class="alert">
            <b>DRIFT EVIDENCE</b><br>
            Ocean current: <b>{current_speed:.2f} m/s @ {current_dir:.0f}°</b> ·
            Hindcast origin: <b>{origin_point[0]:.5f}, {origin_point[1]:.5f}</b> ·
            Release window: {spill_time.strftime('%Y-%m-%d %H:%M UTC')}<br>
            Forecast (+{forecast_hours} h): <b>{future_lat:.5f}, {future_lon:.5f}</b>
            at {future_time.strftime('%Y-%m-%d %H:%M UTC')}
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- TAB 3: VESSEL ATTRIBUTION MAP -----------------------------------------
with tab_attribution:
    fig_attr = go.Figure()

    if show_search_ring:
        clat, clon = circle_points(origin_point[0], origin_point[1], search_radius)
        fig_attr.add_trace(go.Scattergeo(
            lat=clat, lon=clon, mode="lines", line=dict(width=1.25, dash="dot", color="#9FB2BE"),
            name=f"{search_radius} km AIS zone", hoverinfo="skip",
        ))

    fig_attr.add_trace(go.Scattergeo(
        lat=[origin_point[0]], lon=[origin_point[1]], mode="markers",
        marker=dict(size=12, symbol="circle", color="#FF4D5A", line=dict(width=1.5, color="#071018")),
        name="Reconstructed origin",
        hovertemplate="Reconstructed origin<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>",
    ))

    if show_tracks:
        for mmsi, vessel in ais.groupby("MMSI"):
            vessel = vessel.sort_values("BaseDateTime")
            name = str(vessel["VesselName"].iloc[0])
            is_top = suspect_mmsi is not None and int(mmsi) == int(suspect_mmsi)
            fig_attr.add_trace(go.Scattergeo(
                lat=vessel["LAT"], lon=vessel["LON"], mode="lines",
                line=dict(
                    width=3 if is_top else 1, dash="solid" if is_top else "dot",
                    color="#FFB000" if is_top else "#6B7F87",
                ),
                opacity=0.95 if is_top else 0.5,
                name=f"Suspect vessel — {name}" if is_top else f"{name} (other vessel)",
                hovertemplate=f"{name}<br>MMSI: {mmsi}<br>Lat: %{{lat:.4f}}<br>Lon: %{{lon:.4f}}<extra></extra>",
                showlegend=is_top,
            ))

    if not suspects.empty and "Min_Distance_km" in suspects.columns:
        for _, row in suspects.iterrows():
            if row["Min_Distance_km"] > search_radius:
                continue
            approach = closest_approach(ais, row["MMSI"], origin_point[0], origin_point[1])
            if approach is None:
                continue
            c_lat, c_lon, c_dist = approach
            is_top = suspect_mmsi is not None and int(row["MMSI"]) == int(suspect_mmsi)

            if is_top:
                fig_attr.add_trace(go.Scattergeo(
                    lat=[c_lat], lon=[c_lon], mode="markers",
                    marker=dict(size=24, symbol="circle", color="rgba(255,176,0,0.25)"),
                    hoverinfo="skip", showlegend=False,
                ))

            fig_attr.add_trace(go.Scattergeo(
                lat=[c_lat], lon=[c_lon], mode="markers+text" if is_top else "markers",
                marker=dict(
                    size=12 if is_top else 8, symbol="circle",
                    color="#FFB000" if is_top else "#9FB2BE",
                    line=dict(width=1.5, color="#071018") if is_top else None,
                ),
                text=["④ TOP SUSPECT"] if is_top else None,
                textposition="bottom center", textfont=dict(color="#FFE8B0", size=11),
                name=f"4 · {row['VesselName']} (closest approach)" if is_top else f"{row['VesselName']} closest point",
                hovertemplate=(
                    f"<b>{'Evidence 4 — ' if is_top else ''}{row['VesselName']}</b><br>"
                    f"Distance to origin: {c_dist:.2f} km<br>"
                    f"Time offset: {row['Time_Offset_hrs']:.1f} h<br>"
                    f"Risk score: {row['Risk_Score']:.1f}<extra></extra>"
                ),
                showlegend=is_top,
            ))

    attr_lats = [origin_point[0]] + ais["LAT"].tolist()
    attr_lons = [origin_point[1]] + ais["LON"].tolist()
    lat_range, lon_range = compute_geo_range(
        attr_lats, attr_lons, min_span_deg=zoom_min_deg, max_span_deg=zoom_max_deg,
    )
    _apply_geo_layout(fig_attr, lat_range, lon_range)
    st.plotly_chart(fig_attr, use_container_width=True, config={"displayModeBar": True})

    if top_suspect is not None:
        st.markdown(
            f"""
            <div class="alert">
                <b>VESSEL EVIDENCE — TOP SUSPECT</b><br>
                <b>{top_suspect['VesselName']}</b> ({top_suspect['VesselType']}, MMSI {top_suspect['MMSI']}) ·
                {top_suspect['Min_Distance_km']:.2f} km from origin ·
                {top_suspect['Time_Offset_hrs']:.1f} h time offset ·
                risk score {top_suspect['Risk_Score']:.1f}
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("All evaluated vessels (evidence table)"):
            st.dataframe(suspects, use_container_width=True, hide_index=True)
    else:
        st.info("No vessel could be linked to the reconstructed origin with the current AIS data and search radius.")


# ---------------------------------------------------------------------
# NATURAL SOURCE SCREENING — seeps / platforms / day-night check
# ---------------------------------------------------------------------

st.markdown('<div class="section-title">NATURAL SOURCE SCREENING</div>', unsafe_allow_html=True)
st.caption(
    "Before treating the top AIS lead as a vessel discharge, checks the reconstructed origin against "
    "documented natural seeps and fixed offshore platforms, and flags whether the estimated release "
    "time falls in nautical darkness."
)

badges = [
    ("Natural seeps", seep_eval["label"], seep_eval["flag"]),
    ("Offshore platforms", platform_eval["label"], platform_eval["flag"]),
    ("Release timing", night_check["label"], night_check["flag"]),
]
pills_html = "".join(
    f'<span class="pill{" " if False else ""}" style="{"border-color:#FFB000;color:#FFE6A3;background:#281D06;" if flg else ""}">'
    f"<b>{title}:</b> {label}</span>"
    for title, label, flg in badges
)
st.markdown(f'<div style="margin-bottom:10px;">{pills_html}</div>', unsafe_allow_html=True)

if seep_eval["flag"] or platform_eval["flag"]:
    st.markdown(
        '<div class="alert danger"><b>POSSIBLE NATURAL / INFRASTRUCTURE SOURCE</b><br>'
        'The reconstructed origin falls within the search radius of a documented seep or platform above. '
        'Treat any AIS vessel lead with added caution until this is ruled out.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="alert good"><b>CLEARED FROM KNOWN NATURAL / FIXED SOURCES</b><br>'
        'No documented natural seep or offshore platform lies within the current search radius of the '
        'reconstructed origin.</div>',
        unsafe_allow_html=True,
    )

with st.expander("Full natural seep / platform proximity tables"):
    st.markdown("**Natural seeps evaluated**")
    st.dataframe(seep_eval["table"], use_container_width=True, hide_index=True)
    st.markdown("**Offshore platforms evaluated**")
    st.dataframe(platform_eval["table"], use_container_width=True, hide_index=True)

st.divider()


# ---------------------------------------------------------------------
# LOWER ANALYTICS
# ---------------------------------------------------------------------

left, middle, right = st.columns([1.15, 1.7, 1.0])

with left:
    st.markdown('<div class="section-title">EVIDENCE LOG</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="timeline">
            <div class="timeline-row">
                <span class="dot-red"></span>
                <div><b>1 · SAR OBSERVATION (T0)</b><br>
                <span class="small">
                    {spill_source} imagery detected a slick at {centroid[0]:.5f}, {centroid[1]:.5f}
                    — {spill_data['area_sqkm']:.2f} km², est. {spill_data['volume_barrels']:,.0f} barrels,
                    age ≈ {age_hours:.1f} h
                </span></div>
            </div>
            <div class="timeline-row">
                <span class="dot"></span>
                <div><b>2 · HINDCAST RELEASE WINDOW</b><br>
                <span class="small">
                    Backward drift modelling (current: {current_speed:.2f} m/s @ {current_dir:.0f}°) points to
                    release around {spill_time.strftime('%Y-%m-%d %H:%M UTC')}
                </span></div>
            </div>
            <div class="timeline-row">
                <span class="dot"></span>
                <div><b>3 · RECONSTRUCTED ORIGIN</b><br>
                <span class="small">Probable source at {origin_point[0]:.5f}, {origin_point[1]:.5f}</span></div>
            </div>
            <div class="timeline-row">
                <span class="dot"></span>
                <div><b>4 · AIS CORRELATION</b><br>
                <span class="small">
                    {len(suspects)} vessel(s) evaluated via {ais_source} AIS,
                    {candidate_count} within {search_radius} km of the origin
                </span></div>
            </div>
            <div class="timeline-row">
                <span class="dot"></span>
                <div><b>5 · FORECAST (T+{forecast_hours} h)</b><br>
                <span class="small">Projected position by {future_time.strftime('%Y-%m-%d %H:%M UTC')}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if top_suspect is not None:
        strength_label, strength_class = evidence_strength(top_suspect, candidate_count)
        st.markdown(
            f"""
            <div class="alert {strength_class}">
                <b>INVESTIGATION LEAD — {strength_label} CONFIDENCE</b><br>
                <b>{top_suspect['VesselName']}</b> ({top_suspect['VesselType']}, MMSI {top_suspect['MMSI']})
                — {top_suspect['Min_Distance_km']:.2f} km from reconstructed origin,
                {top_suspect['Time_Offset_hrs']:.1f} h from estimated release, risk score {top_suspect['Risk_Score']:.1f}.
                This is an investigative lead, not proof of responsibility.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert">No vessel could be linked to the reconstructed origin with the current AIS data and search radius.</div>',
            unsafe_allow_html=True,
        )

with middle:
    st.markdown('<div class="section-title">AIS ATTRIBUTION MATRIX</div>', unsafe_allow_html=True)

    if suspects.empty:
        st.info("No vessels could be ranked from the available AIS data.")
    else:
        display = suspects[["VesselName", "VesselType", "Min_Distance_km", "Time_Offset_hrs", "Speed_at_CPA", "Risk_Score"]].copy()
        display = display.rename(columns={
            "VesselName": "VESSEL", "VesselType": "TYPE",
            "Min_Distance_km": "DIST km", "Time_Offset_hrs": "ΔT h",
            "Speed_at_CPA": "SPEED kt", "Risk_Score": "RISK SCORE",
        })
        format_dict = {c: "{:.2f}" for c in display.columns if c not in {"VESSEL", "TYPE"}}
        st.dataframe(display.style.format(format_dict), use_container_width=True, hide_index=True, height=220)

        ranked = suspects.sort_values("Risk_Score", ascending=True).tail(8)
        bar_colors = [
            "#FFB000" if suspect_mmsi is not None and int(m) == int(suspect_mmsi) else "#3B5664"
            for m in ranked["MMSI"]
        ]
        risk_fig = go.Figure(go.Bar(
            x=ranked["Risk_Score"], y=ranked["VesselName"], orientation="h",
            marker=dict(color=bar_colors),
            hovertemplate="%{y}<br>Risk score: %{x:.1f}<extra></extra>",
        ))
        risk_fig.update_layout(
            height=220, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#071018", plot_bgcolor="#071018",
            font=dict(color="#DCE8EE", size=11),
            xaxis=dict(title="Risk score", gridcolor="#17303B", zerolinecolor="#17303B"),
            yaxis=dict(gridcolor="#17303B"),
        )
        st.plotly_chart(risk_fig, use_container_width=True)

with right:
    st.markdown('<div class="section-title">TOP VESSEL PROFILE</div>', unsafe_allow_html=True)

    if top_suspect is None:
        st.info("No attribution lead available.")
    else:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TOP RANKED VESSEL</div>
                <div class="metric-value" style="font-size:18px">{top_suspect['VesselName']}</div>
                <div class="metric-note">{top_suspect['VesselType']} · MMSI {top_suspect['MMSI']}</div>
            </div>
            <br>
            <div class="metric-card">
                <div class="metric-label">PROXIMITY</div>
                <div class="metric-value">{top_suspect['Min_Distance_km']:.2f} km</div>
                <div class="metric-note">closest AIS observation to hindcast origin</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge_max = max(100.0, float(top_suspect["Risk_Score"]) * 1.2)
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(top_suspect["Risk_Score"]),
            number=dict(font=dict(color="#DCE8EE", size=26)),
            gauge=dict(
                axis=dict(range=[0, gauge_max], tickcolor="#8299A6", tickfont=dict(color="#8299A6", size=9)),
                bar=dict(color="#FFB000"),
                bgcolor="#0A1720",
                borderwidth=1, bordercolor="#263C49",
                steps=[
                    dict(range=[0, gauge_max * 0.4], color="#101E27"),
                    dict(range=[gauge_max * 0.4, gauge_max * 0.7], color="#152530"),
                    dict(range=[gauge_max * 0.7, gauge_max], color="#1B2E3A"),
                ],
            ),
        ))
        gauge_fig.update_layout(
            height=170, margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="#071018", font=dict(color="#DCE8EE"),
        )
        st.markdown('<div class="metric-label">RISK SCORE</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown(
            '<div class="metric-note" style="margin-top:-8px">ranking score from attribution module</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# BEHAVIOUR PANEL
# ---------------------------------------------------------------------

if top_suspect is not None:
    st.markdown('<div class="section-title">WHY THIS VESSEL WAS RANKED HIGH</div>', unsafe_allow_html=True)

    behaviour_columns = st.columns(4)
    behaviours = [
        ("01", "PROXIMITY", top_suspect["Min_Distance_km"], "km to reconstructed origin"),
        ("02", "TIME MATCH", top_suspect["Time_Offset_hrs"], "hours from estimated spill time"),
        ("03", "SPEED AT CPA", top_suspect["Speed_at_CPA"], "vessel speed, knots"),
        ("04", "RISK SCORE", top_suspect["Risk_Score"], "composite attribution score"),
    ]

    for col, (number, title, value, note) in zip(behaviour_columns, behaviours):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{number} · {title}</div>
                    <div class="metric-value">{value:.1f}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------
# AIS SPEED PROFILE
# ---------------------------------------------------------------------

if top_suspect is not None:
    st.markdown('<div class="section-title">AIS SPEED / REPORTING TRACE</div>', unsafe_allow_html=True)

    suspect_track = ais[ais["MMSI"] == top_suspect["MMSI"]].sort_values("BaseDateTime")

    if not suspect_track.empty:
        fig2 = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=("Speed over time", "Distance from reconstructed spill origin"),
        )

        fig2.add_trace(go.Scatter(
            x=suspect_track["BaseDateTime"], y=suspect_track["SOG"],
            mode="lines+markers", name="SOG", line=dict(width=2), marker=dict(size=4),
        ), row=1, col=1)

        fig2.add_hline(y=float(suspect_track["SOG"].median()), line_dash="dot",
                        annotation_text="vessel median", row=1, col=1)

        distances = suspect_track.apply(
            lambda r: haversine_km(r["LAT"], r["LON"], origin_point[0], origin_point[1]), axis=1,
        )

        fig2.add_trace(go.Scatter(
            x=suspect_track["BaseDateTime"], y=distances,
            mode="lines+markers", name="Distance", line=dict(width=2), marker=dict(size=4),
        ), row=2, col=1)

        fig2.add_hline(y=search_radius, line_dash="dot",
                        annotation_text=f"{search_radius} km investigation radius", row=2, col=1)

        fig2.update_layout(
            height=440, margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="#071018", plot_bgcolor="#091720",
            font=dict(color="#DCE8EE"), legend=dict(orientation="h"),
        )
        fig2.update_xaxes(gridcolor="#18303A", zerolinecolor="#18303A")
        fig2.update_yaxes(gridcolor="#18303A", zerolinecolor="#18303A")

        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------
# SAR PREVIEW + FORENSIC SUMMARY
# ---------------------------------------------------------------------

sar_col, report_col = st.columns([1, 1.35])

with sar_col:
    st.markdown('<div class="section-title">SAR EVIDENCE</div>', unsafe_allow_html=True)
    sar_path = Path(st.session_state.sar_file)

    if sar_path.exists() and sar_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        raw = cv2.imread(str(sar_path), cv2.IMREAD_COLOR)
        if raw is None:
            st.warning("Could not decode the SAR output for interactive viewing.")
        else:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            view_mode = st.radio(
                "View", ["Backscatter intensity", "RGB composite"],
                horizontal=True, key="sar_view_mode", label_visibility="collapsed",
            )

            fig_sar = go.Figure()
            if view_mode == "Backscatter intensity":
                fig_sar.add_trace(go.Heatmap(
                    z=gray, colorscale="Cividis", reversescale=True,
                    colorbar=dict(title="backscatter<br>(0-255)", thickness=10),
                    hovertemplate="row %{y}, col %{x}<br>intensity: %{z}<extra></extra>",
                ))
            else:
                fig_sar.add_trace(go.Image(z=rgb, hovertemplate="row %{y}, col %{x}<extra></extra>"))

            row_pick = st.slider("Inspect row (px)", 0, h - 1, h // 2, key="sar_row_pick")
            fig_sar.add_shape(
                type="line", x0=0, x1=w - 1, y0=row_pick, y1=row_pick,
                line=dict(color="#45D6FF", width=1.5, dash="dot"),
            )

            fig_sar.update_yaxes(autorange="reversed", showgrid=False, visible=False)
            fig_sar.update_xaxes(showgrid=False, visible=False)
            fig_sar.update_layout(
                height=340, margin=dict(l=0, r=0, t=5, b=0),
                paper_bgcolor="#071018", plot_bgcolor="#071018",
                font=dict(color="#DCE8EE", size=10),
            )
            st.plotly_chart(fig_sar, use_container_width=True, config={"displayModeBar": True})

            profile_fig = go.Figure(go.Scatter(
                y=gray[row_pick, :], mode="lines", line=dict(color="#45D6FF", width=1.5),
                fill="tozeroy", fillcolor="rgba(69,214,255,0.12)",
            ))
            profile_fig.update_layout(
                height=140, margin=dict(l=30, r=10, t=8, b=20),
                paper_bgcolor="#071018", plot_bgcolor="#091720",
                font=dict(color="#8299A6", size=9),
                xaxis=dict(title="column (px)", gridcolor="#18303A"),
                yaxis=dict(title="intensity", gridcolor="#18303A", range=[0, 255]),
            )
            st.plotly_chart(profile_fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                f"{spill_source} SAR output · {w}×{h} px · drag to zoom, hover for pixel intensity, "
                "drag the row slider to trace a backscatter cross-section."
            )
    elif sar_path.exists():
        st.info(f"SAR raster prepared: `{sar_path}`")
    else:
        st.warning("The SAR output file is no longer available.")

with report_col:
    st.markdown('<div class="section-title">INVESTIGATION SNAPSHOT</div>', unsafe_allow_html=True)

    lead_text = (
        f"<b>{top_suspect['VesselName']}</b> ({top_suspect['VesselType']}, MMSI {top_suspect['MMSI']}) "
        f"ranks highest — {top_suspect['Min_Distance_km']:.2f} km / {top_suspect['Time_Offset_hrs']:.1f} h "
        f"from the reconstructed origin, risk score {top_suspect['Risk_Score']:.1f}."
        if top_suspect is not None else "No attribution lead was produced."
    )

    st.markdown(
        f"""
        <div class="alert">
            <b>① DETECTION</b><br>
            {spill_source} SAR processing detected a slick of {spill_data['area_sqkm']:.2f} km²
            (est. {spill_data['volume_barrels']:,.0f} barrels) at
            <b>{centroid[0]:.5f}, {centroid[1]:.5f}</b>.
        </div>
        <div class="alert">
            <b>② HINDCAST ORIGIN</b><br>
            Backward drift modelling (current {current_speed:.2f} m/s @ {current_dir:.0f}°) estimated a
            probable origin at <b>{origin_point[0]:.5f}, {origin_point[1]:.5f}</b> around
            {spill_time.strftime('%Y-%m-%d %H:%M UTC')}.
        </div>
        <div class="alert">
            <b>③ FORECAST</b><br>
            Projected position by {future_time.strftime('%Y-%m-%d %H:%M UTC')}
            (+{forecast_hours} h): <b>{future_lat:.5f}, {future_lon:.5f}</b>.
        </div>
        <div class="alert">
            <b>AIS CORRELATION</b><br>
            {len(suspects)} vessel(s) evaluated using {ais_source} AIS observations,
            {candidate_count} within {search_radius} km of the origin.
        </div>
        <div class="alert">
            <b>④ TOP LEAD</b><br>
            {lead_text}
        </div>
        <div class="alert good">
            <b>IMPORTANT</b><br>
            Attribution is an investigative prioritisation signal, not a legal determination of liability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    yaml_report_text = generate_yaml_report(
        spill_data, origin_point, spill_time, future_point, future_time,
        current_speed, current_dir, suspects, min_lat, max_lat, min_lon, max_lon,
        forecast_hours, spill_source, ais_source,
        seep_eval=seep_eval, platform_eval=platform_eval, night_check=night_check,
    )

    st.download_button(
        label="Download Investigation Report (YAML)",
        data=yaml_report_text,
        file_name=f"oil_spill_investigation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
        mime="application/x-yaml",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# FLEET INVENTORY
# ---------------------------------------------------------------------

st.markdown('<div class="section-title">AIS FLEET INVENTORY</div>', unsafe_allow_html=True)

vessel_summary = (
    ais.groupby("MMSI")
    .agg(VesselName=("VesselName", "first"), Type=("VesselType", "first"),
         PointsLogged=("LAT", "count"), AvgSpeed=("SOG", "mean"))
    .reset_index()
)

fleet_table_col, fleet_chart_col = st.columns([1.6, 1.0])

with fleet_table_col:
    st.dataframe(vessel_summary.style.format({"AvgSpeed": "{:.1f} kt"}), use_container_width=True, hide_index=True)

with fleet_chart_col:
    type_counts = vessel_summary["Type"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    fleet_fig = go.Figure(go.Pie(
        labels=type_counts["Type"], values=type_counts["Count"],
        hole=0.55,
        marker=dict(colors=["#45D6FF", "#FFB000", "#6B7F87", "#FF4D5A", "#36D399"]),
        textfont=dict(color="#071018", size=11),
        hovertemplate="%{label}<br>%{value} vessel(s)<extra></extra>",
    ))
    fleet_fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#071018", font=dict(color="#DCE8EE", size=11),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        annotations=[dict(text="Fleet mix", showarrow=False, font=dict(color="#8299A6", size=11))],
    )
    st.plotly_chart(fleet_fig, use_container_width=True)
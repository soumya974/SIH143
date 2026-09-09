"""
OILTRACE — Maritime Spill Intelligence Dashboard

SAR-based spill detection, Monte Carlo drift reconstruction and AIS vessel
attribution. Spill imagery can come from a synthetic simulation or a real
Sentinel-1 (EODAG) retrieval; AIS vessel traffic can come from a synthetic
simulation or the real NOAA Marine Cadastre archive — chosen independently
from the sidebar.

Run:
    streamlit run app.py
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import io
import os
import math
import zipfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.eodag_loader import fetch_and_preprocess_sentinel
from modules.real_ais import fetch_marine_cadastre_data
from modules.synthetic import generate_dynamic_dataset
from modules.detection import detect_spill
from modules.trajectory import simulate_drift, fetch_real_ocean_currents
from modules.attribution import score_vessels

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="OILTRACE — Maritime Spill Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

AIS_FILE = DATA_DIR / "sample_ais.csv"
SAR_FILE = DATA_DIR / "sample_sar.png"

# Default test area — Gulf of Mexico, inside NOAA Marine Cadastre coverage.
DEFAULT_MIN_LAT = 28.000
DEFAULT_MAX_LAT = 29.500
DEFAULT_MIN_LON = -94.000
DEFAULT_MAX_LON = -92.000

SPILL_SYNTHETIC = "Synthetic"
SPILL_SATELLITE = "Satellite (Sentinel-1)"
AIS_SYNTHETIC = "Synthetic"
AIS_REAL = "Real (Marine Cadastre)"


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

def ensure_raw_ais_downloaded(target_date):
    """Downloads and caches the NOAA Marine Cadastre daily AIS archive for
    target_date, skipping the download if already cached for that date."""
    os.makedirs("data", exist_ok=True)
    raw_path = os.path.join("data", "raw_ais.csv")
    marker_path = os.path.join("data", "raw_ais_date.txt")
    date_str = target_date.strftime("%Y_%m_%d")

    if os.path.exists(raw_path) and os.path.exists(marker_path):
        with open(marker_path, "r") as f:
            if f.read().strip() == date_str:
                return raw_path

    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{target_date.year}/AIS_{date_str}.zip"
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_candidates = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if not csv_candidates:
            raise FileNotFoundError("No CSV file found inside the NOAA AIS archive.")
        with z.open(csv_candidates[0]) as f:
            keep_cols = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "VesselName", "VesselType"]
            df = pd.read_csv(f)
            rename_map = {
                "Latitude": "LAT", "Longitude": "LON",
                "Vessel_Name": "VesselName", "Vessel_Type": "VesselType",
            }
            df = df.rename(columns=rename_map)
            df = df[[c for c in keep_cols if c in df.columns]]

    df.to_csv(raw_path, index=False)
    with open(marker_path, "w") as f:
        f.write(date_str)
    return raw_path


def run_data_pipeline(spill_source, ais_source, min_lat, max_lat, min_lon, max_lon, start_date, end_date):
    """
    Populates data/sample_sar.png (+ spill_metadata.json) and data/sample_ais.csv
    according to the chosen sources, without letting one source's output clobber
    the other's. Returns (sar_file_path, ais_record_count).
    """
    # Synthetic generation covers whichever side(s) requested it. Run first so a
    # real satellite fetch / real AIS fetch below can overwrite only their own
    # output file afterwards, without wiping out the other side's data.
    if spill_source == SPILL_SYNTHETIC or ais_source == AIS_SYNTHETIC:
        generate_dynamic_dataset(min_lat, max_lat, min_lon, max_lon, age_hours=12.0)

    if spill_source == SPILL_SATELLITE:
        result = fetch_and_preprocess_sentinel(min_lat, max_lat, min_lon, max_lon, start_date, end_date)
        sar_file = str(result["overview"]) if isinstance(result, dict) and result.get("overview") else str(SAR_FILE)
    else:
        sar_file = str(SAR_FILE)

    if ais_source == AIS_REAL:
        ensure_raw_ais_downloaded(start_date)
        _, ais_count = fetch_marine_cadastre_data(
            min_lat, max_lat, min_lon, max_lon,
            target_date=start_date.strftime("%Y_%m_%d"),
        )
    else:
        ais_count = len(pd.read_csv(AIS_FILE)) if AIS_FILE.exists() else 0

    return sar_file, ais_count


def load_ais_data():
    """Loads data/sample_ais.csv, whichever source produced it, with consistent typing."""
    if not AIS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(AIS_FILE)

    required = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "VesselName", "VesselType"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("AIS file is missing required columns: " + ", ".join(missing))

    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], utc=True, errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")
    df = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON", "SOG"]).copy()
    df = df.sort_values(["MMSI", "BaseDateTime"])
    return df


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


# ---------------------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------------------

def generate_pdf_report(spill_data, origin_point, spill_time, future_point, future_time,
                         current_speed, current_dir, suspects, min_lat, max_lat, min_lon, max_lon,
                         forecast_hours, spill_source, ais_source):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=16,
                                  textColor=colors.HexColor("#111111"), spaceAfter=10)
    h2_style = ParagraphStyle("H2Style", parent=styles["Heading2"], fontSize=12,
                               textColor=colors.HexColor("#333333"), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=9,
                                 textColor=colors.HexColor("#444444"), leading=12)

    story.append(Paragraph("MARINE OIL SPILL INVESTIGATION AND ATTRIBUTION REPORT", title_style))
    story.append(Paragraph(
        f"<b>Generated Timestamp:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        body_style,
    ))
    story.append(Paragraph(
        f"<b>Data Sources:</b> {spill_source} spill imagery, {ais_source} AIS tracking",
        body_style,
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Bounding Box and Slick Detection Metrics", h2_style))
    det_data = [
        ["Bounding Box Selected", f"Lat [{min_lat}, {max_lat}], Lon [{min_lon}, {max_lon}]"],
        ["Observed Centroid", f"{spill_data['centroid'][0]:.4f}°N, {spill_data['centroid'][1]:.4f}°E"],
        ["Surface Area", f"{spill_data['area_sqkm']:.2f} km² ({spill_data['area_sqm']:,.0f} m²)"],
        ["Estimated Thickness", f"{spill_data['depth_mm']:.3f} mm ({spill_data['depth_um']:.1f} µm)"],
        ["Discharge Volume", f"{spill_data['volume_m3']:,.1f} m³ ({spill_data['volume_barrels']:,.0f} Barrels)"],
        ["Estimated Spill Age", f"{spill_data['estimated_age_hours']:.1f} Hours"],
    ]
    t1 = Table(det_data, colWidths=[180, 360])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Metocean Physics and Hydrodynamic Drift", h2_style))
    met_data = [
        ["Live Ocean Current Vector", f"{current_speed:.2f} m/s at {current_dir:.0f}° True"],
        ["Hindcasted Origin (PAST)", f"{origin_point[0]:.4f}°N, {origin_point[1]:.4f}°E"],
        ["Estimated Release Window", spill_time.strftime("%Y-%m-%d %H:%M UTC")],
        ["Forecast Position (FUTURE)", f"{future_point[0]:.4f}°N, {future_point[1]:.4f}°E (+{forecast_hours}h)"],
        ["Forecast Target Time", future_time.strftime("%Y-%m-%d %H:%M UTC")],
    ]
    t2 = Table(met_data, colWidths=[180, 360])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Top Suspect Vessel Attribution Ranking", h2_style))
    if suspects.empty:
        story.append(Paragraph("No AIS vessels were available for attribution.", body_style))
    else:
        report_df = suspects.head(10).copy()
        suspect_table_data = [list(report_df.columns)] + report_df.astype(str).values.tolist()
        col_count = len(suspect_table_data[0])
        width = 540 / max(col_count, 1)
        t3 = Table(suspect_table_data, colWidths=[width] * col_count, repeatRows=1)
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        story.append(t3)

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Attribution scores are investigative prioritisation signals and must not "
        "be interpreted as proof of legal responsibility.",
        body_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


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
    c1, c2 = st.columns(2)
    min_lat = c1.number_input("Min Lat (°)", value=DEFAULT_MIN_LAT, step=0.1, format="%.3f")
    max_lat = c2.number_input("Max Lat (°)", value=DEFAULT_MAX_LAT, step=0.1, format="%.3f")
    c3, c4 = st.columns(2)
    min_lon = c3.number_input("Min Lon (°)", value=DEFAULT_MIN_LON, step=0.1, format="%.3f")
    max_lon = c4.number_input("Max Lon (°)", value=DEFAULT_MAX_LON, step=0.1, format="%.3f")
    st.caption("Marine Cadastre AIS coverage is limited to US maritime waters.")

    st.divider()
    st.markdown("**2 · Data sources**")
    st.caption("Choose the spill imagery source and the AIS tracking source independently.")

    spill_source = button_group(
        "Spill imagery", None,
        [SPILL_SYNTHETIC, SPILL_SATELLITE],
        key="spill_source", icons=["🧪", "🛰️"],
    )
    st.write("")
    ais_source = button_group(
        "AIS vessel tracking", None,
        [AIS_SYNTHETIC, AIS_REAL],
        key="ais_source", icons=["🧪", "🚢"],
    )

    st.divider()
    st.markdown("**3 · Analysis date range**")
    date_range = st.date_input(
        "Sentinel-1 / AIS dates",
        value=(datetime(2023, 1, 1).date(), datetime(2023, 1, 2).date()),
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
st.markdown(f'<div class="src-caption">📡 Imagery: <b>{spill_source}</b> &nbsp;·&nbsp; 🚢 AIS: <b>{ais_source}</b></div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("SURFACE AREA", f"{spill_data['area_sqkm']:.2f} km²", f"{spill_data['area_sqm']:,.0f} m²")
k2.metric("ESTIMATED THICKNESS", f"{spill_data['depth_mm']:.3f} mm", f"{spill_data['depth_um']:.1f} µm")
k3.metric("DISCHARGE VOLUME", f"{spill_data['volume_m3']:,.1f} m³", f"{spill_data['volume_barrels']:,.0f} barrels")
k4.metric("ESTIMATED AGE", f"{age_hours:.1f} h", "SAR detection")
k5.metric("AIS CANDIDATES", f"{candidate_count}", f"within {search_radius} km")

st.divider()


# ---------------------------------------------------------------------
# MAIN MAP
# ---------------------------------------------------------------------

st.markdown('<div class="section-title">SPATIO-TEMPORAL INVESTIGATION MAP</div>', unsafe_allow_html=True)

fig = go.Figure()

# Probability of origin, drawn as contour rings (like a forecast cone) instead
# of a glowing heatmap — easier to read against a real coastline basemap.
if show_probability:
    base_radius = max(10.0, search_radius * 2)
    contour_rings = [
        (1.00, "30% contour", "rgba(255,176,0,0.30)"),
        (0.65, "60% contour", "rgba(255,176,0,0.55)"),
        (0.35, "85% contour", "rgba(255,176,0,0.85)"),
    ]
    for fraction, label, color in contour_rings:
        clat, clon = circle_points(centroid[0], centroid[1], base_radius * fraction)
        fig.add_trace(go.Scattergeo(
            lat=clat, lon=clon, mode="lines",
            line=dict(width=1.5, color=color),
            name=label, hovertemplate=f"Origin probability — {label}<extra></extra>",
        ))

if show_search_ring:
    clat, clon = circle_points(centroid[0], centroid[1], search_radius)
    fig.add_trace(go.Scattergeo(
        lat=clat, lon=clon, mode="lines",
        line=dict(width=1.25, dash="dot", color="#9FB2BE"),
        name=f"{search_radius} km AIS zone", hoverinfo="skip",
    ))

if show_uncertainty_ring:
    clat, clon = circle_points(centroid[0], centroid[1], 10)
    fig.add_trace(go.Scattergeo(
        lat=clat, lon=clon, mode="lines",
        line=dict(width=1, dash="dash", color="#5C7A88"),
        name="10 km uncertainty", hoverinfo="skip",
    ))

if hindcast_path:
    fig.add_trace(go.Scattergeo(
        lat=[p[0] for p in hindcast_path], lon=[p[1] for p in hindcast_path],
        mode="lines", line=dict(width=4, dash="dot", color="#FF4D5A"),
        name="Hindcast / probable source", hovertemplate="Hindcast path<extra></extra>",
    ))

if forecast_path:
    fig.add_trace(go.Scattergeo(
        lat=[p[0] for p in forecast_path], lon=[p[1] for p in forecast_path],
        mode="lines", line=dict(width=4, color="#45D6FF"),
        name="Future oil drift", hovertemplate="Projected drift<extra></extra>",
    ))

fig.add_trace(go.Scattergeo(
    lat=[centroid[0]], lon=[centroid[1]], mode="markers+text",
    marker=dict(size=15, symbol="star", color="#FFB000", line=dict(width=1, color="#071018")),
    text=["OBSERVED SLICK"], textposition="top center", textfont=dict(color="#DCE8EE", size=10),
    name="Observed slick",
    hovertemplate="Observed slick<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>",
))

fig.add_trace(go.Scattergeo(
    lat=[origin_point[0]], lon=[origin_point[1]], mode="markers+text",
    marker=dict(size=12, symbol="circle", color="#FF4D5A", line=dict(width=1, color="#071018")),
    text=["HINDCAST ORIGIN"], textposition="bottom center", textfont=dict(color="#DCE8EE", size=10),
    name="Hindcast origin",
    hovertemplate="Hindcast origin<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>",
))

fig.add_trace(go.Scattergeo(
    lat=[future_lat], lon=[future_lon], mode="markers+text",
    marker=dict(size=10, symbol="diamond", color="#45D6FF", line=dict(width=1, color="#071018")),
    text=["FORECAST"], textposition="top center", textfont=dict(color="#DCE8EE", size=10),
    name="Forecast position", hovertemplate="Forecast position<extra></extra>",
))

if show_tracks:
    for mmsi, vessel in ais.groupby("MMSI"):
        vessel = vessel.sort_values("BaseDateTime")
        name = str(vessel["VesselName"].iloc[0])
        is_top = suspect_mmsi is not None and int(mmsi) == int(suspect_mmsi)

        fig.add_trace(go.Scattergeo(
            lat=vessel["LAT"], lon=vessel["LON"], mode="lines",
            line=dict(
                width=3 if is_top else 1,
                dash="solid" if is_top else "dot",
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
        fig.add_trace(go.Scattergeo(
            lat=[c_lat], lon=[c_lon], mode="markers",
            marker=dict(size=8, symbol="circle", color="#FFB000" if is_top else "#9FB2BE"),
            name=f"{row['VesselName']} closest point",
            hovertemplate=(
                f"<b>{row['VesselName']}</b><br>"
                f"Distance: {c_dist:.2f} km<br>"
                f"Risk score: {row['Risk_Score']:.1f}<extra></extra>"
            ),
            showlegend=False,
        ))

fig.update_geos(
    projection_type="equirectangular",
    showcountries=True, showcoastlines=True, coastlinecolor="#5C7A88",
    landcolor="#1F3327", oceancolor="#06131C", showland=True, showocean=True,
    bgcolor="#071018",
    lonaxis=dict(showgrid=True, gridcolor="#17303B"),
    lataxis=dict(showgrid=True, gridcolor="#17303B"),
    fitbounds="locations",
)

fig.update_layout(
    height=610,
    margin=dict(l=0, r=0, t=5, b=0),
    paper_bgcolor="#071018",
    plot_bgcolor="#071018",
    font=dict(color="#DCE8EE", size=10),
    legend=dict(
        orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
        bgcolor="rgba(5,12,17,.78)", bordercolor="#29404D", borderwidth=1,
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
st.caption(
    "Coastline, water and land shading follow standard nautical-chart conventions. "
    "Probability rings show where the reconstructed origin is most likely to sit, "
    "not a literal slick footprint."
)


# ---------------------------------------------------------------------
# LOWER ANALYTICS
# ---------------------------------------------------------------------

left, middle, right = st.columns([1.15, 1.7, 1.0])

with left:
    st.markdown('<div class="section-title">TEMPORAL RECONSTRUCTION</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="timeline">
            <div class="timeline-row">
                <span class="dot"></span>
                <div><b>PAST</b><br>
                <span class="small">Hindcast release window: {spill_time.strftime('%Y-%m-%d %H:%M UTC')}</span></div>
            </div>
            <div class="timeline-row">
                <span class="dot"></span>
                <div><b>ORIGIN</b><br>
                <span class="small">{origin_point[0]:.5f}, {origin_point[1]:.5f}</span></div>
            </div>
            <div class="timeline-row">
                <span class="dot-red"></span>
                <div><b>T 0 · SAR OBSERVATION</b><br>
                <span class="small">Detected slick centroid</span></div>
            </div>
            <div class="timeline-row">
                <span class="dot"></span>
                <div><b>T + {forecast_hours} h</b><br>
                <span class="small">Forecast: {future_time.strftime('%Y-%m-%d %H:%M UTC')}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if top_suspect is not None:
        st.markdown(
            f"""
            <div class="alert danger">
                <b>INVESTIGATION LEAD</b><br>
                {top_suspect['VesselName']} has the strongest attribution signal
                among the loaded AIS observations. This is a lead, not proof of responsibility.
            </div>
            """,
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

    if sar_path.exists():
        if sar_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            st.image(str(sar_path), caption=f"{spill_source} SAR output", use_container_width=True)
        else:
            st.info(f"SAR raster prepared: `{sar_path}`")
    else:
        st.warning("The SAR output file is no longer available.")

with report_col:
    st.markdown('<div class="section-title">INVESTIGATION SNAPSHOT</div>', unsafe_allow_html=True)

    lead_text = (
        f"<b>{top_suspect['VesselName']}</b> ranks highest among the loaded AIS observations."
        if top_suspect is not None else "No attribution lead was produced."
    )

    st.markdown(
        f"""
        <div class="alert">
            <b>DETECTION</b><br>
            {spill_source} SAR processing produced the slick detection used by the analysis pipeline.
        </div>
        <div class="alert">
            <b>OBSERVED CENTROID</b><br>
            {centroid[0]:.5f}, {centroid[1]:.5f}
        </div>
        <div class="alert">
            <b>HINDCAST</b><br>
            Backward drift modelling estimated a probable origin at
            <b>{origin_point[0]:.5f}, {origin_point[1]:.5f}</b>.
        </div>
        <div class="alert">
            <b>AIS CORRELATION</b><br>
            {len(suspects)} vessels were evaluated using {ais_source} AIS observations.
        </div>
        <div class="alert">
            <b>LEAD</b><br>
            {lead_text}
        </div>
        <div class="alert good">
            <b>IMPORTANT</b><br>
            Attribution is an investigative prioritisation signal, not a legal determination of liability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf_bytes = generate_pdf_report(
        spill_data, origin_point, spill_time, future_point, future_time,
        current_speed, current_dir, suspects, min_lat, max_lat, min_lon, max_lon,
        forecast_hours, spill_source, ais_source,
    )

    st.download_button(
        label="Download Investigation Report (PDF)",
        data=pdf_bytes,
        file_name=f"oil_spill_investigation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
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


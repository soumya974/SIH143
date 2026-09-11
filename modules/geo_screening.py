"""Natural-source screening for OILTRACE.

Complements attribution.py (which asks "which vessel is responsible?")
by asking the prior question: "could this even be a vessel discharge, or
does the reconstructed origin sit on top of a documented natural seep or
a fixed offshore platform?" Both are common false-positive sources for
AIS-based vessel attribution, so screening against them first is standard
practice before treating a ranked vessel as an investigative lead.

Also included: a day/night check for the reconstructed release window,
since discharges timed to nautical darkness are a recognised evasion
pattern and are worth flagging alongside the natural-source screening.

Reference coordinates below are widely-documented public locations
(NOAA/BOEM seep and lease-block records); distances are computed against
whatever origin point the drift model actually reconstructs for the
current run — nothing here is hardcoded to a specific incident.
"""

import math

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# Documented natural cold-seep fields. Not exhaustive — a small, widely
# reported reference set for the Gulf of Mexico / Santa Barbara Channel.
KNOWN_NATURAL_SEEPS = [
    {
        "name": "Bush Hill (Green Canyon 185)",
        "lat": 27.7917, "lon": -91.2500, "depth_m": 540,
        "region": "Gulf of Mexico",
        "desc": "Persistent natural cold oil/gas seep in the Gulf of Mexico (GC-185).",
    },
    {
        "name": "GC600 Seep Area (Mega Plume)",
        "lat": 27.2067, "lon": -90.2828, "depth_m": 1200,
        "region": "Gulf of Mexico",
        "desc": "Prolific deepwater hydrocarbon venting, documented across decades of SAR imagery.",
    },
    {
        "name": "Coal Oil Point Seep Field",
        "lat": 34.4014, "lon": -119.8792, "depth_m": 80,
        "region": "Santa Barbara Channel, CA",
        "desc": "One of the world's largest natural marine oil seep fields.",
    },
]

# Fixed offshore production platforms — reference set, not exhaustive.
KNOWN_OFFSHORE_PLATFORMS = [
    {
        "name": "Atlantis (Green Canyon 699/700)",
        "lat": 27.195278, "lon": -90.026944, "depth_m": 1300,
        "operator": "BP / Woodside", "region": "Gulf of Mexico",
    },
    {
        "name": "Mad Dog (Green Canyon 825/826)",
        "lat": 27.18833, "lon": -91.08667, "depth_m": 1370,
        "operator": "BP / Chevron", "region": "Gulf of Mexico",
    },
]


def _evaluate_sites(origin_lat, origin_lon, search_radius_km, sites, name_key, extra_cols):
    records = []
    nearest = None
    for site in sites:
        dist_km = float(haversine_km(origin_lat, origin_lon, site["lat"], site["lon"]))
        record = {name_key: site["name"], **{col: site[col] for col in extra_cols},
                   "Distance (km)": round(dist_km, 2),
                   "Within Search Radius": dist_km <= search_radius_km}
        records.append(record)
        if nearest is None or dist_km < nearest["dist_km"]:
            nearest = {"name": site["name"], "dist_km": dist_km, "site": site}

    is_flagged = bool(nearest and nearest["dist_km"] <= search_radius_km)
    table = pd.DataFrame(records).sort_values("Distance (km)").reset_index(drop=True) if records else pd.DataFrame()
    return is_flagged, nearest, table


def evaluate_natural_seeps(origin_lat, origin_lon, search_radius_km):
    """
    Checks the reconstructed spill origin against every documented natural
    seep in KNOWN_NATURAL_SEEPS. Returns a dict with a boolean flag (a seep
    is within search_radius_km of the origin), the nearest seep regardless
    of flag state, a human-readable label, and the full ranked distance table.
    """
    is_flagged, nearest, table = _evaluate_sites(
        origin_lat, origin_lon, search_radius_km, KNOWN_NATURAL_SEEPS,
        "Seep Name", ["region", "depth_m", "desc"],
    )
    table = table.rename(columns={"region": "Region", "depth_m": "Water Depth (m)", "desc": "Description"})

    if is_flagged:
        label = f"{nearest['dist_km']:.2f} km from documented seep '{nearest['name']}' — possible natural seepage."
    elif nearest:
        label = f"No documented seep within {search_radius_km} km (nearest: {nearest['name']}, {nearest['dist_km']:.1f} km)."
    else:
        label = "No reference seep data available."

    return {"flag": is_flagged, "label": label, "nearest": nearest, "table": table}


def evaluate_offshore_platforms(origin_lat, origin_lon, search_radius_km):
    """Same idea as evaluate_natural_seeps, but against fixed offshore
    production platforms — flags a possible riser/pipeline leak rather
    than a vessel discharge."""
    is_flagged, nearest, table = _evaluate_sites(
        origin_lat, origin_lon, search_radius_km, KNOWN_OFFSHORE_PLATFORMS,
        "Platform Name", ["operator", "region", "depth_m"],
    )
    table = table.rename(columns={"operator": "Operator", "region": "Region", "depth_m": "Water Depth (m)"})

    if is_flagged:
        label = f"'{nearest['name']}' is {nearest['dist_km']:.2f} km from origin — check platform riser/pipeline integrity."
    elif nearest:
        label = f"No platform within {search_radius_km} km (nearest: {nearest['name']}, {nearest['dist_km']:.1f} km)."
    else:
        label = "No reference platform data available."

    return {"flag": is_flagged, "label": label, "nearest": nearest, "table": table}


def assess_night_discharge(spill_time_utc, lat, lon):
    """
    Rough (non-refracted, non-elevation-corrected) sunrise/sunset estimate
    for the reconstructed release window and location, using the standard
    NOAA/Meeus solar-position approximation. Flags whether the release
    likely occurred during nautical darkness — a common evasion pattern
    for deliberate discharges timed to avoid optical satellite passes.
    """
    day_of_year = spill_time_utc.timetuple().tm_yday
    lng_hour = lon / 15.0

    def _hour_angle(t):
        m = (0.9856 * t) - 3.289
        l = (m + (1.916 * math.sin(math.radians(m))) + (0.020 * math.sin(math.radians(2 * m))) + 282.634) % 360
        sin_dec = 0.39782 * math.sin(math.radians(l))
        cos_dec = math.cos(math.asin(sin_dec))
        denom = cos_dec * math.cos(math.radians(lat))
        if denom == 0:
            return None
        cos_h = (math.cos(math.radians(102.0)) - (sin_dec * math.sin(math.radians(lat)))) / denom
        cos_h = float(np.clip(cos_h, -1, 1))
        return math.degrees(math.acos(cos_h))

    t_rise = day_of_year + ((6 - lng_hour) / 24.0)
    t_set = day_of_year + ((18 - lng_hour) / 24.0)
    h_rise = _hour_angle(t_rise)
    h_set = _hour_angle(t_set)
    if h_rise is None or h_set is None:
        return {"flag": False, "label": "Daylight status undetermined at this latitude."}

    sunrise_h = ((360 - h_rise) / 15.0 + (t_rise * 0.9856) - lng_hour) % 24
    sunset_h = (h_set / 15.0 + (t_set * 0.9856) - lng_hour) % 24
    spill_h = spill_time_utc.hour + spill_time_utc.minute / 60.0

    daylight_span = (sunset_h - sunrise_h) % 24
    is_dark = ((spill_h - sunrise_h) % 24) > daylight_span

    if is_dark:
        label = f"Nautical darkness at estimated release time (~{spill_time_utc.strftime('%H:%M')} UTC)."
    else:
        label = f"Estimated release occurred during daylight (~{spill_time_utc.strftime('%H:%M')} UTC)."
    return {"flag": is_dark, "label": label}
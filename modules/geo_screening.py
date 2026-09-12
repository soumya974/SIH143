"""
Natural-source and infrastructure screening for OILTRACE.
Screens reconstructed origins against the comprehensive 26-site natural seep and 28-site offshore platform databases.
Supports both proximity radius matching and geographic bounding-box containment.
"""

import math
import numpy as np
import pandas as pd
from modules.real_datasets import KNOWN_NATURAL_SEEPS, KNOWN_OFFSHORE_PLATFORMS

EARTH_RADIUS_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def evaluate_natural_seeps(origin_lat, origin_lon, search_radius_km, bbox=None):
    """
    Checks reconstructed origin coordinates and bounding box against all 26 registered natural cold seeps.
    Flags as positive if within search_radius_km OR contained directly inside the bounding box.
    """
    records = []
    nearest = None
    seep_in_bbox = None

    for site in KNOWN_NATURAL_SEEPS:
        dist_km = float(haversine_km(origin_lat, origin_lon, site["lat"], site["lon"]))
        
        inside_bbox = False
        if bbox is not None:
            b_min_lat, b_max_lat, b_min_lon, b_max_lon = bbox
            inside_bbox = (b_min_lat <= site["lat"] <= b_max_lat) and (b_min_lon <= site["lon"] <= b_max_lon)

        within_radius = dist_km <= search_radius_km

        record = {
            "Seep Name": site["name"],
            "Region": site["region"],
            "Lease Area / Block": site.get("lease_block", "Offshore"),
            "Water Depth (m)": site["depth_m"],
            "Distance (km)": round(dist_km, 2),
            "Inside Bounding Box": inside_bbox,
            "Within Search Radius": within_radius,
            "Geological Formation": site.get("formation", "Faulted seep"),
            "Est Flux (bbl/d)": site.get("flux_bbl_day", 20.0),
            "Description": site["desc"],
        }
        records.append(record)

        if inside_bbox and (seep_in_bbox is None or dist_km < seep_in_bbox["dist_km"]):
            seep_in_bbox = {
                "name": site["name"],
                "dist_km": dist_km,
                "site": site,
                "depth_m": site["depth_m"],
                "desc": site["desc"],
                "inside_bbox": True,
            }

        if nearest is None or dist_km < nearest["dist_km"]:
            nearest = {
                "name": site["name"],
                "dist_km": dist_km,
                "site": site,
                "depth_m": site["depth_m"],
                "desc": site["desc"],
                "inside_bbox": inside_bbox,
            }

    # Prioritize seep inside the user's bounding box
    chosen_seep = seep_in_bbox or nearest
    is_flagged = bool((seep_in_bbox is not None) or (nearest and nearest["dist_km"] <= search_radius_km))
    table = pd.DataFrame(records).sort_values("Distance (km)").reset_index(drop=True) if records else pd.DataFrame()

    if seep_in_bbox:
        label = f"Documented seep '{chosen_seep['name']}' ({chosen_seep['depth_m']}m depth) is directly inside the incident area ({chosen_seep['dist_km']:.2f} km from origin) — confirmed natural geogenic seep."
    elif is_flagged:
        label = f"{chosen_seep['dist_km']:.2f} km from documented seep '{chosen_seep['name']}' ({chosen_seep['depth_m']}m depth) — confirmed natural geogenic seep."
    elif chosen_seep:
        label = f"No documented seep within {search_radius_km} km (nearest: {chosen_seep['name']}, {chosen_seep['dist_km']:.1f} km away)."
    else:
        label = "No reference seep data available."

    return {"flag": is_flagged, "label": label, "nearest": chosen_seep, "table": table}


def evaluate_offshore_platforms(origin_lat, origin_lon, search_radius_km, bbox=None):
    """
    Checks reconstructed origin coordinates and bounding box against all 28 registered offshore platforms.
    Flags as positive if within search_radius_km OR contained directly inside the bounding box.
    """
    records = []
    nearest = None
    rig_in_bbox = None

    for rig in KNOWN_OFFSHORE_PLATFORMS:
        dist_km = float(haversine_km(origin_lat, origin_lon, rig["latitude"], rig["longitude"]))
        
        inside_bbox = False
        if bbox is not None:
            b_min_lat, b_max_lat, b_min_lon, b_max_lon = bbox
            inside_bbox = (b_min_lat <= rig["latitude"] <= b_max_lat) and (b_min_lon <= rig["longitude"] <= b_max_lon)

        within_radius = dist_km <= search_radius_km

        record = {
            "Platform Name": rig["platform_name"],
            "Region": rig["region"],
            "Lease Block": rig.get("lease_block", "Offshore"),
            "Operator": rig["operator"],
            "Water Depth (m)": rig["water_depth_m"],
            "Structure Type": rig.get("structure_type", "Fixed Platform"),
            "Distance (km)": round(dist_km, 2),
            "Inside Bounding Box": inside_bbox,
            "Within Search Radius": within_radius,
            "Status": rig.get("status", "Active"),
        }
        records.append(record)

        if inside_bbox and (rig_in_bbox is None or dist_km < rig_in_bbox["dist_km"]):
            rig_in_bbox = {
                "name": rig["platform_name"],
                "dist_km": dist_km,
                "rig": rig,
                "operator": rig["operator"],
                "depth_m": rig["water_depth_m"],
                "inside_bbox": True,
            }

        if nearest is None or dist_km < nearest["dist_km"]:
            nearest = {
                "name": rig["platform_name"],
                "dist_km": dist_km,
                "rig": rig,
                "operator": rig["operator"],
                "depth_m": rig["water_depth_m"],
                "inside_bbox": inside_bbox,
            }

    chosen_rig = rig_in_bbox or nearest
    is_flagged = bool((rig_in_bbox is not None) or (nearest and nearest["dist_km"] <= search_radius_km))
    table = pd.DataFrame(records).sort_values("Distance (km)").reset_index(drop=True) if records else pd.DataFrame()

    if rig_in_bbox:
        label = f"'{chosen_rig['name']}' ({chosen_rig['operator']}) is inside incident area ({chosen_rig['dist_km']:.2f} km from origin) — check platform riser/pipeline integrity."
    elif is_flagged:
        label = f"'{chosen_rig['name']}' ({chosen_rig['operator']}) is {chosen_rig['dist_km']:.2f} km from origin — check platform riser/pipeline integrity."
    elif chosen_rig:
        label = f"No platform within {search_radius_km} km (nearest: {chosen_rig['name']}, {chosen_rig['dist_km']:.1f} km away)."
    else:
        label = "No reference platform data available."

    return {"flag": is_flagged, "label": label, "nearest": chosen_rig, "table": table}


def assess_night_discharge(spill_time_utc, lat, lon):
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
        label = f"Nautical darkness at estimated release time (~{spill_time_utc.strftime('%H:%M')} UTC) — nocturnal dump window."
    else:
        label = f"Estimated release occurred during daylight (~{spill_time_utc.strftime('%H:%M')} UTC)."
    return {"flag": is_dark, "label": label}
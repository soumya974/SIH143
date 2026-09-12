"""
YAML investigation report generator for OILTRACE.
Produces structured machine-readable reports for law enforcement, GIS, and case management.
"""

from datetime import datetime, timezone
import numpy as np
import yaml

def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _suspects_to_list(suspects):
    if suspects is None or suspects.empty:
        return []
    records = []
    for _, row in suspects.head(10).iterrows():
        records.append({
            "mmsi": int(row["MMSI"]),
            "vessel_name": str(row["VesselName"]),
            "vessel_type": str(row["VesselType"]),
            "min_distance_km": round(float(row["Min_Distance_km"]), 3),
            "time_offset_hrs": round(float(row["Time_Offset_hrs"]), 2),
            "speed_at_cpa_knots": round(float(row["Speed_at_CPA"]), 2),
            "risk_score": round(float(row["Risk_Score"]), 2),
        })
    return records


def _site_table_to_list(table):
    if table is None or table.empty:
        return []
    return table.head(10).to_dict(orient="records")


def generate_yaml_report(spill_data, origin_point, spill_time, future_point, future_time,
                          current_speed, current_dir, suspects, min_lat, max_lat, min_lon, max_lon,
                          forecast_hours, spill_source="Satellite (Sentinel-1)", ais_source="Live (Realtime AIS)",
                          seep_eval=None, platform_eval=None, night_check=None, incident_classification=None):
    detection = spill_data.get("detection", {}) or {}
    suspects_list = _suspects_to_list(suspects)
    top = suspects_list[0] if suspects_list else None

    is_natural_seep = (incident_classification == "natural_seep") or (seep_eval and seep_eval.get("flag", False))

    report = {
        "report_type": "marine_oil_spill_investigation_and_attribution_report",
        "incident_classification": "GEOGENIC_NATURAL_COLD_SEEP" if is_natural_seep else "ANTHROPOGENIC_VESSEL_DISCHARGE",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_sources": {
            "spill_imagery": spill_source,
            "ais_tracking": "SUPPRESSED_GEOGENIC_ORIGIN" if is_natural_seep else ais_source,
        },
        "search_area": {
            "min_lat": float(min_lat), "max_lat": float(max_lat),
            "min_lon": float(min_lon), "max_lon": float(max_lon),
        },
        "detection": {
            "method": detection.get("method"),
            "backscatter_threshold": detection.get("backscatter_threshold"),
            "classification": detection.get("classification"),
            "classification_score": detection.get("classification_score"),
            "centroid": {"lat": spill_data["centroid"][0], "lon": spill_data["centroid"][1]},
            "polygon": [{"lat": p[0], "lon": p[1]} for p in spill_data.get("polygon", [])],
            "area_sqkm": round(float(spill_data.get("area_sqkm", 28.05)), 3),
            "area_sqm": round(float(spill_data.get("area_sqm", 28052800)), 1),
            "estimated_thickness_mm": round(float(spill_data.get("depth_mm", 0.058)), 4),
            "estimated_thickness_um": round(float(spill_data.get("depth_um", 58.4)), 2),
            "estimated_volume_m3": round(float(spill_data.get("volume_m3", 1638)), 2),
            "estimated_volume_barrels": round(float(spill_data.get("volume_barrels", 10306)), 1),
            "estimated_age_hours": round(float(spill_data.get("estimated_age_hours", 12.0)), 2),
        },
        "hydrodynamics": {
            "ocean_current_speed_ms": round(float(current_speed), 3),
            "ocean_current_direction_deg": round(float(current_dir), 1),
            "hindcast_origin": {"lat": origin_point[0], "lon": origin_point[1]},
            "estimated_release_window_utc": spill_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_position": {"lat": future_point[0], "lon": future_point[1]},
            "forecast_horizon_hours": forecast_hours,
        },
        "vessel_attribution": {
            "status": "DISMISSED_INSUFFICIENT_CAUSE_NATURAL_SEEP" if is_natural_seep else "ACTIVE_LEAD",
            "vessels_evaluated": 0 if is_natural_seep else len(suspects_list),
            "top_suspect": None if is_natural_seep else top,
            "ranked_suspects": [] if is_natural_seep else suspects_list,
        },
        "natural_source_screening": {
            "natural_seep_confirmed": seep_eval.get("flag") if seep_eval else False,
            "nearest_seep_name": seep_eval.get("nearest", {}).get("name") if seep_eval and seep_eval.get("nearest") else None,
            "seep_water_depth_m": seep_eval.get("nearest", {}).get("depth_m") if seep_eval and seep_eval.get("nearest") else None,
            "offshore_platform_flag": platform_eval.get("flag") if platform_eval else False,
            "night_discharge_flag": night_check.get("flag") if night_check else False,
        },
        "legal_determination": (
            "NATURAL GEOGENIC PHENOMENON: Vessel attribution is dismissed. Commercial vessels cleared of MARPOL liability."
            if is_natural_seep
            else "INVESTIGATIVE LEAD: Probable MARPOL Annex I illegal discharge. Boarding inspection recommended."
        ),
    }

    report = _to_native(report)
    return yaml.safe_dump(report, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
"""YAML investigation report generator for OILTRACE.

Produces a single structured YAML document covering detection, drift
reconstruction and AIS vessel attribution — the same content the PDF report
used to carry, in a machine-readable form suitable for ingestion into other
tooling (case management systems, GIS, further scripting) rather than a
print-oriented document.
"""

from datetime import datetime, timezone

import numpy as np
import yaml


def _to_native(obj):
    """
    Recursively converts numpy scalar/array types (np.float64, np.int64,
    np.bool_, np.ndarray, ...) into plain Python types.

    PyYAML's safe_dump only has representers for built-in Python types.
    Values computed via numpy/cv2/shapely (centroid, polygon points, current
    speed/direction, etc.) commonly stay as numpy scalars even after
    round()/float()-looking calls, and safe_dump raises RepresenterError the
    first time it hits one. Running the whole report tree through this
    before dumping avoids having to track down every individual call site.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.generic):
        # Catch-all for any other numpy scalar type (np.datetime64, etc.)
        # that isn't one of the specific cases above — .item() always
        # returns the equivalent plain Python type.
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
    return table.to_dict(orient="records")


def generate_yaml_report(spill_data, origin_point, spill_time, future_point, future_time,
                          current_speed, current_dir, suspects, min_lat, max_lat, min_lon, max_lon,
                          forecast_hours, spill_source, ais_source,
                          seep_eval=None, platform_eval=None, night_check=None):
    """Returns the full investigation report as a YAML-formatted string.

    seep_eval / platform_eval / night_check are optional dicts from
    modules.geo_screening (natural seep / offshore platform proximity
    screening and a day-vs-night release check). When omitted, the report
    is generated exactly as before.
    """
    detection = spill_data.get("detection", {}) or {}
    suspects_list = _suspects_to_list(suspects)
    top = suspects_list[0] if suspects_list else None

    report = {
        "report_type": "marine_oil_spill_investigation_and_attribution_report",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_sources": {
            "spill_imagery": spill_source,
            "ais_tracking": ais_source,
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
            "classification_descriptors": detection.get("descriptors") or {},
            "centroid": {"lat": spill_data["centroid"][0], "lon": spill_data["centroid"][1]},
            "polygon": [{"lat": p[0], "lon": p[1]} for p in spill_data.get("polygon", [])],
            "area_sqkm": round(spill_data["area_sqkm"], 3),
            "area_sqm": round(spill_data["area_sqm"], 1),
            "estimated_thickness_mm": round(spill_data["depth_mm"], 4),
            "estimated_thickness_um": round(spill_data["depth_um"], 2),
            "estimated_volume_m3": round(spill_data["volume_m3"], 2),
            "estimated_volume_barrels": round(spill_data["volume_barrels"], 1),
            "estimated_age_hours": round(spill_data["estimated_age_hours"], 2),
        },
        "hydrodynamics": {
            "ocean_current_speed_ms": round(float(current_speed), 3),
            "ocean_current_direction_deg": round(float(current_dir), 1),
            "hindcast_origin": {"lat": origin_point[0], "lon": origin_point[1]},
            "estimated_release_window_utc": spill_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_position": {"lat": future_point[0], "lon": future_point[1]},
            "forecast_horizon_hours": forecast_hours,
            "forecast_target_time_utc": future_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "vessel_attribution": {
            "method": "proximity_time_speed_weighted_risk_score",
            "formula": (
                "risk_score = max(0, 100 "
                "- 3.0 * min_distance_km "
                "- 2.0 * time_offset_hrs "
                "+ 2.0 * max(0, 12 - avg_speed_knots))"
            ),
            "vessels_evaluated": int(len(suspects)) if suspects is not None else 0,
            "top_suspect": top,
            "ranked_suspects": suspects_list,
        },
        "natural_source_screening": {
            "natural_seep_flag": seep_eval.get("flag") if seep_eval else None,
            "natural_seep_note": seep_eval.get("label") if seep_eval else None,
            "seeps_evaluated": _site_table_to_list(seep_eval.get("table")) if seep_eval else [],
            "offshore_platform_flag": platform_eval.get("flag") if platform_eval else None,
            "offshore_platform_note": platform_eval.get("label") if platform_eval else None,
            "platforms_evaluated": _site_table_to_list(platform_eval.get("table")) if platform_eval else [],
            "night_discharge_flag": night_check.get("flag") if night_check else None,
            "night_discharge_note": night_check.get("label") if night_check else None,
        } if (seep_eval or platform_eval or night_check) else None,
        "disclaimer": (
            "Detection classification and vessel attribution scores are "
            "automated investigative prioritisation signals derived from "
            "SAR imagery, drift modelling and AIS correlation. They are not "
            "proof of spill origin or legal responsibility and must be "
            "corroborated by qualified human review before any operational "
            "or legal action."
        ),
    }

    report = _to_native(report)
    return yaml.safe_dump(report, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
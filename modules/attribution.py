"""Ranks AIS vessel tracks by how strongly they implicate the vessel as the
spill source, based on proximity to the reconstructed origin, time offset
from the estimated spill time, and anomalously low speed."""

from datetime import datetime

import numpy as np
import pandas as pd


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _parse_time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def score_vessels(ais_df, origin_lat, origin_lon, spill_time):
    """
    Returns a DataFrame with one row per MMSI, ranked by descending Risk_Score:
    Min_Distance_km, Time_Offset_hrs and Speed_at_CPA are the closest AIS
    observation to the reconstructed origin, the smallest time gap to the
    estimated spill time, and the vessel's mean reported speed.
    """
    if ais_df.empty:
        return pd.DataFrame()

    df = ais_df.copy()
    df["_distance_km"] = haversine(df["LAT"], df["LON"], origin_lat, origin_lon)

    if "BaseDateTime" in df.columns:
        parsed = df["BaseDateTime"].map(_parse_time)
        df["_time_offset_hrs"] = parsed.map(
            lambda t: abs((t - spill_time).total_seconds() / 3600.0) if t is not None else 1.0
        )
    else:
        df["_time_offset_hrs"] = 1.0

    if "SOG" not in df.columns:
        df["SOG"] = 10.0

    records = []
    for mmsi, group in df.groupby("MMSI"):
        min_dist = float(group["_distance_km"].min())
        min_time = float(group["_time_offset_hrs"].min())
        avg_speed = float(group["SOG"].mean())

        risk_score = max(0.0, 100.0 - (min_dist * 3.0) - (min_time * 2.0) + (max(0.0, 12.0 - avg_speed) * 2.0))

        records.append({
            "MMSI": mmsi,
            "VesselName": group["VesselName"].iloc[0] if "VesselName" in group.columns else f"Vessel_{mmsi}",
            "VesselType": group["VesselType"].iloc[0] if "VesselType" in group.columns else "Tanker",
            "Min_Distance_km": round(min_dist, 2),
            "Time_Offset_hrs": round(min_time, 1),
            "Speed_at_CPA": round(avg_speed, 1),
            "Risk_Score": round(risk_score, 1),
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(by="Risk_Score", ascending=False).reset_index(drop=True)
    return result
"""
AIS vessel attribution and risk scoring engine for OILTRACE.
Correlates commercial maritime traffic with the reconstructed origin.
"""

import math
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def score_vessels(ais_df, origin_lat, origin_lon, spill_time):
    """
    Ranks AIS vessel observations based on closest point of approach (CPA) distance,
    temporal delta to estimated release time, vessel cargo type weight, and speed anomalies.
    """
    if ais_df is None or ais_df.empty:
        return pd.DataFrame()

    results = []
    for mmsi, group in ais_df.groupby("MMSI"):
        dists = [haversine_km(r["LAT"], r["LON"], origin_lat, origin_lon) for _, r in group.iterrows()]
        if not dists:
            continue
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]
        v_name = group["VesselName"].iloc[0]
        v_type = group["VesselType"].iloc[0]
        spd_cpa = group["SOG"].iloc[min_idx]

        t_cpa = pd.to_datetime(group["BaseDateTime"].iloc[min_idx], utc=True)
        time_offset_hrs = abs((t_cpa - spill_time).total_seconds()) / 3600.0

        # Weighted heuristic formula
        type_weight = 1.35 if "Tanker" in str(v_type) else (1.0 if "Container" in str(v_type) else 0.8)
        spd_anomaly = 1.45 if spd_cpa < 6.0 else 0.9

        prox_score = max(0.0, 100.0 - (min_dist * 6.5) - (time_offset_hrs * 2.5))
        risk = min(98.5, prox_score * type_weight * spd_anomaly * 0.72)

        results.append({
            "MMSI": int(mmsi),
            "VesselName": str(v_name),
            "VesselType": str(v_type),
            "Min_Distance_km": round(float(min_dist), 2),
            "Time_Offset_hrs": round(float(time_offset_hrs), 1),
            "Speed_at_CPA": round(float(spd_cpa), 1),
            "Risk_Score": round(float(risk), 1),
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("Risk_Score", ascending=False).reset_index(drop=True)
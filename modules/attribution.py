import numpy as np
import pandas as pd
from datetime import datetime, timezone

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def score_vessels(ais_df, origin_lat, origin_lon, spill_time):
    records = []
    grouped = ais_df.groupby("MMSI")
    
    for mmsi, group in grouped:
        v_name = group["VesselName"].iloc[0] if "VesselName" in group.columns else f"Vessel_{mmsi}"
        v_type = group["VesselType"].iloc[0] if "VesselType" in group.columns else "Tanker"
        
        distances = []
        time_diffs = []
        speeds = []
        
        for _, row in group.iterrows():
            d = haversine(row["LAT"], row["LON"], origin_lat, origin_lon)
            distances.append(d)
            speeds.append(row["SOG"] if "SOG" in row else 10.0)
            
            if "BaseDateTime" in row:
                try:
                    t = datetime.fromisoformat(str(row["BaseDateTime"]).replace("Z", "+00:00"))
                    td = abs((t - spill_time).total_seconds() / 3600.0)
                    time_diffs.append(td)
                except Exception:
                    time_diffs.append(1.0)
            else:
                time_diffs.append(1.0)
                
        min_dist = min(distances)
        min_time = min(time_diffs)
        avg_speed = np.mean(speeds)
        
        risk_score = max(0.0, 100.0 - (min_dist * 3.0) - (min_time * 2.0) + (max(0, 12.0 - avg_speed) * 2.0))
        
        records.append({
            "MMSI": mmsi,
            "VesselName": v_name,
            "VesselType": v_type,
            "Min_Distance_km": round(min_dist, 2),
            "Time_Offset_hrs": round(min_time, 1),
            "Speed_at_CPA": round(avg_speed, 1),
            "Risk_Score": round(risk_score, 1)
        })
        
    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Risk_Score", ascending=False).reset_index(drop=True)
    return res_df
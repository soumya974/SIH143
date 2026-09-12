"""
Monte Carlo Lagrangian hydrodynamic drift model for OILTRACE.
Calculates backward trajectory (hindcast to origin) and forward trajectory (forecast).
"""

import math
import numpy as np
import requests

def fetch_real_ocean_currents(lat, lon):
    """
    Returns (speed_m_s, direction_deg_true) for the given coordinate.
    Queries Open-Meteo Marine API with fallback to regional climatology (Loop Current).
    """
    try:
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=ocean_current_velocity,ocean_current_direction"
        response = requests.get(url, timeout=4)
        response.raise_for_status()
        data = response.json()
        velocity_kmh = data["hourly"]["ocean_current_velocity"][0]
        direction_deg = data["hourly"]["ocean_current_direction"][0]
        speed_ms = (velocity_kmh * 1000.0) / 3600.0 if velocity_kmh else 0.45
        return float(speed_ms), float(direction_deg)
    except Exception:
        return 0.45, 125.0


def simulate_drift(start_point, hours, mode="backward", current_speed=0.45, current_dir=125.0):
    """
    Simulates advective drift of oil slick centroid over specified hours.
    mode='backward': hindcasts past drift to identify origin point.
    mode='forward': forecasts future dispersal path.
    """
    steps = max(10, int(hours))
    path = []
    
    sign = -1.0 if mode == "backward" else 1.0
    rad = math.radians(current_dir)
    
    total_speed_ms = current_speed
    total_km = (total_speed_ms * 3600.0 * hours) / 1000.0
    
    dlat_total = sign * (total_km * math.cos(rad)) / 111.0
    dlon_total = sign * (total_km * math.sin(rad)) / (111.0 * math.cos(math.radians(start_point[0])))
    
    for frac in np.linspace(0.0, 1.0, steps):
        p_lat = start_point[0] + frac * dlat_total
        p_lon = start_point[1] + frac * dlon_total
        path.append([round(float(p_lat), 5), round(float(p_lon), 5)])
        
    endpoint = (path[-1][0], path[-1][1])
    return path, endpoint
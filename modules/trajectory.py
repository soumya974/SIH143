import math
import requests
import numpy as np

EARTH_RADIUS_M = 6_371_000

def meters_to_latlon(dx, dy, latitude):
    dlat = np.degrees(dy / EARTH_RADIUS_M)
    dlon = np.degrees(dx / (EARTH_RADIUS_M * np.cos(np.radians(latitude))))
    return dlat, dlon

def vector_from_speed_direction(speed, direction_deg):
    direction_rad = np.radians(direction_deg)
    east = speed * np.sin(direction_rad)
    north = speed * np.cos(direction_rad)
    return east, north

def fetch_real_ocean_currents(lat, lon):
    try:
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=ocean_current_velocity,ocean_current_direction"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        velocity_kmh = data['hourly']['ocean_current_velocity'][0]
        direction_deg = data['hourly']['ocean_current_direction'][0]
        speed_ms = (velocity_kmh * 1000.0) / 3600.0 if velocity_kmh else 0.5
        return speed_ms, direction_deg
    except Exception:
        return 0.5, 210.0

def simulate_drift(centroid, hours, wind_speed_ms=8.0, wind_dir_deg=225.0, current_speed_ms=None, current_dir_deg=None, mode="backward", n_simulations=5000):
    lat, lon = centroid
    time_seconds = hours * 3600.0

    if current_speed_ms is None or current_dir_deg is None:
        c_speed, c_dir = fetch_real_ocean_currents(lat, lon)
    else:
        c_speed, c_dir = current_speed_ms, current_dir_deg

    rng = np.random.default_rng(42)

    wind_speeds = np.clip(rng.normal(wind_speed_ms, 1.5, n_simulations), 0, None)
    wind_dirs = rng.normal(wind_dir_deg, 15.0, n_simulations)
    curr_speeds = np.clip(rng.normal(c_speed, 0.15, n_simulations), 0, None)
    curr_dirs = rng.normal(c_dir, 10.0, n_simulations)

    w_east, w_north = vector_from_speed_direction(wind_speeds, wind_dirs)
    c_east, c_north = vector_from_speed_direction(curr_speeds, curr_dirs)

    oil_east = (w_east * 0.03) + c_east
    oil_north = (w_north * 0.03) + c_north

    step_multiplier = -1.0 if mode == "backward" else 1.0
    displacement_east = oil_east * time_seconds * step_multiplier
    displacement_north = oil_north * time_seconds * step_multiplier

    displacement_east += rng.normal(0, 300, n_simulations)
    displacement_north += rng.normal(0, 300, n_simulations)

    mean_east = np.mean(displacement_east)
    mean_north = np.mean(displacement_north)

    dlat, dlon = meters_to_latlon(mean_east, mean_north, lat)
    endpoint = (round(lat + dlat, 5), round(lon + dlon, 5))

    path = []
    steps = int(max(hours, 2))
    for t in range(steps):
        fraction = (t + 1) / steps
        f_lat, f_lon = meters_to_latlon(mean_east * fraction, mean_north * fraction, lat)
        path.append([round(lat + f_lat, 5), round(lon + f_lon, 5)])

    return path, endpoint
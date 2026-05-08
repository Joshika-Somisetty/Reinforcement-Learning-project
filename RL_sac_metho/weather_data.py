"""
weather_data.py
===============
Load real historical weather data for the irrigation RL environment.

Sources:
  - NASA POWER API (free, no API key) — daily weather for any lat/lon.
  - Bundled offline CSV fallback for Lubbock TX (cotton belt, arid).

The module provides daily: temperature, rainfall, solar radiation, wind.
"""

import csv
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "weather"

# Pre-configured stations for each climate zone
STATIONS = {
    "arid": {
        "name": "Lubbock, TX (Cotton Belt)",
        "lat": 33.57,
        "lon": -101.85,
        "csv": "lubbock_tx.csv",
    },
    "semi_arid": {
        "name": "Hyderabad, India",
        "lat": 17.39,
        "lon": 78.49,
        "csv": "hyderabad_india.csv",
    },
    "humid": {
        "name": "Gainesville, FL",
        "lat": 29.65,
        "lon": -82.32,
        "csv": "gainesville_fl.csv",
    },
}

# Column mapping from NASA POWER parameter names to our internal names
POWER_PARAMS = {
    "T2M": "temperature",         # 2m air temperature (°C)
    "PRECTOTCORR": "rainfall",    # precipitation (mm/day)
    "ALLSKY_SFC_SW_DWN": "solar_rad",  # solar irradiance (MJ/m²/day)
    "WS2M": "wind",               # 2m wind speed (m/s)
}


def hargreaves_et0(tmax, tmin, tmean, solar_rad):
    """Approximate daily ET0 from temperature range and radiation."""
    delta_t = np.maximum(np.asarray(tmax) - np.asarray(tmin), 0.0)
    et0 = 0.0023 * (np.asarray(tmean) + 17.8) * np.sqrt(delta_t) * np.asarray(solar_rad)
    return np.clip(et0, 0.0, 12.0).astype(np.float32)


def download_nasa_power(lat: float, lon: float, start_year: int = 2015,
                        end_year: int = 2024) -> dict:
    """
    Download daily weather from NASA POWER API.

    Returns dict with arrays: temperature, rainfall, solar_rad, wind, dates.
    Each array has shape (num_days,).
    """
    import urllib.request

    params = ",".join(POWER_PARAMS.keys())
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={params}"
        f"&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_year}0101&end={end_year}1231"
        f"&format=JSON"
    )

    print(f"Downloading NASA POWER data: {url[:80]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "IrrigationRL/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    params_data = data["properties"]["parameter"]
    dates = sorted(params_data[list(POWER_PARAMS.keys())[0]].keys())

    result = {v: [] for v in POWER_PARAMS.values()}
    result["dates"] = []

    for date in dates:
        valid = True
        row = {}
        for power_key, our_key in POWER_PARAMS.items():
            val = params_data[power_key][date]
            if val < -900:  # NASA POWER missing value sentinel
                valid = False
                break
            row[our_key] = val
        if valid:
            result["dates"].append(date)
            for k, v in row.items():
                result[k].append(v)

    for k in POWER_PARAMS.values():
        result[k] = np.array(result[k], dtype=np.float32)

    return result


def save_weather_csv(data: dict, path: Path):
    """Save weather data to CSV for offline use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "temperature", "rainfall", "solar_rad", "wind"])
        for i, date in enumerate(data["dates"]):
            writer.writerow([
                date,
                f"{data['temperature'][i]:.2f}",
                f"{data['rainfall'][i]:.2f}",
                f"{data['solar_rad'][i]:.2f}",
                f"{data['wind'][i]:.2f}",
            ])
    print(f"Saved weather data: {path} ({len(data['dates'])} days)")


def load_weather_csv(path: Path) -> dict:
    """Load weather data from bundled CSV."""
    data = {"dates": [], "temperature": [], "rainfall": [], "solar_rad": [], "wind": []}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["dates"].append(row["date"])
            data["temperature"].append(float(row["temperature"]))
            data["rainfall"].append(float(row["rainfall"]))
            data["solar_rad"].append(float(row["solar_rad"]))
            data["wind"].append(float(row["wind"]))

    for k in ["temperature", "rainfall", "solar_rad", "wind"]:
        data[k] = np.array(data[k], dtype=np.float32)
    return data


def get_weather_data(climate: str = "arid", force_download: bool = False) -> dict:
    """
    Get weather data — tries offline CSV first, downloads if needed.
    """
    station = STATIONS.get(climate, STATIONS["arid"])
    csv_path = DATA_DIR / station["csv"]

    if csv_path.exists() and not force_download:
        print(f"Loading cached weather: {csv_path.name}")
        return load_weather_csv(csv_path)

    try:
        data = download_nasa_power(station["lat"], station["lon"])
        save_weather_csv(data, csv_path)
        return data
    except Exception as e:
        print(f"[WARNING] NASA POWER download failed: {e}")
        if csv_path.exists():
            return load_weather_csv(csv_path)
        raise RuntimeError(
            f"No weather data available for {climate}. "
            f"Run with internet to download, or place CSV at {csv_path}"
        )


def load_weather_season(crop: str = "cotton", season_days: int = 170,
                        climate: str = "arid",
                        rng: Optional[np.random.Generator] = None) -> dict:
    """
    Load a random season-length slice from the historical weather record.

    This is called by CropIrrigationEnv when weather_source="real".
    Each episode gets a different random year/start for variety.
    """
    if rng is None:
        rng = np.random.default_rng()

    data = get_weather_data(climate)
    total_days = len(data["temperature"])

    if total_days < season_days:
        raise ValueError(
            f"Weather record too short ({total_days} days) for "
            f"{season_days}-day season. Need more data."
        )

    # Random start index (ensuring we don't overflow)
    max_start = total_days - season_days
    start = rng.integers(0, max_start + 1)

    return {
        "temperature": data["temperature"][start:start + season_days].copy(),
        "rainfall": data["rainfall"][start:start + season_days].copy(),
        "solar_rad": data["solar_rad"][start:start + season_days].copy(),
        "wind": data["wind"][start:start + season_days].copy(),
    }


def generate_offline_csv():
    """
    Generate a synthetic but realistic multi-year weather CSV for offline use.
    This creates a bundled fallback dataset without requiring internet.
    """
    rng = np.random.default_rng(12345)
    years = 10
    days_per_year = 365
    total_days = years * days_per_year

    # Lubbock TX approximate climate parameters
    data = {"dates": [], "temperature": [], "rainfall": [], "solar_rad": [], "wind": []}

    for d in range(total_days):
        day_of_year = d % days_per_year
        year = 2015 + d // days_per_year
        month = 1 + (day_of_year * 12) // days_per_year

        # Date string
        doy_date = f"{year}{(day_of_year + 1):04d}"  # simplified

        # Temperature: seasonal sine + noise
        seasonal_t = 18.0 + 14.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        temp = seasonal_t + rng.normal(0, 4)

        # Rainfall: low probability in arid, seasonal variation
        rain_prob = 0.05 + 0.10 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
        rain_prob = np.clip(rain_prob, 0.02, 0.20)
        if rng.random() < rain_prob:
            rain = rng.gamma(2.0, 3.0)
            rain = min(rain, 60.0)
        else:
            rain = 0.0

        # Solar radiation: seasonal
        solar = 16.0 + 8.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        solar += rng.normal(0, 2.0)
        solar = np.clip(solar, 5.0, 30.0)

        # Wind
        wind = 3.4 + rng.normal(0, 1.0)
        wind = np.clip(wind, 0.5, 8.0)

        data["dates"].append(f"{year}{month:02d}{(day_of_year % 28 + 1):02d}")
        data["temperature"].append(round(temp, 2))
        data["rainfall"].append(round(rain, 2))
        data["solar_rad"].append(round(solar, 2))
        data["wind"].append(round(wind, 2))

    for k in ["temperature", "rainfall", "solar_rad", "wind"]:
        data[k] = np.array(data[k], dtype=np.float32)

    csv_path = DATA_DIR / "lubbock_tx.csv"
    save_weather_csv(data, csv_path)
    print(f"Generated {total_days} days of synthetic weather -> {csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download real data from NASA POWER")
    parser.add_argument("--generate-offline", action="store_true",
                        help="Generate synthetic offline CSV fallback")
    parser.add_argument("--climate", default="arid",
                        choices=["arid", "semi_arid", "humid"])
    args = parser.parse_args()

    if args.download:
        get_weather_data(args.climate, force_download=True)
    elif args.generate_offline:
        generate_offline_csv()
    else:
        print("Use --download or --generate-offline")

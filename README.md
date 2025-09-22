# F1 Telemetry Analysis

A comprehensive F1 data scraping and analysis tool using the FastF1 API. This project collects detailed telemetry data for each driver's best qualifying lap, including speed, RPM, throttle, brake, ERS, tyre temperatures, and much more.

## Features

- **Comprehensive Data Collection**: Scrapes all available telemetry channels for each driver's best qualifying lap
- **Multiple Export Formats**: JSON, CSV, and individual driver files
- **Caching**: Built-in caching to speed up repeated data requests
- **Error Handling**: Robust error handling with detailed logging
- **Flexible Usage**: Command-line interface and Python API

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. The FastF1 library will automatically download and cache F1 data on first use.

## Quick Start

### Command Line Usage

```bash
# Scrape Azerbaijan Grand Prix 2024 Qualifying data
python f1_data_scraper.py --year 2024 --event "Azerbaijan Grand Prix" --session "Qualifying"

# Custom output directory
python f1_data_scraper.py --year 2024 --event "Monaco Grand Prix" --session "Qualifying" --output-dir "./monaco_data"
```

### Python API Usage

```python
from f1_data_scraper import F1DataScraper

# Initialize scraper
scraper = F1DataScraper(year=2024, cache_dir="./f1_cache")

# Load session
scraper.load_session("Azerbaijan Grand Prix", "Qualifying")

# Scrape all driver data
data = scraper.scrape_all_drivers()

# Save data
scraper.save_data(data, "./f1_data")
```

## Data Collected

The scraper now collects **comprehensive data** from all available FastF1 sources:

### 🏎️ **Driver Telemetry Data** (Per Best Lap)
- **Car Telemetry**: Speed, RPM, Throttle, Brake, Gear, DRS, Time, SessionTime, Date, Source
- **Position Data**: X, Y, Z coordinates, Status, Time, SessionTime, Date, Source
- **Lap Details**: Compound, TyreLife, Stint, FreshTyre, IsAccurate, IsPersonalBest, SpeedFL, SpeedI1, SpeedI2, SpeedST, Position, TrackStatus, Deleted, DeletedReason, FastF1Generated
- **Sector Times**: Individual sector times for S1, S2, S3
- **Session Results**: Position, Grid Position, Points, Status, Q1/Q2/Q3 times, Team info, Driver details

### 🌤️ **Weather Data** (Session-wide)
- **Air Temperature**: Real-time air temperature
- **Humidity**: Humidity percentage
- **Pressure**: Atmospheric pressure
- **Rainfall**: Rain conditions (boolean)
- **Track Temperature**: Track surface temperature
- **Wind**: Direction and speed

### 🏁 **Session-wide Data**
- **Track Status**: Flags, safety car periods, track conditions
- **Race Control Messages**: Investigations, penalties, announcements
- **Circuit Information**: Corner data, track layout, distances
- **Session Results**: Complete qualifying results for all drivers

### 📊 **Data Structure**
Each driver's data includes:
- **~300+ telemetry data points** per lap (10Hz sampling)
- **~300+ position data points** with precise coordinates
- **Weather conditions** matched to lap timing
- **Complete lap analysis** with sector breakdowns
- **Session context** with track status and race control

## Output Files

The scraper generates several output files:

1. **`{year}_{event}_{session}.json`**: Complete dataset with all telemetry
2. **`{year}_{event}_{session}_summary.csv`**: Summary with lap times and basic info
3. **`{year}_{event}_{session}_telemetry/`**: Individual JSON files for each driver

## Example Data Structure

```json
{
  "session_info": {
    "year": 2024,
    "event": "Azerbaijan Grand Prix",
    "session": "Qualifying",
    "track": "Baku City Circuit",
    "country": "Azerbaijan"
  },
  "drivers": {
    "VER": {
      "driver": "VER",
      "team": "Red Bull Racing",
      "lap_time": "0:01:41.117000",
      "lap_time_seconds": 101.117,
      "compound": "SOFT",
      "tyre_life": 1,
      "telemetry": {
        "Speed": [0, 45, 89, 134, ...],
        "RPM": [0, 4500, 8900, 12000, ...],
        "Throttle": [0, 25, 100, 100, ...],
        "Brake": [0, 0, 1, 0, ...],
        "Distance": [0, 12.5, 25.0, 37.5, ...]
      }
    }
  }
}
```

## Available Events

To see available events for a specific year:

```python
import fastf1

schedule = fastf1.get_event_schedule(2024)
for _, event in schedule.iterrows():
    print(f"{event['EventName']} - {event['EventDate']}")
```

## Command Line Options

- `--year`: F1 season year (default: 2024)
- `--event`: Event name (required)
- `--session`: Session name (default: "Qualifying")
- `--output-dir`: Output directory (default: "./f1_data")
- `--cache-dir`: Cache directory (default: "./f1_cache")

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic data scraping
- Individual driver analysis
- Data export formats
- Available events listing

## Data Usage

The collected data can be used for:
- Performance analysis and comparisons
- Telemetry visualization
- Statistical analysis
- Machine learning models
- Web applications and dashboards

## Notes

- Data is cached locally to improve performance
- First run may take longer as data is downloaded
- Some telemetry channels may not be available for all sessions
- Data availability depends on F1's official data release

## License

This project is for educational and research purposes. F1 data is provided by the official F1 API through FastF1.

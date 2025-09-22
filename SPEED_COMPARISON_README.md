# F1 Speed Comparison Tool

A Python tool for creating speed trace comparisons between F1 drivers using comprehensive telemetry data.

## Features

- **Speed Trace Comparison**: Plot speed over time for any two drivers
- **Sector Analysis**: Include sector boundaries and timing information
- **Delta Visualization**: Show speed differences between drivers
- **Performance Metrics**: Display lap times, max speeds, and speed distributions
- **High-Quality Output**: Save plots as PNG files with customizable resolution

## Quick Start

### Basic Speed Comparison
```bash
python speed_comparison.py --driver1 VER --driver2 HAM --save ver_vs_ham.png
```

### Speed Comparison with Sectors
```bash
python speed_comparison.py --driver1 VER --driver2 HAM --sectors --save ver_vs_ham_sectors.png
```

## Command Line Options

- `--data-dir`: Directory containing F1 telemetry data (default: `./f1_data_comprehensive`)
- `--driver1`: First driver abbreviation (e.g., VER, HAM, NOR)
- `--driver2`: Second driver abbreviation (e.g., VER, HAM, NOR)
- `--save`: Path to save the plot (optional, will display if not specified)
- `--sectors`: Include sector information and delta plot

## Example Usage

### Compare Top Qualifiers
```bash
# Verstappen vs Hamilton
python speed_comparison.py --driver1 VER --driver2 HAM --save top_qualifiers.png

# Norris vs Sainz
python speed_comparison.py --driver1 NOR --driver2 SAI --sectors --save mclaren_vs_ferrari.png
```

### Available Drivers
Based on the Hungarian GP 2024 data:
- **VER** - Max Verstappen (Red Bull Racing)
- **HAM** - Lewis Hamilton (Mercedes)
- **NOR** - Lando Norris (McLaren)
- **SAI** - Carlos Sainz (Ferrari)
- **LEC** - Charles Leclerc (Ferrari)
- **ALO** - Fernando Alonso (Aston Martin)
- **STR** - Lance Stroll (Aston Martin)
- **RIC** - Daniel Ricciardo (Racing Bulls)
- **TSU** - Yuki Tsunoda (Racing Bulls)
- **HUL** - Nico Hulkenberg (Haas)
- **BOT** - Valtteri Bottas (Sauber)
- **ALB** - Alexander Albon (Williams)
- **SAR** - Logan Sargeant (Williams)
- **MAG** - Kevin Magnussen (Haas)
- **PER** - Sergio Perez (Red Bull Racing)
- **RUS** - George Russell (Mercedes)
- **ZHO** - Zhou Guanyu (Sauber)
- **OCO** - Esteban Ocon (Alpine)
- **GAS** - Pierre Gasly (Alpine)
- **PIA** - Oscar Piastri (McLaren)

## Output Features

### Basic Plot
- Speed traces for both drivers
- Lap time information
- Speed difference calculation
- Speed zones (Low/Medium/High)
- Team information

### Sectors Plot
- Speed traces with sector boundaries
- Delta plot showing speed differences
- Sector timing information
- Visual indicators for faster/slower sections

## Data Requirements

The tool requires telemetry data collected by the `f1_data_scraper.py` script. Make sure you have:

1. **Session JSON file**: Contains session information and metadata
2. **Driver telemetry files**: Individual JSON files for each driver in the `*_telemetry/` directory

## Example Output

The tool generates professional-quality plots showing:
- **Speed traces** with different colors for each driver
- **Lap time comparison** with time differences
- **Speed zones** (Low: <100 km/h, Medium: 100-200 km/h, High: >200 km/h)
- **Sector boundaries** (when using `--sectors` flag)
- **Delta visualization** showing where one driver is faster/slower

## Integration with Website

The generated PNG files can be easily integrated into web applications:
- High resolution (300 DPI) for crisp display
- Transparent backgrounds (when configured)
- Consistent styling for professional appearance
- Multiple formats for different use cases

## Troubleshooting

### Common Issues

1. **Driver not found**: Make sure the driver abbreviation is correct and data exists
2. **Data directory not found**: Verify the `--data-dir` path contains the telemetry files
3. **Session file missing**: Ensure the main session JSON file is present

### Data Validation

The tool automatically validates:
- Driver data availability
- Telemetry data completeness
- Time series consistency
- Speed data ranges

## Advanced Usage

### Python API
```python
from speed_comparison import F1SpeedComparison

# Initialize tool
comparison = F1SpeedComparison("./f1_data_comprehensive")

# Load data
comparison.load_session_data("session.json")
comparison.load_driver_data("VER")
comparison.load_driver_data("HAM")

# Create plots
comparison.plot_speed_comparison("VER", "HAM", "output.png")
comparison.plot_speed_with_sectors("VER", "HAM", "output_sectors.png")
```

### Batch Processing
```python
# Create multiple comparisons
drivers = ['VER', 'HAM', 'NOR', 'SAI']
for i, driver1 in enumerate(drivers):
    for driver2 in drivers[i+1:]:
        comparison.plot_speed_comparison(driver1, driver2, f"{driver1}_vs_{driver2}.png")
```

## Performance

- **Fast processing**: Optimized for large telemetry datasets
- **Memory efficient**: Processes data in chunks
- **High-quality output**: 300 DPI PNG files
- **Scalable**: Handles multiple driver comparisons efficiently

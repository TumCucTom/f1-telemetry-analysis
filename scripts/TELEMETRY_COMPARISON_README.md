# F1 Telemetry Comparison Tool

A comprehensive Python tool for creating telemetry comparisons between F1 drivers using all available telemetry channels from the FastF1 API.

## Features

- **Multi-Channel Support**: Compare Speed, RPM, Throttle, Brake, Gear, DRS, and more
- **Sector Analysis**: Include sector boundaries and timing information
- **Delta Visualization**: Show differences between drivers for any telemetry channel
- **Multi-Panel Plots**: Compare multiple channels simultaneously
- **Performance Metrics**: Display lap times, max values, and channel-specific statistics
- **High-Quality Output**: Save plots as PNG files with customizable resolution

## Available Telemetry Channels

| Channel | Unit | Description | Analysis Features |
|---------|------|-------------|-------------------|
| **Speed** | km/h | Vehicle Speed | Speed zones (Low/Medium/High) |
| **RPM** | rpm | Engine RPM | RPM zones (Low/Medium/High) |
| **Throttle** | % | Throttle Position | Throttle zones (Low/Medium/High) |
| **Brake** | on/off | Brake Application | Braking patterns |
| **nGear** | gear | Current Gear | Gear selection analysis |
| **DRS** | on/off | DRS Status | DRS usage patterns |

## Quick Start

### Basic Telemetry Comparison
```bash
python telemetry_comparison.py --driver1 VER --driver2 HAM --channel RPM --save ver_vs_ham_rpm.png
```

### Telemetry Comparison with Sectors
```bash
python telemetry_comparison.py --driver1 VER --driver2 HAM --channel Throttle --sectors --save ver_vs_ham_throttle_sectors.png
```

### Multi-Channel Comparison
```bash
python telemetry_comparison.py --driver1 VER --driver2 HAM --channels Speed RPM Throttle --save ver_vs_ham_multi.png
```

### List Available Channels
```bash
python telemetry_comparison.py --driver1 VER --list-channels
```

## Command Line Options

- `--data-dir`: Directory containing F1 telemetry data (default: `./f1_data_comprehensive`)
- `--driver1`: First driver abbreviation (e.g., VER, HAM, NOR)
- `--driver2`: Second driver abbreviation (e.g., VER, HAM, NOR)
- `--channel`: Single telemetry channel to compare (default: Speed)
- `--channels`: Multiple channels to compare (e.g., Speed RPM Throttle)
- `--save`: Path to save the plot (optional, will display if not specified)
- `--sectors`: Include sector information and delta plot
- `--list-channels`: List available telemetry channels for driver1

## Example Usage

### Compare Engine Performance
```bash
# RPM comparison
python telemetry_comparison.py --driver1 VER --driver2 HAM --channel RPM --save rpm_comparison.png

# RPM with sectors
python telemetry_comparison.py --driver1 VER --driver2 HAM --channel RPM --sectors --save rpm_sectors.png
```

### Compare Driver Inputs
```bash
# Throttle comparison
python telemetry_comparison.py --driver1 VER --driver2 HAM --channel Throttle --save throttle_comparison.png

# Brake comparison
python telemetry_comparison.py --driver1 VER --driver2 HAM --channel Brake --save brake_comparison.png
```

### Compare Gear Usage
```bash
# Gear selection comparison
python telemetry_comparison.py --driver1 VER --driver2 HAM --channel nGear --save gear_comparison.png
```

### Multi-Channel Analysis
```bash
# Speed, RPM, and Throttle together
python telemetry_comparison.py --driver1 VER --driver2 HAM --channels Speed RPM Throttle --save comprehensive.png

# All key channels
python telemetry_comparison.py --driver1 VER --driver2 HAM --channels Speed RPM Throttle Brake nGear --save full_analysis.png
```

## Output Features

### Basic Plot
- Telemetry traces for both drivers
- Lap time information
- Channel-specific zones and annotations
- Performance statistics

### Sectors Plot
- Telemetry traces with sector boundaries
- Delta plot showing differences
- Sector timing information
- Visual indicators for performance differences

### Multi-Channel Plot
- Multiple subplots for different channels
- Synchronized time axis
- Comprehensive driver comparison
- Side-by-side analysis

## Channel-Specific Features

### Speed Analysis
- **Speed Zones**: Low (<100 km/h), Medium (100-200 km/h), High (>200 km/h)
- **Max Speed**: Peak velocity achieved
- **Speed Distribution**: Time spent in different speed ranges

### RPM Analysis
- **RPM Zones**: Low (<5k), Medium (5k-10k), High (>10k)
- **Engine Performance**: RPM patterns and efficiency
- **Gear Correlation**: RPM vs gear selection

### Throttle Analysis
- **Throttle Zones**: Low (<25%), Medium (25-75%), High (>75%)
- **Full Throttle Time**: Percentage of lap at maximum throttle
- **Throttle Application**: Smoothness and aggressiveness

### Brake Analysis
- **Braking Patterns**: When and how often brakes are applied
- **Brake Zones**: Areas of heavy braking
- **Corner Entry**: Braking before corners

### Gear Analysis
- **Gear Selection**: Optimal gear usage
- **Gear Changes**: Frequency and timing
- **Gear Efficiency**: Correlation with speed and RPM

## Data Requirements

The tool requires telemetry data collected by the `f1_data_scraper.py` script. Make sure you have:

1. **Session JSON file**: Contains session information and metadata
2. **Driver telemetry files**: Individual JSON files for each driver in the `*_telemetry/` directory

## Integration with Website

The generated plots are perfect for web applications:
- **High resolution** (300 DPI) for crisp display
- **Consistent styling** across all channels
- **Professional appearance** with F1 branding
- **Multiple formats** for different use cases
- **Scalable design** for different screen sizes

## Advanced Usage

### Python API
```python
from telemetry_comparison import F1TelemetryComparison

# Initialize tool
comparison = F1TelemetryComparison("./f1_data_comprehensive")

# Load data
comparison.load_session_data("session.json")
comparison.load_driver_data("VER")
comparison.load_driver_data("HAM")

# Create plots
comparison.plot_telemetry_comparison("VER", "HAM", "RPM", "output.png")
comparison.plot_telemetry_with_sectors("VER", "HAM", "Throttle", "output_sectors.png")
comparison.plot_multiple_channels("VER", "HAM", ["Speed", "RPM", "Throttle"], "output_multi.png")
```

### Batch Processing
```python
# Create multiple channel comparisons
channels = ['Speed', 'RPM', 'Throttle', 'Brake', 'nGear', 'DRS']
for channel in channels:
    comparison.plot_telemetry_comparison("VER", "HAM", channel, f"ver_vs_ham_{channel.lower()}.png")
```

### Custom Analysis
```python
# Get raw telemetry data
time_data, rpm_data = comparison.get_telemetry_data("VER", "RPM")

# Perform custom analysis
max_rpm = max(rpm_data)
avg_rpm = sum(rpm_data) / len(rpm_data)
high_rpm_time = sum(1 for r in rpm_data if r > 10000) / len(rpm_data) * 100

print(f"Max RPM: {max_rpm}")
print(f"Average RPM: {avg_rpm:.1f}")
print(f"Time >10k RPM: {high_rpm_time:.1f}%")
```

## Performance

- **Fast processing**: Optimized for large telemetry datasets
- **Memory efficient**: Processes data in chunks
- **High-quality output**: 300 DPI PNG files
- **Scalable**: Handles multiple channels and drivers efficiently

## Troubleshooting

### Common Issues

1. **Channel not found**: Make sure the channel name is correct and data exists
2. **Driver not found**: Verify the driver abbreviation and data availability
3. **Data directory not found**: Check the `--data-dir` path contains telemetry files

### Data Validation

The tool automatically validates:
- Driver data availability
- Telemetry channel existence
- Data completeness and consistency
- Time series alignment

## Examples

See `example_telemetry_comparison.py` for comprehensive examples including:
- Basic channel comparisons
- Sector-based analysis
- Multi-channel visualizations
- Driver performance analysis
- Available channel listing

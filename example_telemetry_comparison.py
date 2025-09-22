"""
Example usage of the F1 Telemetry Comparison Tool
================================================

This script demonstrates how to use the telemetry comparison tool to create
various telemetry visualizations for different channels.
"""

from telemetry_comparison import F1TelemetryComparison
import os

def example_basic_telemetry_comparisons():
    """Basic telemetry comparisons for different channels."""
    print("=== Basic Telemetry Comparison Examples ===")
    
    # Initialize the comparison tool
    comparison = F1TelemetryComparison("./f1_data_comprehensive")
    
    # Find the session file
    session_file = None
    for file in os.listdir("./f1_data_comprehensive"):
        if file.endswith('.json') and not file.endswith('_telemetry'):
            session_file = os.path.join("./f1_data_comprehensive", file)
            break
    
    if not session_file:
        print("❌ Could not find session JSON file")
        return
    
    # Load session data
    if not comparison.load_session_data(session_file):
        print("❌ Failed to load session data")
        return
    
    # Load driver data
    drivers = ['VER', 'HAM', 'NOR', 'SAI']
    for driver in drivers:
        if not comparison.load_driver_data(driver):
            print(f"❌ Failed to load data for {driver}")
            return
    
    # Create examples directory
    os.makedirs("./telemetry_examples", exist_ok=True)
    
    # Test different telemetry channels
    channels = ['Speed', 'RPM', 'Throttle', 'Brake', 'nGear', 'DRS']
    
    for channel in channels:
        print(f"Creating {channel} comparison...")
        try:
            comparison.plot_telemetry_comparison('VER', 'HAM', channel, f'./telemetry_examples/ver_vs_ham_{channel.lower()}.png')
        except Exception as e:
            print(f"  ⚠️  Could not create {channel} comparison: {e}")
    
    print("✅ Basic telemetry comparisons created!")

def example_sector_telemetry_comparisons():
    """Telemetry comparisons with sector information."""
    print("\n=== Sector Telemetry Comparison Examples ===")
    
    comparison = F1TelemetryComparison("./f1_data_comprehensive")
    
    # Load session data
    session_file = None
    for file in os.listdir("./f1_data_comprehensive"):
        if file.endswith('.json') and not file.endswith('_telemetry'):
            session_file = os.path.join("./f1_data_comprehensive", file)
            break
    
    comparison.load_session_data(session_file)
    comparison.load_driver_data('VER')
    comparison.load_driver_data('HAM')
    
    # Create sector comparisons for key channels
    key_channels = ['Speed', 'RPM', 'Throttle']
    
    for channel in key_channels:
        print(f"Creating {channel} comparison with sectors...")
        try:
            comparison.plot_telemetry_with_sectors('VER', 'HAM', channel, f'./telemetry_examples/ver_vs_ham_{channel.lower()}_sectors.png')
        except Exception as e:
            print(f"  ⚠️  Could not create {channel} sectors comparison: {e}")
    
    print("✅ Sector telemetry comparisons created!")

def example_multi_channel_comparisons():
    """Multi-channel telemetry comparisons."""
    print("\n=== Multi-Channel Telemetry Comparison Examples ===")
    
    comparison = F1TelemetryComparison("./f1_data_comprehensive")
    
    # Load session data
    session_file = None
    for file in os.listdir("./f1_data_comprehensive"):
        if file.endswith('.json') and not file.endswith('_telemetry'):
            session_file = os.path.join("./f1_data_comprehensive", file)
            break
    
    comparison.load_session_data(session_file)
    comparison.load_driver_data('VER')
    comparison.load_driver_data('HAM')
    
    # Create multi-channel comparisons
    channel_combinations = [
        ['Speed', 'RPM'],
        ['Speed', 'Throttle', 'Brake'],
        ['RPM', 'nGear'],
        ['Speed', 'RPM', 'Throttle', 'Brake']
    ]
    
    for i, channels in enumerate(channel_combinations):
        print(f"Creating multi-channel comparison {i+1}: {', '.join(channels)}")
        try:
            channel_names = '_'.join([ch.lower() for ch in channels])
            comparison.plot_multiple_channels('VER', 'HAM', channels, f'./telemetry_examples/ver_vs_ham_multi_{channel_names}.png')
        except Exception as e:
            print(f"  ⚠️  Could not create multi-channel comparison: {e}")
    
    print("✅ Multi-channel telemetry comparisons created!")

def example_driver_analysis():
    """Analyze individual driver telemetry characteristics."""
    print("\n=== Driver Telemetry Analysis ===")
    
    comparison = F1TelemetryComparison("./f1_data_comprehensive")
    
    # Load session data
    session_file = None
    for file in os.listdir("./f1_data_comprehensive"):
        if file.endswith('.json') and not file.endswith('_telemetry'):
            session_file = os.path.join("./f1_data_comprehensive", file)
            break
    
    comparison.load_session_data(session_file)
    
    # Analyze top drivers
    top_drivers = ['VER', 'HAM', 'NOR', 'SAI']
    
    for driver in top_drivers:
        if comparison.load_driver_data(driver):
            print(f"\n{driver} Telemetry Analysis:")
            
            # Get basic info
            driver_info = comparison.driver_data[driver]
            print(f"  Lap Time: {driver_info['lap_time']}")
            print(f"  Team: {driver_info['team']}")
            print(f"  Compound: {driver_info['compound']}")
            
            # Analyze key telemetry channels
            channels_to_analyze = ['Speed', 'RPM', 'Throttle']
            
            for channel in channels_to_analyze:
                try:
                    time_data, telemetry_data = comparison.get_telemetry_data(driver, channel)
                    
                    if telemetry_data:
                        print(f"  {channel}:")
                        print(f"    Max: {max(telemetry_data):.1f}")
                        print(f"    Min: {min(telemetry_data):.1f}")
                        print(f"    Average: {sum(telemetry_data)/len(telemetry_data):.1f}")
                        
                        # Channel-specific analysis
                        if channel == 'Speed':
                            high_speed_time = sum(1 for s in telemetry_data if s > 200) / len(telemetry_data) * 100
                            print(f"    Time >200 km/h: {high_speed_time:.1f}%")
                        elif channel == 'RPM':
                            high_rpm_time = sum(1 for r in telemetry_data if r > 10000) / len(telemetry_data) * 100
                            print(f"    Time >10k RPM: {high_rpm_time:.1f}%")
                        elif channel == 'Throttle':
                            full_throttle_time = sum(1 for t in telemetry_data if t > 95) / len(telemetry_data) * 100
                            print(f"    Full throttle time: {full_throttle_time:.1f}%")
                
                except Exception as e:
                    print(f"    ⚠️  Could not analyze {channel}: {e}")

def example_available_channels():
    """Show available telemetry channels."""
    print("\n=== Available Telemetry Channels ===")
    
    comparison = F1TelemetryComparison("./f1_data_comprehensive")
    
    # Load session data
    session_file = None
    for file in os.listdir("./f1_data_comprehensive"):
        if file.endswith('.json') and not file.endswith('_telemetry'):
            session_file = os.path.join("./f1_data_comprehensive", file)
            break
    
    comparison.load_session_data(session_file)
    comparison.load_driver_data('VER')
    
    channels = comparison.list_available_channels('VER')
    print(f"Available telemetry channels:")
    for channel in channels:
        channel_info = comparison.telemetry_channels.get(channel, {'description': channel, 'unit': ''})
        print(f"  - {channel}: {channel_info['description']} ({channel_info['unit']})")

if __name__ == "__main__":
    print("F1 Telemetry Comparison - Example Usage")
    print("=" * 50)
    
    # Run examples
    example_basic_telemetry_comparisons()
    example_sector_telemetry_comparisons()
    example_multi_channel_comparisons()
    example_driver_analysis()
    example_available_channels()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nGenerated files in ./telemetry_examples/:")
    print("- Individual channel comparisons (Speed, RPM, Throttle, Brake, nGear, DRS)")
    print("- Sector-based comparisons")
    print("- Multi-channel comparisons")
    
    print("\nTo create your own comparisons:")
    print("python telemetry_comparison.py --driver1 VER --driver2 HAM --channel RPM --save my_rpm_comparison.png")
    print("python telemetry_comparison.py --driver1 VER --driver2 HAM --channel Throttle --sectors --save my_throttle_sectors.png")
    print("python telemetry_comparison.py --driver1 VER --driver2 HAM --channels Speed RPM Throttle --save my_multi_channel.png")
    print("python telemetry_comparison.py --driver1 VER --list-channels")

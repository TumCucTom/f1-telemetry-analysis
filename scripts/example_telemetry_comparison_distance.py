"""
Example usage of the F1 Telemetry Comparison Tool (Distance-Based)
================================================================

This script demonstrates how to use the distance-based telemetry comparison tool
to create telemetry visualizations using distance as a percentage of total lap distance.
"""

from telemetry_comparison_distance import F1TelemetryComparisonDistance
import os

def example_basic_distance_comparisons():
    """Basic distance-based telemetry comparisons for different channels."""
    print("=== Basic Distance-Based Telemetry Comparison Examples ===")
    
    # Initialize the comparison tool
    comparison = F1TelemetryComparisonDistance("./f1_data_comprehensive")
    
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
    os.makedirs("./distance_examples", exist_ok=True)
    
    # Test different telemetry channels
    channels = ['Speed', 'RPM', 'Throttle', 'Brake', 'nGear', 'DRS']
    
    for channel in channels:
        print(f"Creating {channel} distance-based comparison...")
        try:
            comparison.plot_telemetry_comparison('VER', 'HAM', channel, f'./distance_examples/ver_vs_ham_{channel.lower()}_distance.png')
        except Exception as e:
            print(f"  ⚠️  Could not create {channel} comparison: {e}")
    
    print("✅ Basic distance-based telemetry comparisons created!")

def example_sector_distance_comparisons():
    """Distance-based telemetry comparisons with sector information."""
    print("\n=== Sector Distance-Based Telemetry Comparison Examples ===")
    
    comparison = F1TelemetryComparisonDistance("./f1_data_comprehensive")
    
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
        print(f"Creating {channel} distance-based comparison with sectors...")
        try:
            comparison.plot_telemetry_with_sectors('VER', 'HAM', channel, f'./distance_examples/ver_vs_ham_{channel.lower()}_distance_sectors.png')
        except Exception as e:
            print(f"  ⚠️  Could not create {channel} sectors comparison: {e}")
    
    print("✅ Sector distance-based telemetry comparisons created!")

def example_multi_channel_distance_comparisons():
    """Multi-channel distance-based telemetry comparisons."""
    print("\n=== Multi-Channel Distance-Based Telemetry Comparison Examples ===")
    
    comparison = F1TelemetryComparisonDistance("./f1_data_comprehensive")
    
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
        print(f"Creating multi-channel distance comparison {i+1}: {', '.join(channels)}")
        try:
            channel_names = '_'.join([ch.lower() for ch in channels])
            comparison.plot_multiple_channels('VER', 'HAM', channels, f'./distance_examples/ver_vs_ham_multi_{channel_names}_distance.png')
        except Exception as e:
            print(f"  ⚠️  Could not create multi-channel comparison: {e}")
    
    print("✅ Multi-channel distance-based telemetry comparisons created!")

def example_distance_analysis():
    """Analyze distance-based telemetry characteristics."""
    print("\n=== Distance-Based Telemetry Analysis ===")
    
    comparison = F1TelemetryComparisonDistance("./f1_data_comprehensive")
    
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
            print(f"\n{driver} Distance-Based Analysis:")
            
            # Get basic info
            driver_info = comparison.driver_data[driver]
            print(f"  Lap Time: {driver_info['lap_time']}")
            print(f"  Team: {driver_info['team']}")
            print(f"  Compound: {driver_info['compound']}")
            
            # Analyze distance-based telemetry
            try:
                distance_percentages = comparison.calculate_distance_percentage(driver)
                print(f"  Total Distance Points: {len(distance_percentages)}")
                print(f"  Distance Range: {min(distance_percentages):.1f}% - {max(distance_percentages):.1f}%")
                
                # Analyze key telemetry channels at different track positions
                channels_to_analyze = ['Speed', 'RPM', 'Throttle']
                
                for channel in channels_to_analyze:
                    try:
                        distance_data, telemetry_data = comparison.get_telemetry_data(driver, channel)
                        
                        if telemetry_data:
                            print(f"  {channel} by Track Position:")
                            
                            # Analyze different track sections
                            track_sections = [
                                (0, 25, "Start/Straight"),
                                (25, 50, "Mid-Lap"),
                                (50, 75, "Mid-Lap"),
                                (75, 100, "End/Straight")
                            ]
                            
                            for start, end, section_name in track_sections:
                                section_data = [data for dist, data in zip(distance_data, telemetry_data) 
                                              if start <= dist <= end]
                                if section_data:
                                    avg_value = sum(section_data) / len(section_data)
                                    max_value = max(section_data)
                                    print(f"    {section_name} ({start}-{end}%): Avg={avg_value:.1f}, Max={max_value:.1f}")
                
            except Exception as e:
                print(f"    ⚠️  Could not analyze distance data: {e}")

def example_track_position_comparison():
    """Compare drivers at specific track positions."""
    print("\n=== Track Position Comparison ===")
    
    comparison = F1TelemetryComparisonDistance("./f1_data_comprehensive")
    
    # Load session data
    session_file = None
    for file in os.listdir("./f1_data_comprehensive"):
        if file.endswith('.json') and not file.endswith('_telemetry'):
            session_file = os.path.join("./f1_data_comprehensive", file)
            break
    
    comparison.load_session_data(session_file)
    comparison.load_driver_data('VER')
    comparison.load_driver_data('HAM')
    
    # Compare at specific track positions
    track_positions = [
        (10, "Early Lap"),
        (25, "Sector 1 End"),
        (50, "Mid Lap"),
        (75, "Sector 3 Start"),
        (90, "Final Straight")
    ]
    
    print("Speed Comparison at Track Positions:")
    print("Position | VER | HAM | Difference")
    print("-" * 40)
    
    for position, description in track_positions:
        try:
            # Get speed data
            ver_distance, ver_speed = comparison.get_telemetry_data('VER', 'Speed')
            ham_distance, ham_speed = comparison.get_telemetry_data('HAM', 'Speed')
            
            # Find closest data points to the track position
            ver_idx = min(range(len(ver_distance)), key=lambda i: abs(ver_distance[i] - position))
            ham_idx = min(range(len(ham_distance)), key=lambda i: abs(ham_distance[i] - position))
            
            ver_speed_at_pos = ver_speed[ver_idx]
            ham_speed_at_pos = ham_speed[ham_idx]
            speed_diff = ver_speed_at_pos - ham_speed_at_pos
            
            print(f"{position:8.0f}% | {ver_speed_at_pos:4.0f} | {ham_speed_at_pos:4.0f} | {speed_diff:+6.1f} km/h")
            
        except Exception as e:
            print(f"{position:8.0f}% | Error: {e}")

if __name__ == "__main__":
    print("F1 Telemetry Comparison (Distance-Based) - Example Usage")
    print("=" * 60)
    
    # Run examples
    example_basic_distance_comparisons()
    example_sector_distance_comparisons()
    example_multi_channel_distance_comparisons()
    example_distance_analysis()
    example_track_position_comparison()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nGenerated files in ./distance_examples/:")
    print("- Individual channel comparisons (Speed, RPM, Throttle, Brake, nGear, DRS)")
    print("- Sector-based comparisons")
    print("- Multi-channel comparisons")
    print("- All using distance as percentage of lap")
    
    print("\nTo create your own distance-based comparisons:")
    print("python telemetry_comparison_distance.py --driver1 VER --driver2 HAM --channel RPM --save my_rpm_distance.png")
    print("python telemetry_comparison_distance.py --driver1 VER --driver2 HAM --channel Throttle --sectors --save my_throttle_distance_sectors.png")
    print("python telemetry_comparison_distance.py --driver1 VER --driver2 HAM --channels Speed RPM Throttle --save my_multi_distance.png")

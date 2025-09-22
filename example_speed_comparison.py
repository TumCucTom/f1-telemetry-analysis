"""
Example usage of the F1 Speed Comparison Tool
============================================

This script demonstrates how to use the speed comparison tool to create
various speed trace visualizations.
"""

from speed_comparison import F1SpeedComparison
import os

def example_basic_comparison():
    """Basic speed comparison between two drivers."""
    print("=== Basic Speed Comparison Example ===")
    
    # Initialize the comparison tool
    comparison = F1SpeedComparison("./f1_data_comprehensive")
    
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
    
    # Create basic speed comparison
    print("Creating VER vs HAM speed comparison...")
    comparison.plot_speed_comparison('VER', 'HAM', './examples/ver_vs_ham_basic.png')
    
    # Create comparison with sectors
    print("Creating VER vs HAM speed comparison with sectors...")
    comparison.plot_speed_with_sectors('VER', 'HAM', './examples/ver_vs_ham_sectors.png')
    
    # Create multiple comparisons
    comparisons = [
        ('VER', 'HAM', 'Red Bull vs Mercedes'),
        ('NOR', 'SAI', 'McLaren vs Ferrari'),
        ('VER', 'NOR', 'Red Bull vs McLaren'),
        ('HAM', 'SAI', 'Mercedes vs Ferrari')
    ]
    
    for driver1, driver2, description in comparisons:
        print(f"Creating {description} comparison...")
        comparison.plot_speed_comparison(driver1, driver2, f'./examples/{driver1}_vs_{driver2}.png')
    
    print("✅ All speed comparisons created successfully!")

def example_analyze_driver_performance():
    """Analyze individual driver performance."""
    print("\n=== Driver Performance Analysis Example ===")
    
    comparison = F1SpeedComparison("./f1_data_comprehensive")
    
    # Load session data
    session_file = None
    for file in os.listdir("./f1_data_comprehensive"):
        if file.endswith('.json') and not file.endswith('_telemetry'):
            session_file = os.path.join("./f1_data_comprehensive", file)
            break
    
    if not session_file:
        print("❌ Could not find session JSON file")
        return
    
    comparison.load_session_data(session_file)
    
    # Analyze top drivers
    top_drivers = ['VER', 'HAM', 'NOR', 'SAI', 'LEC']
    
    for driver in top_drivers:
        if comparison.load_driver_data(driver):
            time_data, speed_data = comparison.get_speed_data(driver)
            driver_info = comparison.driver_data[driver]
            
            print(f"\n{driver} ({driver_info['team']}):")
            print(f"  Lap Time: {driver_info['lap_time']}")
            print(f"  Max Speed: {max(speed_data):.1f} km/h")
            print(f"  Min Speed: {min(speed_data):.1f} km/h")
            print(f"  Average Speed: {sum(speed_data)/len(speed_data):.1f} km/h")
            print(f"  Tyre: {driver_info['compound']} (Life: {driver_info['tyre_life']})")
            
            # Calculate time spent in different speed zones
            low_speed = sum(1 for s in speed_data if s < 100)
            medium_speed = sum(1 for s in speed_data if 100 <= s < 200)
            high_speed = sum(1 for s in speed_data if s >= 200)
            total_points = len(speed_data)
            
            print(f"  Speed Distribution:")
            print(f"    Low Speed (<100 km/h): {low_speed/total_points*100:.1f}%")
            print(f"    Medium Speed (100-200 km/h): {medium_speed/total_points*100:.1f}%")
            print(f"    High Speed (>200 km/h): {high_speed/total_points*100:.1f}%")

def example_create_comparison_matrix():
    """Create a matrix of speed comparisons."""
    print("\n=== Speed Comparison Matrix Example ===")
    
    comparison = F1SpeedComparison("./f1_data_comprehensive")
    
    # Load session data
    session_file = None
    for file in os.listdir("./f1_data_comprehensive"):
        if file.endswith('.json') and not file.endswith('_telemetry'):
            session_file = os.path.join("./f1_data_comprehensive", file)
            break
    
    if not session_file:
        print("❌ Could not find session JSON file")
        return
    
    comparison.load_session_data(session_file)
    
    # Load top 4 drivers
    top_drivers = ['VER', 'HAM', 'NOR', 'SAI']
    for driver in top_drivers:
        comparison.load_driver_data(driver)
    
    # Create all possible pairwise comparisons
    import itertools
    pairs = list(itertools.combinations(top_drivers, 2))
    
    print(f"Creating {len(pairs)} pairwise comparisons...")
    
    for i, (driver1, driver2) in enumerate(pairs, 1):
        print(f"  {i}/{len(pairs)}: {driver1} vs {driver2}")
        comparison.plot_speed_comparison(
            driver1, driver2, 
            f'./examples/matrix_{driver1}_vs_{driver2}.png'
        )
    
    print("✅ Speed comparison matrix created!")

if __name__ == "__main__":
    print("F1 Speed Comparison - Example Usage")
    print("=" * 50)
    
    # Create examples directory
    os.makedirs("./examples", exist_ok=True)
    
    # Run examples
    example_basic_comparison()
    example_analyze_driver_performance()
    example_create_comparison_matrix()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nGenerated files:")
    print("- ./examples/ver_vs_ham_basic.png")
    print("- ./examples/ver_vs_ham_sectors.png")
    print("- ./examples/nor_vs_sai.png")
    print("- ./examples/ver_vs_nor.png")
    print("- ./examples/ham_vs_sai.png")
    print("- ./examples/matrix_*.png")
    
    print("\nTo create your own comparisons:")
    print("python speed_comparison.py --driver1 VER --driver2 HAM --save my_comparison.png")
    print("python speed_comparison.py --driver1 NOR --driver2 SAI --sectors --save my_comparison_with_sectors.png")

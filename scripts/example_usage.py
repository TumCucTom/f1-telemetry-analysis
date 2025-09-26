"""
Example usage of the F1 Data Scraper
====================================

This script demonstrates how to use the F1DataScraper class to collect
comprehensive telemetry data for F1 qualifying sessions.
"""

from f1_data_scraper import F1DataScraper
import json
import pandas as pd

def example_basic_usage():
    """Basic example of scraping F1 data."""
    print("=== Basic F1 Data Scraping Example ===")
    
    # Initialize scraper for 2024 season
    scraper = F1DataScraper(year=2024, cache_dir="./f1_cache")
    
    # Load a qualifying session (example: Azerbaijan Grand Prix)
    if scraper.load_session("Azerbaijan Grand Prix", "Qualifying"):
        print("✓ Session loaded successfully")
        
        # Scrape all driver data
        data = scraper.scrape_all_drivers()
        
        # Save to files
        scraper.save_data(data, "./f1_data")
        
        # Print summary
        print(f"\nSession: {data['session_info']['event']} - {data['session_info']['session']}")
        print(f"Track: {data['session_info']['track']}, {data['session_info']['country']}")
        print(f"Date: {data['session_info']['session_date']}")
        print(f"Drivers processed: {data['summary']['drivers_processed']}")
        
        # Show top 3 fastest laps
        print("\nTop 3 Fastest Laps:")
        driver_times = []
        for driver, driver_data in data['drivers'].items():
            if 'error' not in driver_data:
                driver_times.append({
                    'driver': driver,
                    'team': driver_data['team'],
                    'lap_time': driver_data['lap_time'],
                    'lap_time_seconds': driver_data['lap_time_seconds']
                })
        
        # Sort by lap time
        driver_times.sort(key=lambda x: x['lap_time_seconds'])
        
        for i, driver_info in enumerate(driver_times[:3], 1):
            print(f"{i}. {driver_info['driver']} ({driver_info['team']}) - {driver_info['lap_time']}")
    
    else:
        print("✗ Failed to load session")

def example_individual_driver_analysis():
    """Example of analyzing individual driver telemetry."""
    print("\n=== Individual Driver Analysis Example ===")
    
    scraper = F1DataScraper(year=2024, cache_dir="./f1_cache")
    
    if scraper.load_session("Azerbaijan Grand Prix", "Qualifying"):
        # Get telemetry for a specific driver (e.g., Max Verstappen)
        driver = "VER"  # You can change this to any driver abbreviation
        
        try:
            telemetry_data = scraper.get_comprehensive_telemetry(driver)
            
            print(f"\nDriver: {driver}")
            print(f"Team: {telemetry_data['team']}")
            print(f"Lap Time: {telemetry_data['lap_time']}")
            print(f"Compound: {telemetry_data['compound']}")
            print(f"Tyre Life: {telemetry_data['tyre_life']}")
            
            # Analyze telemetry data
            telemetry = telemetry_data['telemetry']
            
            if 'Speed' in telemetry:
                speeds = [s for s in telemetry['Speed'] if s is not None]
                if speeds:
                    print(f"Max Speed: {max(speeds):.1f} km/h")
                    print(f"Min Speed: {min(speeds):.1f} km/h")
                    print(f"Average Speed: {sum(speeds)/len(speeds):.1f} km/h")
            
            if 'Throttle' in telemetry:
                throttle_data = [t for t in telemetry['Throttle'] if t is not None]
                if throttle_data:
                    full_throttle_percentage = (sum(1 for t in throttle_data if t >= 95) / len(throttle_data)) * 100
                    print(f"Full Throttle Percentage: {full_throttle_percentage:.1f}%")
            
            if 'Brake' in telemetry:
                brake_data = [b for b in telemetry['Brake'] if b is not None]
                if brake_data:
                    braking_percentage = (sum(1 for b in brake_data if b > 0) / len(brake_data)) * 100
                    print(f"Braking Percentage: {braking_percentage:.1f}%")
            
            # Show available telemetry channels
            print(f"\nAvailable telemetry channels: {len(telemetry)}")
            for channel in sorted(telemetry.keys()):
                data_points = len([d for d in telemetry[channel] if d is not None])
                print(f"  - {channel}: {data_points} data points")
        
        except Exception as e:
            print(f"Error analyzing driver {driver}: {e}")

def example_data_export_formats():
    """Example showing different data export formats."""
    print("\n=== Data Export Formats Example ===")
    
    scraper = F1DataScraper(year=2024, cache_dir="./f1_cache")
    
    if scraper.load_session("Azerbaijan Grand Prix", "Qualifying"):
        data = scraper.scrape_all_drivers()
        
        # Export to different formats
        scraper.save_data(data, "./f1_data")
        
        print("Data exported in multiple formats:")
        print("1. JSON: Complete telemetry data with all channels")
        print("2. CSV: Summary with lap times and basic info")
        print("3. Individual JSON files: One file per driver with full telemetry")
        
        # Show structure of exported data
        print(f"\nData structure:")
        print(f"- Session info: {len(data['session_info'])} fields")
        print(f"- Drivers: {len(data['drivers'])} drivers")
        print(f"- Summary: {len(data['summary'])} summary fields")
        
        # Example of accessing specific telemetry data
        if data['drivers']:
            first_driver = list(data['drivers'].keys())[0]
            driver_data = data['drivers'][first_driver]
            
            if 'telemetry' in driver_data:
                telemetry_channels = list(driver_data['telemetry'].keys())
                print(f"\nExample telemetry channels for {first_driver}:")
                for channel in telemetry_channels[:10]:  # Show first 10 channels
                    data_count = len([d for d in driver_data['telemetry'][channel] if d is not None])
                    print(f"  - {channel}: {data_count} data points")

def example_available_events():
    """Example showing how to find available events."""
    print("\n=== Available Events Example ===")
    
    import fastf1
    
    # Get available events for 2024
    try:
        schedule = fastf1.get_event_schedule(2024)
        print("Available events for 2024:")
        
        for _, event in schedule.iterrows():
            print(f"- {event['EventName']} ({event['Country']}) - {event['EventDate']}")
    
    except Exception as e:
        print(f"Error getting event schedule: {e}")

if __name__ == "__main__":
    print("F1 Data Scraper - Example Usage")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_individual_driver_analysis()
    example_data_export_formats()
    example_available_events()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run the scraper from command line:")
    print("python f1_data_scraper.py --year 2024 --event 'Azerbaijan Grand Prix' --session 'Qualifying'")

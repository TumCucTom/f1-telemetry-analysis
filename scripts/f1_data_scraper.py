"""
F1 Telemetry Data Scraper
=========================

This script scrapes comprehensive telemetry data for each driver's best qualifying lap
using the FastF1 API. It collects all available sensor data including speed, RPM, 
throttle, brake, ERS, tyre temperatures, and more.

Usage:
    python f1_data_scraper.py --year 2024 --event "Azerbaijan Grand Prix" --session "Qualifying"
"""

import fastf1
import pandas as pd
import numpy as np
import json
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1DataScraper:
    """Comprehensive F1 telemetry data scraper for qualifying sessions."""
    
    def __init__(self, year: int, cache_dir: str = "./f1_cache"):
        """
        Initialize the F1 data scraper.
        
        Args:
            year: F1 season year
            cache_dir: Directory to store cached data
        """
        self.year = year
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Enable caching
        fastf1.Cache.enable_cache(cache_dir)
        logger.info(f"Cache enabled at: {cache_dir}")
        
        # Initialize session variables
        self.session = None
        self.session_data = {}
        
    def load_session(self, event_name: str, session_name: str = "Qualifying") -> bool:
        """
        Load F1 session data.
        
        Args:
            event_name: Name of the F1 event (e.g., "Azerbaijan Grand Prix")
            session_name: Name of the session (default: "Qualifying")
            
        Returns:
            bool: True if session loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading {self.year} {event_name} - {session_name}")
            self.session = fastf1.get_session(self.year, event_name, session_name)
            self.session.load()
            logger.info("Session loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False
    
    def get_driver_best_laps(self) -> Dict[str, Any]:
        """
        Get the best lap for each driver in the session.
        
        Returns:
            Dict containing driver best lap data
        """
        if not self.session:
            raise ValueError("Session not loaded. Call load_session() first.")
        
        logger.info("Extracting driver best laps...")
        best_laps = {}
        
        # Get all laps
        laps = self.session.laps
        
        # Debug: Print available columns
        logger.info(f"Available columns in laps data: {list(laps.columns)}")
        
        # Group by driver and find best lap for each
        for driver in tqdm(laps['Driver'].unique(), desc="Processing drivers"):
            driver_laps = laps[laps['Driver'] == driver]
            
            # Find the fastest lap
            best_lap = driver_laps.loc[driver_laps['LapTime'].idxmin()]
            
            # Get team name - try different possible column names
            team_name = None
            for team_col in ['TeamName', 'Team', 'Constructor']:
                if team_col in best_lap.index:
                    team_name = best_lap[team_col]
                    break
            
            best_laps[driver] = {
                'lap_number': best_lap['LapNumber'],
                'lap_time': str(best_lap['LapTime']),
                'lap_time_seconds': best_lap['LapTime'].total_seconds(),
                'sector_1_time': str(best_lap['Sector1Time']) if pd.notna(best_lap['Sector1Time']) else None,
                'sector_2_time': str(best_lap['Sector2Time']) if pd.notna(best_lap['Sector2Time']) else None,
                'sector_3_time': str(best_lap['Sector3Time']) if pd.notna(best_lap['Sector3Time']) else None,
                'compound': best_lap['Compound'],
                'tyre_life': best_lap['TyreLife'],
                'team': team_name,
                'driver_number': best_lap['DriverNumber']
            }
        
        return best_laps
    
    def get_comprehensive_telemetry(self, driver: str) -> Dict[str, Any]:
        """
        Get comprehensive telemetry data for a driver's best lap.
        
        Args:
            driver: Driver abbreviation (e.g., 'VER', 'HAM')
            
        Returns:
            Dict containing all available telemetry data
        """
        if not self.session:
            raise ValueError("Session not loaded. Call load_session() first.")
        
        logger.info(f"Extracting comprehensive telemetry for {driver}...")
        
        # Get driver's best lap
        driver_laps = self.session.laps[self.session.laps['Driver'] == driver]
        best_lap = driver_laps.loc[driver_laps['LapTime'].idxmin()]
        
        # Get team name - try different possible column names
        team_name = None
        for team_col in ['TeamName', 'Team', 'Constructor']:
            if team_col in best_lap.index:
                team_name = best_lap[team_col]
                break
        
        # Initialize telemetry dictionary
        telemetry_data = {
            'driver': driver,
            'lap_number': best_lap['LapNumber'],
            'lap_time': str(best_lap['LapTime']),
            'lap_time_seconds': best_lap['LapTime'].total_seconds(),
            'team': team_name,
            'compound': best_lap['Compound'],
            'tyre_life': best_lap['TyreLife'],
            'telemetry': {},
            'position_data': {},
            'weather_data': {},
            'lap_details': {}
        }
        
        # 1. CAR TELEMETRY DATA
        try:
            car_telemetry = best_lap.get_car_data()
            logger.info(f"Car telemetry: {len(car_telemetry)} records, {len(car_telemetry.columns)} channels")
            
            for channel in car_telemetry.columns:
                data = car_telemetry[channel].tolist()
                
                # Handle NaN values
                if pd.api.types.is_numeric_dtype(car_telemetry[channel]):
                    data = [float(x) if pd.notna(x) else None for x in data]
                else:
                    data = [str(x) if pd.notna(x) else None for x in data]
                
                telemetry_data['telemetry'][channel] = data
                
        except Exception as e:
            logger.warning(f"Could not extract car telemetry for {driver}: {e}")
        
        # 2. POSITION DATA
        try:
            position_data = best_lap.get_pos_data()
            logger.info(f"Position data: {len(position_data)} records, {len(position_data.columns)} channels")
            
            for channel in position_data.columns:
                data = position_data[channel].tolist()
                
                # Handle NaN values
                if pd.api.types.is_numeric_dtype(position_data[channel]):
                    data = [float(x) if pd.notna(x) else None for x in data]
                else:
                    data = [str(x) if pd.notna(x) else None for x in data]
                
                telemetry_data['position_data'][channel] = data
                
        except Exception as e:
            logger.warning(f"Could not extract position data for {driver}: {e}")
        
        # 3. WEATHER DATA (session-wide, not lap-specific)
        try:
            weather_data = self.session.weather_data
            logger.info(f"Weather data: {len(weather_data)} records, {len(weather_data.columns)} channels")
            
            # Get weather data closest to the lap time
            lap_start_time = best_lap['LapStartTime']
            
            # Find closest weather record
            time_diffs = abs((weather_data['Time'] - lap_start_time).dt.total_seconds())
            closest_weather_idx = time_diffs.idxmin()
            closest_weather = weather_data.iloc[closest_weather_idx]
            
            for channel in weather_data.columns:
                if channel != 'Time':  # Skip time column
                    value = closest_weather[channel]
                    if pd.notna(value):
                        if pd.api.types.is_numeric_dtype(weather_data[channel]):
                            telemetry_data['weather_data'][channel] = float(value)
                        else:
                            telemetry_data['weather_data'][channel] = str(value)
                    else:
                        telemetry_data['weather_data'][channel] = None
                        
        except Exception as e:
            logger.warning(f"Could not extract weather data for {driver}: {e}")
        
        # 4. LAP DETAILS (from laps dataframe)
        try:
            lap_detail_columns = [
                'Compound', 'TyreLife', 'Stint', 'FreshTyre', 'IsAccurate', 
                'IsPersonalBest', 'SpeedFL', 'SpeedI1', 'SpeedI2', 'SpeedST',
                'Position', 'TrackStatus', 'Deleted', 'DeletedReason', 'FastF1Generated'
            ]
            
            for col in lap_detail_columns:
                if col in best_lap.index:
                    value = best_lap[col]
                    if pd.notna(value):
                        if pd.api.types.is_numeric_dtype(best_lap[col]):
                            telemetry_data['lap_details'][col] = float(value)
                        else:
                            telemetry_data['lap_details'][col] = str(value)
                    else:
                        telemetry_data['lap_details'][col] = None
                        
        except Exception as e:
            logger.warning(f"Could not extract lap details for {driver}: {e}")
        
        # 5. SECTOR TIMES
        try:
            sector_times = []
            for i in range(1, 4):
                sector_col = f'Sector{i}Time'
                if sector_col in best_lap.index and pd.notna(best_lap[sector_col]):
                    sector_times.append(str(best_lap[sector_col]))
                else:
                    sector_times.append(None)
            
            telemetry_data['sector_times'] = sector_times
            
        except Exception as e:
            logger.warning(f"Could not extract sector times for {driver}: {e}")
            telemetry_data['sector_times'] = [None, None, None]
        
        # 6. SESSION RESULTS DATA
        try:
            results = self.session.results
            driver_result = results[results['Abbreviation'] == driver]
            
            if not driver_result.empty:
                result = driver_result.iloc[0]
                telemetry_data['session_result'] = {
                    'position': result.get('Position', None),
                    'grid_position': result.get('GridPosition', None),
                    'points': result.get('Points', None),
                    'status': result.get('Status', None),
                    'q1_time': str(result.get('Q1', None)) if pd.notna(result.get('Q1', None)) else None,
                    'q2_time': str(result.get('Q2', None)) if pd.notna(result.get('Q2', None)) else None,
                    'q3_time': str(result.get('Q3', None)) if pd.notna(result.get('Q3', None)) else None,
                    'team_name': result.get('TeamName', None),
                    'team_color': result.get('TeamColor', None),
                    'full_name': result.get('FullName', None),
                    'country_code': result.get('CountryCode', None)
                }
        except Exception as e:
            logger.warning(f"Could not extract session results for {driver}: {e}")
        
        return telemetry_data
    
    def _collect_session_data(self, session_data: Dict[str, Any]) -> None:
        """
        Collect session-wide data (weather, track status, race control, etc.).
        
        Args:
            session_data: Dictionary to populate with session data
        """
        logger.info("Collecting session-wide data...")
        
        # 1. WEATHER DATA
        try:
            weather_data = self.session.weather_data
            logger.info(f"Collecting weather data: {len(weather_data)} records")
            
            # Convert weather data to list of dictionaries
            weather_records = []
            for _, row in weather_data.iterrows():
                record = {}
                for col in weather_data.columns:
                    value = row[col]
                    if pd.notna(value):
                        if col == 'Time':
                            record[col] = str(value)
                        elif pd.api.types.is_numeric_dtype(weather_data[col]):
                            record[col] = float(value)
                        else:
                            record[col] = str(value)
                    else:
                        record[col] = None
                weather_records.append(record)
            
            session_data['weather_data'] = {
                'records': weather_records,
                'total_records': len(weather_records),
                'columns': list(weather_data.columns)
            }
            
        except Exception as e:
            logger.warning(f"Could not collect weather data: {e}")
            session_data['weather_data'] = {'error': str(e)}
        
        # 2. TRACK STATUS
        try:
            track_status = self.session.track_status
            logger.info(f"Collecting track status: {len(track_status)} records")
            
            track_records = []
            for _, row in track_status.iterrows():
                record = {}
                for col in track_status.columns:
                    value = row[col]
                    if pd.notna(value):
                        if col == 'Time':
                            record[col] = str(value)
                        elif pd.api.types.is_numeric_dtype(track_status[col]):
                            record[col] = float(value)
                        else:
                            record[col] = str(value)
                    else:
                        record[col] = None
                track_records.append(record)
            
            session_data['track_status'] = {
                'records': track_records,
                'total_records': len(track_records),
                'columns': list(track_status.columns)
            }
            
        except Exception as e:
            logger.warning(f"Could not collect track status: {e}")
            session_data['track_status'] = {'error': str(e)}
        
        # 3. RACE CONTROL MESSAGES
        try:
            rc_messages = self.session.race_control_messages
            logger.info(f"Collecting race control messages: {len(rc_messages)} records")
            
            rc_records = []
            for _, row in rc_messages.iterrows():
                record = {}
                for col in rc_messages.columns:
                    value = row[col]
                    if pd.notna(value):
                        if col == 'Time':
                            record[col] = str(value)
                        elif pd.api.types.is_numeric_dtype(rc_messages[col]):
                            record[col] = float(value)
                        else:
                            record[col] = str(value)
                    else:
                        record[col] = None
                rc_records.append(record)
            
            session_data['race_control_messages'] = {
                'records': rc_records,
                'total_records': len(rc_records),
                'columns': list(rc_messages.columns)
            }
            
        except Exception as e:
            logger.warning(f"Could not collect race control messages: {e}")
            session_data['race_control_messages'] = {'error': str(e)}
        
        # 4. CIRCUIT INFO
        try:
            circuit_info = self.session.get_circuit_info()
            logger.info("Collecting circuit information")
            
            # Extract circuit data
            circuit_data = {
                'corners': [],
                'track_length': None
            }
            
            # Get corners data
            if hasattr(circuit_info, 'corners') and circuit_info.corners is not None:
                corners_df = circuit_info.corners
                for _, corner in corners_df.iterrows():
                    corner_data = {}
                    for col in corners_df.columns:
                        value = corner[col]
                        if pd.notna(value):
                            if pd.api.types.is_numeric_dtype(corners_df[col]):
                                corner_data[col] = float(value)
                            else:
                                corner_data[col] = str(value)
                        else:
                            corner_data[col] = None
                    circuit_data['corners'].append(corner_data)
            
            # Try to get track length from different sources
            try:
                if hasattr(circuit_info, 'track_length'):
                    circuit_data['track_length'] = float(circuit_info.track_length)
            except:
                pass
            
            session_data['circuit_info'] = circuit_data
            
        except Exception as e:
            logger.warning(f"Could not collect circuit info: {e}")
            session_data['circuit_info'] = {'error': str(e)}
        
        # 5. SESSION RESULTS
        try:
            results = self.session.results
            logger.info(f"Collecting session results: {len(results)} drivers")
            
            results_records = []
            for _, row in results.iterrows():
                record = {}
                for col in results.columns:
                    value = row[col]
                    if pd.notna(value):
                        if col in ['Q1', 'Q2', 'Q3', 'Time']:
                            record[col] = str(value)
                        elif pd.api.types.is_numeric_dtype(results[col]):
                            record[col] = float(value)
                        else:
                            record[col] = str(value)
                    else:
                        record[col] = None
                results_records.append(record)
            
            session_data['session_results'] = {
                'records': results_records,
                'total_drivers': len(results_records),
                'columns': list(results.columns)
            }
            
        except Exception as e:
            logger.warning(f"Could not collect session results: {e}")
            session_data['session_results'] = {'error': str(e)}
    
    def scrape_all_drivers(self) -> Dict[str, Any]:
        """
        Scrape comprehensive telemetry data for all drivers.
        
        Returns:
            Dict containing all drivers' telemetry data
        """
        if not self.session:
            raise ValueError("Session not loaded. Call load_session() first.")
        
        logger.info("Starting comprehensive telemetry scraping for all drivers...")
        
        # Get basic lap information
        best_laps = self.get_driver_best_laps()
        
        # Initialize results
        results = {
            'session_info': {
                'year': self.year,
                'event': self.session.event['EventName'],
                'session': self.session.name,
                'session_date': str(self.session.date),
                'track': self.session.event['Location'],
                'country': self.session.event['Country'],
                'scraped_at': datetime.now().isoformat()
            },
            'session_data': {
                'weather_data': {},
                'track_status': {},
                'race_control_messages': {},
                'circuit_info': {},
                'session_results': {}
            },
            'drivers': {},
            'summary': {
                'total_drivers': len(best_laps),
                'drivers_processed': 0,
                'drivers_failed': 0
            }
        }
        
        # Collect session-wide data
        self._collect_session_data(results['session_data'])
        
        # Process each driver
        for driver in tqdm(best_laps.keys(), desc="Scraping driver telemetry"):
            try:
                telemetry_data = self.get_comprehensive_telemetry(driver)
                results['drivers'][driver] = telemetry_data
                results['summary']['drivers_processed'] += 1
                
            except Exception as e:
                logger.error(f"Failed to scrape telemetry for {driver}: {e}")
                results['drivers'][driver] = {
                    'driver': driver,
                    'error': str(e),
                    'basic_info': best_laps[driver]
                }
                results['summary']['drivers_failed'] += 1
        
        return results
    
    def save_data(self, data: Dict[str, Any], output_dir: str = "./f1_data") -> None:
        """
        Save scraped data to files.
        
        Args:
            data: Scraped data dictionary
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on session info
        event_name = data['session_info']['event'].replace(' ', '_').replace('Grand_Prix', 'GP')
        session_name = data['session_info']['session']
        year = data['session_info']['year']
        
        filename_base = f"{year}_{event_name}_{session_name}"
        
        # Save as JSON
        json_file = os.path.join(output_dir, f"{filename_base}.json")
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Data saved to: {json_file}")
        
        # Save summary as CSV
        summary_data = []
        for driver, driver_data in data['drivers'].items():
            if 'error' not in driver_data:
                summary_data.append({
                    'driver': driver,
                    'team': driver_data.get('team', ''),
                    'lap_time': driver_data.get('lap_time', ''),
                    'lap_time_seconds': driver_data.get('lap_time_seconds', 0),
                    'compound': driver_data.get('compound', ''),
                    'tyre_life': driver_data.get('tyre_life', 0),
                    'sector_1': driver_data.get('sector_times', [None, None, None])[0],
                    'sector_2': driver_data.get('sector_times', [None, None, None])[1],
                    'sector_3': driver_data.get('sector_times', [None, None, None])[2]
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('lap_time_seconds')
            csv_file = os.path.join(output_dir, f"{filename_base}_summary.csv")
            summary_df.to_csv(csv_file, index=False)
            logger.info(f"Summary saved to: {csv_file}")
        
        # Save individual driver telemetry as separate JSON files
        telemetry_dir = os.path.join(output_dir, f"{filename_base}_telemetry")
        os.makedirs(telemetry_dir, exist_ok=True)
        
        for driver, driver_data in data['drivers'].items():
            if 'error' not in driver_data:
                driver_file = os.path.join(telemetry_dir, f"{driver}_telemetry.json")
                with open(driver_file, 'w') as f:
                    json.dump(driver_data, f, indent=2, default=str)
        
        logger.info(f"Individual telemetry files saved to: {telemetry_dir}")


def main():
    """Main function to run the F1 data scraper."""
    parser = argparse.ArgumentParser(description='F1 Telemetry Data Scraper')
    parser.add_argument('--year', type=int, default=2024, help='F1 season year')
    parser.add_argument('--event', type=str, required=True, help='Event name (e.g., "Azerbaijan Grand Prix")')
    parser.add_argument('--session', type=str, default='Qualifying', help='Session name')
    parser.add_argument('--output-dir', type=str, default='./f1_data', help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='./f1_cache', help='Cache directory')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = F1DataScraper(year=args.year, cache_dir=args.cache_dir)
    
    # Load session
    if not scraper.load_session(args.event, args.session):
        logger.error("Failed to load session. Exiting.")
        return
    
    # Scrape data
    try:
        data = scraper.scrape_all_drivers()
        
        # Save data
        scraper.save_data(data, args.output_dir)
        
        logger.info("Data scraping completed successfully!")
        logger.info(f"Processed {data['summary']['drivers_processed']} drivers")
        if data['summary']['drivers_failed'] > 0:
            logger.warning(f"Failed to process {data['summary']['drivers_failed']} drivers")
        
    except Exception as e:
        logger.error(f"Error during data scraping: {e}")
        raise


if __name__ == "__main__":
    main()

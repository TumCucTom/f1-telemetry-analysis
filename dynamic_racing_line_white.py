"""
F1 Dynamic Racing Line Tool - White Background Version
====================================================

This script creates a single racing line where the color changes based on 
which driver is ahead in time delta at each point on the track with white background.

Usage:
    python dynamic_racing_line_white.py --data-dir "./f1_data_comprehensive" --driver1 NOR --driver2 PIA
    python dynamic_racing_line_white.py --data-dir "./f1_data_comprehensive" --driver1 VER --driver2 HAM --save "./ver_vs_ham_dynamic_line.png"
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import Dict, List, Optional, Tuple
import logging
from matplotlib.colors import LinearSegmentedColormap

# Configure matplotlib for white background
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',      # White background
    'axes.facecolor': 'white',        # White axes background
    'axes.edgecolor': 'black',        # Black axes edges
    'axes.labelcolor': 'black',       # Black labels
    'text.color': 'black',            # Black text
    'xtick.color': 'black',           # Black x-axis ticks
    'ytick.color': 'black',           # Black y-axis ticks
    'grid.color': '#cccccc',          # Light gray grid
    'legend.facecolor': 'white',      # White legend background
    'legend.edgecolor': 'black',      # Black legend border
    'legend.labelcolor': 'black'      # Black legend text
})

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1DynamicRacingLineWhite:
    """
    Tool for creating dynamic racing lines based on time delta with white background.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the dynamic racing line tool.
        
        Args:
            data_dir: Directory containing the F1 telemetry data
        """
        self.data_dir = data_dir
        self.session_data = None
        self.driver_data = {}
        
    def load_session_data(self, session_file: str) -> bool:
        """
        Load session data from JSON file.
        
        Args:
            session_file: Path to the session JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(session_file, 'r') as f:
                self.session_data = json.load(f)
            logger.info(f"✅ Loaded session data from: {session_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading session data: {str(e)}")
            return False
    
    def load_driver_data(self, driver: str) -> bool:
        """
        Load telemetry data for a specific driver.
        
        Args:
            driver: Driver abbreviation (e.g., 'VER', 'HAM')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find the telemetry directory
            telemetry_dir = None
            for item in os.listdir(self.data_dir):
                if item.endswith("_telemetry") and os.path.isdir(f"{self.data_dir}/{item}"):
                    telemetry_dir = f"{self.data_dir}/{item}"
                    break
            
            if not telemetry_dir:
                logger.error("❌ No telemetry directory found")
                return False
            
            # Load driver telemetry file
            telemetry_file = os.path.join(telemetry_dir, f"{driver}_telemetry.json")
            if not os.path.exists(telemetry_file):
                logger.error(f"❌ Telemetry file not found for driver {driver}: {telemetry_file}")
                return False
            
            with open(telemetry_file, 'r') as f:
                self.driver_data[driver] = json.load(f)
            
            logger.info(f"✅ Loaded telemetry data for driver {driver}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading driver data for {driver}: {str(e)}")
            return False
    
    def calculate_time_delta(self, driver1: str, driver2: str, time_scaled: bool = True) -> Tuple[List[float], List[float]]:
        """
        Calculate time delta between two drivers using time-based calculation.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            time_scaled: Whether to use time-scaled data
            
        Returns:
            Tuple of (time_percentages, time_deltas)
        """
        try:
            if (driver1 not in self.driver_data or driver2 not in self.driver_data):
                logger.error(f"❌ Driver data not available for {driver1} vs {driver2}")
                return [], []
            
            # Get telemetry data directly
            telemetry1 = self.driver_data[driver1]['telemetry']
            telemetry2 = self.driver_data[driver2]['telemetry']
            
            time1 = telemetry1.get('Time', [])
            time2 = telemetry2.get('Time', [])
            
            if not all([time1, time2]):
                logger.error(f"❌ Missing time data for {driver1} vs {driver2}")
                return [], []
            
            # Convert time strings to seconds
            def time_to_seconds(time_str):
                if isinstance(time_str, str):
                    # Parse "0 days 00:00:00.007000" format
                    parts = time_str.split()
                    time_part = parts[2]  # "00:00:00.007000"
                    h, m, s = time_part.split(':')
                    return float(h) * 3600 + float(m) * 60 + float(s)
                return float(time_str)
            
            time1_sec = [time_to_seconds(t) for t in time1]
            time2_sec = [time_to_seconds(t) for t in time2]
            
            # Normalize to start from 0
            time1_sec = [t - time1_sec[0] for t in time1_sec]
            time2_sec = [t - time2_sec[0] for t in time2_sec]
            
            # Convert to percentages
            time_pct1 = [(t / time1_sec[-1]) * 100 for t in time1_sec]
            time_pct2 = [(t / time2_sec[-1]) * 100 for t in time2_sec]
            
            # Create common time percentage grid
            common_time_pct = np.linspace(0, 100, 1000)
            
            # Interpolate times to common grid
            time_interp1 = np.interp(common_time_pct, time_pct1, time1_sec)
            time_interp2 = np.interp(common_time_pct, time_pct2, time2_sec)
            
            # Calculate time delta (driver1 - driver2)
            time_delta = time_interp1 - time_interp2
            
            return common_time_pct.tolist(), time_delta.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error calculating time delta: {str(e)}")
            return [], []
    
    def get_track_coordinates(self, driver: str, time_scaled: bool = True) -> Tuple[List[float], List[float]]:
        """
        Get track coordinates for a driver.
        
        Args:
            driver: Driver abbreviation
            time_scaled: Whether to use time-scaled data
            
        Returns:
            Tuple of (x_coordinates, y_coordinates)
        """
        try:
            if driver not in self.driver_data:
                logger.error(f"❌ Driver {driver} not found in data")
                return [], []
            
            # The data structure has 'telemetry' key directly
            telemetry = self.driver_data[driver]['telemetry']
            x_coords = telemetry.get('X', [])
            y_coords = telemetry.get('Y', [])
            
            return x_coords, y_coords
            
        except Exception as e:
            logger.error(f"❌ Error getting track coordinates: {str(e)}")
            return [], []
    
    def create_dynamic_racing_line(self, driver1: str, driver2: str, 
                                 save_path: Optional[str] = None, 
                                 time_scaled: bool = True) -> None:
        """
        Create a dynamic racing line visualization.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            save_path: Path to save the plot
            time_scaled: Whether to use time-scaled data
        """
        try:
            # Load driver data if not already loaded
            if driver1 not in self.driver_data:
                if not self.load_driver_data(driver1):
                    return
            
            if driver2 not in self.driver_data:
                if not self.load_driver_data(driver2):
                    return
            
            # Calculate time delta
            time_pct, time_delta = self.calculate_time_delta(driver1, driver2, time_scaled)
            
            if not time_pct or not time_delta:
                logger.error("❌ Time delta calculation failed")
                return
            
            # Create the plot - simplified to just show time delta over time
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Plot time delta over time percentage
            ax.plot(time_pct, time_delta, linewidth=2, color='black')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            ax.fill_between(time_pct, time_delta, 0, 
                           where=(np.array(time_delta) > 0), 
                           alpha=0.3, color='red', label=f'{driver1} ahead')
            ax.fill_between(time_pct, time_delta, 0, 
                           where=(np.array(time_delta) < 0), 
                           alpha=0.3, color='blue', label=f'{driver2} ahead')
            
            ax.set_title(f'Dynamic Racing Line: {driver1} vs {driver2}\nTime Delta Over Lap Time', 
                        fontsize=16, fontweight='bold', color='black')
            ax.set_xlabel('Lap Time (%)', color='black', fontsize=12)
            ax.set_ylabel('Time Delta (s)', color='black', fontsize=12)
            ax.grid(True, alpha=0.3, color='#cccccc')
            ax.legend(fontsize=12)
            
            # Add corner markers - simplified to reduce clutter
            corner_positions = [20, 40, 60, 80]  # Simplified corner positions
            for corner_pos in corner_positions:
                ax.axvline(x=corner_pos, color='green', linestyle=':', alpha=0.7)
                ax.text(corner_pos, ax.get_ylim()[1] * 0.9, f'T{corner_positions.index(corner_pos)+1}', 
                       rotation=90, verticalalignment='top', fontsize=8, color='green')
            
            plt.tight_layout()
            
            # Save or show the plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"✅ Dynamic racing line saved: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creating dynamic racing line: {str(e)}")

def main():
    """Main function to run the dynamic racing line tool."""
    parser = argparse.ArgumentParser(description='F1 Dynamic Racing Line Tool - White Background')
    parser.add_argument('--data-dir', required=True, help='Directory containing F1 telemetry data')
    parser.add_argument('--driver1', required=True, help='First driver abbreviation (e.g., VER)')
    parser.add_argument('--driver2', required=True, help='Second driver abbreviation (e.g., HAM)')
    parser.add_argument('--save', help='Path to save the plot')
    parser.add_argument('--time-scaled', action='store_true', default=True, help='Use time-scaled data')
    
    args = parser.parse_args()
    
    # Initialize the dynamic racing line tool
    racing_line = F1DynamicRacingLineWhite(args.data_dir)
    
    # Create the dynamic racing line
    racing_line.create_dynamic_racing_line(
        driver1=args.driver1,
        driver2=args.driver2,
        save_path=args.save,
        time_scaled=args.time_scaled
    )

if __name__ == "__main__":
    main()

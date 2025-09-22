"""
F1 Speed Comparison Visualization
=================================

This script creates speed trace comparisons between two drivers using the
comprehensive telemetry data collected by the F1 data scraper.

Usage:
    python speed_comparison.py --data-dir "./f1_data_comprehensive" --driver1 VER --driver2 HAM
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1SpeedComparison:
    """F1 speed comparison visualization tool."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the speed comparison tool.
        
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
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(session_file, 'r') as f:
                self.session_data = json.load(f)
            logger.info(f"Loaded session data: {self.session_data['session_info']['event']}")
            return True
        except Exception as e:
            logger.error(f"Failed to load session data: {e}")
            return False
    
    def load_driver_data(self, driver: str) -> bool:
        """
        Load telemetry data for a specific driver.
        
        Args:
            driver: Driver abbreviation (e.g., 'VER', 'HAM')
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Find the telemetry directory
            telemetry_dir = None
            for item in os.listdir(self.data_dir):
                if item.endswith('_telemetry'):
                    telemetry_dir = os.path.join(self.data_dir, item)
                    break
            
            if not telemetry_dir:
                logger.error("Could not find telemetry directory")
                return False
            
            # Load driver telemetry file
            driver_file = os.path.join(telemetry_dir, f"{driver}_telemetry.json")
            if not os.path.exists(driver_file):
                logger.error(f"Driver file not found: {driver_file}")
                return False
            
            with open(driver_file, 'r') as f:
                self.driver_data[driver] = json.load(f)
            
            logger.info(f"Loaded telemetry data for {driver}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load driver data for {driver}: {e}")
            return False
    
    def get_speed_data(self, driver: str) -> Tuple[List[float], List[float]]:
        """
        Extract speed and time data for a driver.
        
        Args:
            driver: Driver abbreviation
            
        Returns:
            Tuple of (time_data, speed_data)
        """
        if driver not in self.driver_data:
            raise ValueError(f"Driver {driver} data not loaded")
        
        driver_data = self.driver_data[driver]
        
        # Get time data (convert to seconds from start of lap)
        time_data = driver_data['telemetry']['Time']
        speed_data = driver_data['telemetry']['Speed']
        
        # Convert time strings to relative seconds
        time_seconds = []
        start_time = None
        
        for time_str in time_data:
            if time_str is None:
                time_seconds.append(None)
                continue
                
            # Parse time string (format: "0 days 00:00:00.123000")
            try:
                time_parts = time_str.split()
                time_part = time_parts[2]  # "00:00:00.123000"
                hours, minutes, seconds = time_part.split(':')
                total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                
                if start_time is None:
                    start_time = total_seconds
                
                time_seconds.append(total_seconds - start_time)
            except:
                time_seconds.append(None)
        
        # Filter out None values and create clean arrays
        clean_time = []
        clean_speed = []
        
        for t, s in zip(time_seconds, speed_data):
            if t is not None and s is not None:
                clean_time.append(t)
                clean_speed.append(s)
        
        return clean_time, clean_speed
    
    def plot_speed_comparison(self, driver1: str, driver2: str, save_path: Optional[str] = None) -> None:
        """
        Create a speed comparison plot between two drivers.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            save_path: Optional path to save the plot
        """
        # Get speed data for both drivers
        time1, speed1 = self.get_speed_data(driver1)
        time2, speed2 = self.get_speed_data(driver2)
        
        # Get driver info
        info1 = self.driver_data[driver1]
        info2 = self.driver_data[driver2]
        
        # Get session info
        session_info = self.session_data['session_info']
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot speed traces
        plt.plot(time1, speed1, label=f"{driver1} ({info1['team']})", linewidth=2, color='#1f77b4')
        plt.plot(time2, speed2, label=f"{driver2} ({info2['team']})", linewidth=2, color='#ff7f0e')
        
        # Customize the plot
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Speed (km/h)', fontsize=12)
        plt.title(f'Speed Comparison: {driver1} vs {driver2}\n'
                 f'{session_info["event"]} - {session_info["session"]}', 
                 fontsize=14, fontweight='bold')
        
        # Add lap time information
        lap_time1 = info1['lap_time_seconds']
        lap_time2 = info2['lap_time_seconds']
        time_diff = lap_time2 - lap_time1
        
        plt.text(0.02, 0.98, f'{driver1}: {info1["lap_time"]} ({lap_time1:.3f}s)', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.text(0.02, 0.92, f'{driver2}: {info2["lap_time"]} ({lap_time2:.3f}s)', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        if time_diff > 0:
            plt.text(0.02, 0.86, f'{driver1} faster by {time_diff:.3f}s', 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            plt.text(0.02, 0.86, f'{driver2} faster by {abs(time_diff):.3f}s', 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Set axis limits
        plt.xlim(0, max(max(time1), max(time2)))
        plt.ylim(0, max(max(speed1), max(speed2)) * 1.05)
        
        # Add speed zones
        plt.axhspan(0, 100, alpha=0.1, color='red', label='Low Speed')
        plt.axhspan(100, 200, alpha=0.1, color='yellow', label='Medium Speed')
        plt.axhspan(200, 350, alpha=0.1, color='green', label='High Speed')
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_speed_with_sectors(self, driver1: str, driver2: str, save_path: Optional[str] = None) -> None:
        """
        Create a speed comparison plot with sector information.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            save_path: Optional path to save the plot
        """
        # Get speed data for both drivers
        time1, speed1 = self.get_speed_data(driver1)
        time2, speed2 = self.get_speed_data(driver2)
        
        # Get driver info
        info1 = self.driver_data[driver1]
        info2 = self.driver_data[driver2]
        
        # Get session info
        session_info = self.session_data['session_info']
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        
        # Plot speed traces
        ax1.plot(time1, speed1, label=f"{driver1} ({info1['team']})", linewidth=2, color='#1f77b4')
        ax1.plot(time2, speed2, label=f"{driver2} ({info2['team']})", linewidth=2, color='#ff7f0e')
        
        # Add sector information
        sector_times1 = info1['sector_times']
        sector_times2 = info2['sector_times']
        
        # Calculate sector boundaries (approximate)
        total_time1 = info1['lap_time_seconds']
        total_time2 = info2['lap_time_seconds']
        
        # Add sector lines - calculate cumulative sector boundaries
        cumulative_time1 = 0
        cumulative_time2 = 0
        
        for i, (s1, s2) in enumerate(zip(sector_times1, sector_times2)):
            if s1 and s2:
                # Parse sector times (these are individual sector durations)
                try:
                    s1_seconds = float(s1.split(':')[-1].split('.')[0]) + float(s1.split('.')[-1]) / 1000000
                    s2_seconds = float(s2.split(':')[-1].split('.')[0]) + float(s2.split('.')[-1]) / 1000000
                    
                    # Calculate cumulative sector boundaries
                    cumulative_time1 += s1_seconds
                    cumulative_time2 += s2_seconds
                    
                    # Use average of both drivers' cumulative times for sector boundary
                    avg_sector_boundary = (cumulative_time1 + cumulative_time2) / 2
                    
                    # Add vertical line for sector boundary
                    ax1.axvline(x=avg_sector_boundary, color='gray', linestyle='--', alpha=0.7, linewidth=1)
                    
                    # Add sector labels
                    ax1.text(avg_sector_boundary, max(max(speed1), max(speed2)) * 0.95, f'S{i+1}', 
                            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                except:
                    pass
        
        # Customize the first subplot
        ax1.set_ylabel('Speed (km/h)', fontsize=12)
        ax1.set_title(f'Speed Comparison: {driver1} vs {driver2}\n'
                     f'{session_info["event"]} - {session_info["session"]}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.set_ylim(0, max(max(speed1), max(speed2)) * 1.05)
        
        # Add speed zones
        ax1.axhspan(0, 100, alpha=0.1, color='red')
        ax1.axhspan(100, 200, alpha=0.1, color='yellow')
        ax1.axhspan(200, 350, alpha=0.1, color='green')
        
        # Create delta plot
        # Interpolate speeds to common time grid
        common_time = np.linspace(0, min(max(time1), max(time2)), 1000)
        speed1_interp = np.interp(common_time, time1, speed1)
        speed2_interp = np.interp(common_time, time2, speed2)
        
        # Calculate delta (driver1 - driver2)
        delta = speed1_interp - speed2_interp
        
        ax2.plot(common_time, delta, linewidth=2, color='purple', label=f'{driver1} - {driver2}')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(common_time, delta, 0, where=(delta > 0), alpha=0.3, color='blue', label=f'{driver1} faster')
        ax2.fill_between(common_time, delta, 0, where=(delta < 0), alpha=0.3, color='orange', label=f'{driver2} faster')
        
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Speed Delta (km/h)', fontsize=12)
        ax2.set_title('Speed Difference', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()


def main():
    """Main function to run the speed comparison tool."""
    parser = argparse.ArgumentParser(description='F1 Speed Comparison Visualization')
    parser.add_argument('--data-dir', type=str, default='./f1_data_comprehensive', 
                       help='Directory containing F1 telemetry data')
    parser.add_argument('--driver1', type=str, required=True, help='First driver abbreviation (e.g., VER)')
    parser.add_argument('--driver2', type=str, required=True, help='Second driver abbreviation (e.g., HAM)')
    parser.add_argument('--save', type=str, help='Path to save the plot (optional)')
    parser.add_argument('--sectors', action='store_true', help='Include sector information')
    
    args = parser.parse_args()
    
    # Initialize the comparison tool
    comparison = F1SpeedComparison(args.data_dir)
    
    # Find the session file
    session_file = None
    for file in os.listdir(args.data_dir):
        if file.endswith('.json') and not file.endswith('_telemetry'):
            session_file = os.path.join(args.data_dir, file)
            break
    
    if not session_file:
        logger.error("Could not find session JSON file")
        return
    
    # Load session data
    if not comparison.load_session_data(session_file):
        logger.error("Failed to load session data")
        return
    
    # Load driver data
    if not comparison.load_driver_data(args.driver1):
        logger.error(f"Failed to load data for {args.driver1}")
        return
    
    if not comparison.load_driver_data(args.driver2):
        logger.error(f"Failed to load data for {args.driver2}")
        return
    
    # Create the comparison plot
    try:
        if args.sectors:
            comparison.plot_speed_with_sectors(args.driver1, args.driver2, args.save)
        else:
            comparison.plot_speed_comparison(args.driver1, args.driver2, args.save)
        
        logger.info("Speed comparison plot created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        raise


if __name__ == "__main__":
    main()

"""
F1 Telemetry Comparison Tool (Distance-Based)
============================================

This script creates comprehensive telemetry comparisons between two drivers using
distance as a percentage of total lap distance instead of time.

Usage:
    python telemetry_comparison_distance.py --data-dir "./f1_data_comprehensive" --driver1 VER --driver2 HAM --channel RPM
    python telemetry_comparison_distance.py --data-dir "./f1_data_comprehensive" --driver1 VER --driver2 HAM --channel Throttle --sectors
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

class F1TelemetryComparisonDistance:
    """F1 telemetry comparison visualization tool using distance-based analysis."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the telemetry comparison tool.
        
        Args:
            data_dir: Directory containing the F1 telemetry data
        """
        self.data_dir = data_dir
        self.session_data = None
        self.driver_data = {}
        
        # Define available telemetry channels and their properties
        self.telemetry_channels = {
            'Speed': {'unit': 'km/h', 'color': '#1f77b4', 'description': 'Vehicle Speed'},
            'RPM': {'unit': 'rpm', 'color': '#ff7f0e', 'description': 'Engine RPM'},
            'Throttle': {'unit': '%', 'color': '#2ca02c', 'description': 'Throttle Position'},
            'Brake': {'unit': 'on/off', 'color': '#d62728', 'description': 'Brake Application'},
            'nGear': {'unit': 'gear', 'color': '#9467bd', 'description': 'Current Gear'},
            'DRS': {'unit': 'on/off', 'color': '#8c564b', 'description': 'DRS Status'}
        }
        
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
    
    def calculate_distance_percentage(self, driver: str) -> List[float]:
        """
        Calculate distance as percentage of THIS DRIVER'S total lap distance.
        
        Args:
            driver: Driver abbreviation
            
        Returns:
            List of distance percentages (0-100%) for this driver's lap
        """
        if driver not in self.driver_data:
            raise ValueError(f"Driver {driver} data not loaded")
        
        driver_data = self.driver_data[driver]
        
        # Try to get distance data from position data first
        distance_data = None
        if 'position_data' in driver_data and 'X' in driver_data['position_data']:
            # Calculate distance from X, Y coordinates
            x_coords = driver_data['position_data']['X']
            y_coords = driver_data['position_data']['Y']
            
            if x_coords and y_coords:
                # Calculate cumulative distance from coordinate changes
                distances = [0.0]  # Start at 0
                for i in range(1, len(x_coords)):
                    if (x_coords[i] is not None and y_coords[i] is not None and 
                        x_coords[i-1] is not None and y_coords[i-1] is not None):
                        dx = x_coords[i] - x_coords[i-1]
                        dy = y_coords[i] - y_coords[i-1]
                        distance = (dx**2 + dy**2)**0.5
                        distances.append(distances[-1] + distance)
                    else:
                        distances.append(distances[-1])  # Keep previous distance
                
                distance_data = distances
        
        # If no position data, try to get distance from telemetry
        if distance_data is None and 'Distance' in driver_data['telemetry']:
            distance_data = driver_data['telemetry']['Distance']
        
        # If still no distance data, estimate from time
        if distance_data is None:
            # Estimate distance based on time and average speed
            time_data = driver_data['telemetry']['Time']
            speed_data = driver_data['telemetry']['Speed']
            
            # Convert time to seconds and calculate estimated distance
            time_seconds = []
            start_time = None
            
            for time_str in time_data:
                if time_str is None:
                    time_seconds.append(None)
                    continue
                    
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
            
            # Calculate estimated distance from speed and time
            distances = [0.0]
            for i in range(1, len(time_seconds)):
                if (time_seconds[i] is not None and speed_data[i] is not None and 
                    time_seconds[i-1] is not None and speed_data[i-1] is not None):
                    dt = time_seconds[i] - time_seconds[i-1]
                    avg_speed = (speed_data[i] + speed_data[i-1]) / 2  # km/h
                    distance_increment = avg_speed * dt / 3.6  # Convert to m/s and multiply by time
                    distances.append(distances[-1] + distance_increment)
                else:
                    distances.append(distances[-1])
            
            distance_data = distances
        
        if not distance_data:
            raise ValueError(f"No distance data available for {driver}")
        
        # Convert to percentage of THIS DRIVER'S total distance
        # Each driver should go from 0% to 100% of their own lap
        total_distance = max(distance_data)
        if total_distance == 0:
            return [0.0] * len(distance_data)
        
        # Calculate percentage of this driver's total lap distance
        distance_percentages = [(d / total_distance) * 100 for d in distance_data]
        return distance_percentages
    
    def get_telemetry_data(self, driver: str, channel: str) -> Tuple[List[float], List[float]]:
        """
        Extract telemetry data for a specific channel and driver using distance percentage.
        
        Args:
            driver: Driver abbreviation
            channel: Telemetry channel name (e.g., 'Speed', 'RPM', 'Throttle')
            
        Returns:
            Tuple of (distance_percentage_data, telemetry_data)
        """
        if driver not in self.driver_data:
            raise ValueError(f"Driver {driver} data not loaded")
        
        driver_data = self.driver_data[driver]
        
        # Get telemetry data
        telemetry_data = driver_data['telemetry'][channel]
        
        # Calculate distance percentages
        distance_percentages = self.calculate_distance_percentage(driver)
        
        # Filter out None values and create clean arrays
        clean_distance = []
        clean_telemetry = []
        
        for i, (dist, data) in enumerate(zip(distance_percentages, telemetry_data)):
            if dist is not None and data is not None:
                clean_distance.append(dist)
                clean_telemetry.append(data)
        
        return clean_distance, clean_telemetry
    
    def plot_telemetry_comparison(self, driver1: str, driver2: str, channel: str, save_path: Optional[str] = None) -> None:
        """
        Create a telemetry comparison plot between two drivers using distance percentage.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            channel: Telemetry channel to compare
            save_path: Optional path to save the plot
        """
        # Get telemetry data for both drivers
        distance1, data1 = self.get_telemetry_data(driver1, channel)
        distance2, data2 = self.get_telemetry_data(driver2, channel)
        
        # Get driver info
        info1 = self.driver_data[driver1]
        info2 = self.driver_data[driver2]
        session_info = self.session_data['session_info']
        
        # Get channel properties
        channel_info = self.telemetry_channels.get(channel, {'unit': '', 'color': '#1f77b4', 'description': channel})
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot telemetry traces
        plt.plot(distance1, data1, label=f"{driver1} ({info1['team']})", linewidth=2, color='#1f77b4')
        plt.plot(distance2, data2, label=f"{driver2} ({info2['team']})", linewidth=2, color='#ff7f0e')
        
        # Customize the plot
        plt.xlabel('Distance (% of lap)', fontsize=12)
        plt.ylabel(f'{channel} ({channel_info["unit"]})', fontsize=12)
        plt.title(f'{channel} Comparison: {driver1} vs {driver2}\n'
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
        
        # Add channel-specific information
        if channel == 'Speed':
            # Add speed zones
            plt.axhspan(0, 100, alpha=0.1, color='red', label='Low Speed')
            plt.axhspan(100, 200, alpha=0.1, color='yellow', label='Medium Speed')
            plt.axhspan(200, 350, alpha=0.1, color='green', label='High Speed')
        elif channel == 'RPM':
            # Add RPM zones
            plt.axhspan(0, 5000, alpha=0.1, color='red', label='Low RPM')
            plt.axhspan(5000, 10000, alpha=0.1, color='yellow', label='Medium RPM')
            plt.axhspan(10000, 15000, alpha=0.1, color='green', label='High RPM')
        elif channel == 'Throttle':
            # Add throttle zones
            plt.axhspan(0, 25, alpha=0.1, color='red', label='Low Throttle')
            plt.axhspan(25, 75, alpha=0.1, color='yellow', label='Medium Throttle')
            plt.axhspan(75, 100, alpha=0.1, color='green', label='High Throttle')
        
        # Add distance-based sector markers (approximate)
        plt.axvline(x=33.33, color='gray', linestyle='--', alpha=0.5, label='Sector 1/3')
        plt.axvline(x=66.67, color='gray', linestyle='--', alpha=0.5, label='Sector 2/3')
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Set axis limits
        plt.xlim(0, 100)
        if channel in ['Speed', 'RPM', 'Throttle']:
            plt.ylim(0, max(max(data1), max(data2)) * 1.05)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_telemetry_with_sectors(self, driver1: str, driver2: str, channel: str, save_path: Optional[str] = None) -> None:
        """
        Create a telemetry comparison plot with sector information using distance percentage.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            channel: Telemetry channel to compare
            save_path: Optional path to save the plot
        """
        # Get telemetry data for both drivers
        distance1, data1 = self.get_telemetry_data(driver1, channel)
        distance2, data2 = self.get_telemetry_data(driver2, channel)
        
        # Get driver info
        info1 = self.driver_data[driver1]
        info2 = self.driver_data[driver2]
        session_info = self.session_data['session_info']
        
        # Get channel properties
        channel_info = self.telemetry_channels.get(channel, {'unit': '', 'color': '#1f77b4', 'description': channel})
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        
        # Plot telemetry traces
        ax1.plot(distance1, data1, label=f"{driver1} ({info1['team']})", linewidth=2, color='#1f77b4')
        ax1.plot(distance2, data2, label=f"{driver2} ({info2['team']})", linewidth=2, color='#ff7f0e')
        
        # Add sector information based on distance percentage
        # Calculate sector boundaries as percentages of total distance
        sector_times1 = info1['sector_times']
        sector_times2 = info2['sector_times']
        
        # Calculate cumulative sector boundaries
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
                    
                    # Convert to distance percentage (approximate)
                    total_time1 = info1['lap_time_seconds']
                    total_time2 = info2['lap_time_seconds']
                    
                    # Use average of both drivers' sector boundaries
                    sector_percentage1 = (cumulative_time1 / total_time1) * 100
                    sector_percentage2 = (cumulative_time2 / total_time2) * 100
                    avg_sector_percentage = (sector_percentage1 + sector_percentage2) / 2
                    
                    # Add vertical line for sector boundary
                    ax1.axvline(x=avg_sector_percentage, color='gray', linestyle='--', alpha=0.7, linewidth=1)
                    
                    # Add sector labels
                    ax1.text(avg_sector_percentage, max(max(data1), max(data2)) * 0.95, f'S{i+1}', 
                            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                except:
                    pass
        
        # Customize the first subplot
        ax1.set_ylabel(f'{channel} ({channel_info["unit"]})', fontsize=12)
        ax1.set_title(f'{channel} Comparison: {driver1} vs {driver2}\n'
                     f'{session_info["event"]} - {session_info["session"]}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        if channel in ['Speed', 'RPM', 'Throttle']:
            ax1.set_ylim(0, max(max(data1), max(data2)) * 1.05)
        
        # Add channel-specific zones
        if channel == 'Speed':
            ax1.axhspan(0, 100, alpha=0.1, color='red')
            ax1.axhspan(100, 200, alpha=0.1, color='yellow')
            ax1.axhspan(200, 350, alpha=0.1, color='green')
        elif channel == 'RPM':
            ax1.axhspan(0, 5000, alpha=0.1, color='red')
            ax1.axhspan(5000, 10000, alpha=0.1, color='yellow')
            ax1.axhspan(10000, 15000, alpha=0.1, color='green')
        elif channel == 'Throttle':
            ax1.axhspan(0, 25, alpha=0.1, color='red')
            ax1.axhspan(25, 75, alpha=0.1, color='yellow')
            ax1.axhspan(75, 100, alpha=0.1, color='green')
        
        # Create delta plot
        # Interpolate data to common distance grid
        common_distance = np.linspace(0, 100, 1000)
        data1_interp = np.interp(common_distance, distance1, data1)
        data2_interp = np.interp(common_distance, distance2, data2)
        
        # Calculate delta (driver1 - driver2)
        delta = data1_interp - data2_interp
        
        ax2.plot(common_distance, delta, linewidth=2, color='purple', label=f'{driver1} - {driver2}')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(common_distance, delta, 0, where=(delta > 0), alpha=0.3, color='blue', label=f'{driver1} higher')
        ax2.fill_between(common_distance, delta, 0, where=(delta < 0), alpha=0.3, color='orange', label=f'{driver2} higher')
        
        ax2.set_xlabel('Distance (% of lap)', fontsize=12)
        ax2.set_ylabel(f'{channel} Delta ({channel_info["unit"]})', fontsize=12)
        ax2.set_title(f'{channel} Difference', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def list_available_channels(self, driver: str) -> List[str]:
        """
        List available telemetry channels for a driver.
        
        Args:
            driver: Driver abbreviation
            
        Returns:
            List of available channel names
        """
        if driver not in self.driver_data:
            raise ValueError(f"Driver {driver} data not loaded")
        
        return list(self.driver_data[driver]['telemetry'].keys())
    
    def plot_multiple_channels(self, driver1: str, driver2: str, channels: List[str], save_path: Optional[str] = None) -> None:
        """
        Create a multi-panel plot comparing multiple telemetry channels using distance percentage.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            channels: List of telemetry channels to compare
            save_path: Optional path to save the plot
        """
        n_channels = len(channels)
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 4 * n_channels), sharex=True)
        
        if n_channels == 1:
            axes = [axes]
        
        for i, channel in enumerate(channels):
            # Get telemetry data for both drivers
            distance1, data1 = self.get_telemetry_data(driver1, channel)
            distance2, data2 = self.get_telemetry_data(driver2, channel)
            
            # Get channel properties
            channel_info = self.telemetry_channels.get(channel, {'unit': '', 'color': '#1f77b4', 'description': channel})
            
            # Plot telemetry traces
            axes[i].plot(distance1, data1, label=f"{driver1}", linewidth=2, color='#1f77b4')
            axes[i].plot(distance2, data2, label=f"{driver2}", linewidth=2, color='#ff7f0e')
            
            # Add distance-based sector markers
            axes[i].axvline(x=33.33, color='gray', linestyle='--', alpha=0.3)
            axes[i].axvline(x=66.67, color='gray', linestyle='--', alpha=0.3)
            
            # Customize the subplot
            axes[i].set_ylabel(f'{channel} ({channel_info["unit"]})', fontsize=10)
            axes[i].set_title(f'{channel} Comparison', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=9)
            
            # Set axis limits
            axes[i].set_xlim(0, 100)
            if channel in ['Speed', 'RPM', 'Throttle']:
                axes[i].set_ylim(0, max(max(data1), max(data2)) * 1.05)
        
        # Set x-axis label only on the bottom subplot
        axes[-1].set_xlabel('Distance (% of lap)', fontsize=12)
        
        # Add main title
        fig.suptitle(f'Telemetry Comparison: {driver1} vs {driver2}\n'
                    f'{self.session_data["session_info"]["event"]} - {self.session_data["session_info"]["session"]}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()


def main():
    """Main function to run the telemetry comparison tool."""
    parser = argparse.ArgumentParser(description='F1 Telemetry Comparison Visualization (Distance-Based)')
    parser.add_argument('--data-dir', type=str, default='./f1_data_comprehensive', 
                       help='Directory containing F1 telemetry data')
    parser.add_argument('--driver1', type=str, required=True, help='First driver abbreviation (e.g., VER)')
    parser.add_argument('--driver2', type=str, help='Second driver abbreviation (e.g., HAM)')
    parser.add_argument('--channel', type=str, default='Speed', 
                       help='Telemetry channel to compare (Speed, RPM, Throttle, Brake, nGear, DRS)')
    parser.add_argument('--channels', type=str, nargs='+', 
                       help='Multiple channels to compare (e.g., Speed RPM Throttle)')
    parser.add_argument('--save', type=str, help='Path to save the plot (optional)')
    parser.add_argument('--sectors', action='store_true', help='Include sector information')
    parser.add_argument('--list-channels', action='store_true', help='List available channels for driver1')
    
    args = parser.parse_args()
    
    # Initialize the comparison tool
    comparison = F1TelemetryComparisonDistance(args.data_dir)
    
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
    
    # List available channels if requested
    if args.list_channels:
        channels = comparison.list_available_channels(args.driver1)
        print(f"Available telemetry channels for {args.driver1}:")
        for channel in channels:
            channel_info = comparison.telemetry_channels.get(channel, {'description': channel})
            print(f"  - {channel}: {channel_info['description']}")
        return
    
    # Check if driver2 is provided for comparisons
    if not args.driver2:
        logger.error("Driver2 is required for telemetry comparisons")
        return
    
    if not comparison.load_driver_data(args.driver2):
        logger.error(f"Failed to load data for {args.driver2}")
        return
    
    # Create the comparison plot
    try:
        if args.channels:
            # Multiple channels comparison
            comparison.plot_multiple_channels(args.driver1, args.driver2, args.channels, args.save)
        elif args.sectors:
            # Single channel with sectors
            comparison.plot_telemetry_with_sectors(args.driver1, args.driver2, args.channel, args.save)
        else:
            # Single channel basic comparison
            comparison.plot_telemetry_comparison(args.driver1, args.driver2, args.channel, args.save)
        
        logger.info("Telemetry comparison plot created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        raise


if __name__ == "__main__":
    main()

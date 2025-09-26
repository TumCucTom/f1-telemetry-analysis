"""
F1 Telemetry Comparison Tool
============================

This script creates comprehensive telemetry comparisons between two drivers using
all available telemetry channels from the F1 data scraper.

Usage:
    python telemetry_comparison.py --data-dir "./f1_data_comprehensive" --driver1 VER --driver2 HAM --channel RPM
    python telemetry_comparison.py --data-dir "./f1_data_comprehensive" --driver1 VER --driver2 HAM --channel Throttle --sectors
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import argparse
import os
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1TelemetryComparison:
    """F1 telemetry comparison visualization tool."""
    
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
            'DRS': {'unit': 'on/off', 'color': '#8c564b', 'description': 'DRS Status'},
            'TimeDelta': {'unit': 's', 'color': '#e377c2', 'description': 'Time Delta (vs Reference)'}
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
    
    def get_telemetry_data(self, driver: str, channel: str, time_scaled: bool = False) -> Tuple[List[float], List[float]]:
        """
        Extract telemetry data for a specific channel and driver.
        
        Args:
            driver: Driver abbreviation
            channel: Telemetry channel name (e.g., 'Speed', 'RPM', 'Throttle')
            time_scaled: If True, return time as percentage of lap time (0-100%)
            
        Returns:
            Tuple of (time_data, telemetry_data)
        """
        if driver not in self.driver_data:
            raise ValueError(f"Driver {driver} data not loaded")
        
        driver_data = self.driver_data[driver]
        
        # Get time data (convert to seconds from start of lap)
        time_data = driver_data['telemetry']['Time']
        telemetry_data = driver_data['telemetry'][channel]
        
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
        clean_telemetry = []
        
        for t, data in zip(time_seconds, telemetry_data):
            if t is not None and data is not None:
                clean_time.append(t)
                clean_telemetry.append(data)
        
        # If time_scaled is True, convert to percentage of lap time
        if time_scaled and clean_time:
            total_lap_time = driver_data['lap_time_seconds']
            if total_lap_time > 0:
                clean_time = [(t / total_lap_time) * 100 for t in clean_time]
        
        return clean_time, clean_telemetry

    def animate_channel(self, driver1: str, driver2: str, channel: str, video_path: str, time_scaled: bool = False, duration_sec: float = 28.0) -> None:
        """
        Animate a single channel over time, progressively revealing the trace(s) and export to MP4.
        For TimeDelta, animate the orange delta line with zero baseline. For other channels, animate
        both drivers (grey first then orange on top).
        """
        # Get telemetry data
        if channel == 'TimeDelta':
            x, delta = self.calculate_time_delta(driver1, driver2, time_scaled)
            # Interpolate to a fixed grid for smooth animation
            n_points = 600
            x_grid = np.linspace(min(x), max(x), n_points)
            delta_grid = np.interp(x_grid, x, delta)
        else:
            x1, y1 = self.get_telemetry_data(driver1, channel, time_scaled)
            x2, y2 = self.get_telemetry_data(driver2, channel, time_scaled)
            # Build common grid
            n_points = 600
            x_min = max(min(x1), min(x2))
            x_max = min(max(x1), max(x2))
            x_grid = np.linspace(x_min, x_max, n_points)
            y1 = np.interp(x_grid, x1, y1)
            y2 = np.interp(x_grid, x2, y2)

        # Prepare figure (halve vertical height for Speed and TimeDelta)
        fig_height = 4 if channel in ['TimeDelta', 'Speed'] else 8
        fig, ax = plt.subplots(figsize=(15, fig_height))
        ax.set_xlabel('' if time_scaled else '')
        ax.set_xticks([])
        ax.grid(False)

        # Remove box for Throttle, nGear, Brake (match static styling)
        if channel in ['Throttle', 'nGear', 'Brake']:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # Set titles
        title_suffix = ' (Time-Scaled)' if time_scaled else ''
        ax.set_title(f'{channel} Animation: {driver1} vs {driver2}{title_suffix}\n'
                     f"{self.session_data['session_info']['event']} - {self.session_data['session_info']['session']}",
                     fontsize=14, fontweight='bold')

        # Plot elements
        if channel == 'TimeDelta':
            # Corner lines
            corner_positions = self.get_corner_positions(time_scaled)
            for corner_pos, _corner_name in corner_positions:
                ax.axvline(x=corner_pos, color='red', linestyle=':', alpha=0.7, linewidth=1)

            zero_line = ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            line_orange, = ax.plot([], [], color='#FF8C00', linewidth=2)
            y_min = float(np.min(delta_grid))
            y_max = float(np.max(delta_grid))
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0
            ax.set_ylim(y_min * 1.05, y_max * 1.05)
            ax.set_xlim(x_grid[0], x_grid[-1])

            # For colored zones, we will (re)draw fills per frame
            fill_artists = []

            def init():
                # Clear any fills
                for art in fill_artists:
                    art.remove()
                fill_artists.clear()
                line_orange.set_data([], [])
                return (line_orange,)

            def update(i):
                xi = x_grid[:i+1]
                yi = delta_grid[:i+1]
                line_orange.set_data(xi, yi)
                # Remove previous fills
                for art in fill_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                fill_artists.clear()
                # Create masks
                pos_mask = yi > 0
                neg_mask = yi < 0
                if np.any(pos_mask):
                    fp = ax.fill_between(xi, yi, 0, where=pos_mask, color='#FF8C00', alpha=0.3)
                    fill_artists.append(fp)
                if np.any(neg_mask):
                    fn = ax.fill_between(xi, yi, 0, where=neg_mask, color='#404040', alpha=0.3)
                    fill_artists.append(fn)
                return (line_orange, *fill_artists)
        else:
            # Grey first (driver2), orange on top (driver1)
            # Corner lines if Speed
            if channel == 'Speed':
                corner_positions = self.get_corner_positions(time_scaled)
                for corner_pos, _corner_name in corner_positions:
                    ax.axvline(x=corner_pos, color='red', linestyle=':', alpha=0.7, linewidth=1)
            line_grey, = ax.plot([], [], color='#404040', linewidth=2, label=driver2)
            line_orange, = ax.plot([], [], color='#FF8C00', linewidth=2, label=driver1)
            # Y limits
            y_all = np.concatenate([y1, y2])
            y_min = float(np.min(y_all))
            y_max = float(np.max(y_all))
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0
            if channel in ['Speed', 'RPM', 'Throttle']:
                y_min = 0.0
            ax.set_ylim(y_min, y_max * 1.05)
            ax.set_xlim(x_grid[0], x_grid[-1])

            def init():
                line_grey.set_data([], [])
                line_orange.set_data([], [])
                return (line_grey, line_orange)

            def update(i):
                xi = x_grid[:i+1]
                line_grey.set_data(xi, y2[:i+1])
                line_orange.set_data(xi, y1[:i+1])
                return (line_grey, line_orange)

        # Animation timing
        duration_sec = max(10.0, float(duration_sec))
        interval_ms = (duration_sec * 1000.0) / n_points
        fps = max(1, int(round(1000.0 / interval_ms)))

        anim = animation.FuncAnimation(fig, update, init_func=init, frames=n_points, interval=interval_ms, blit=True, repeat=False)

        # Save video
        os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
        writer = animation.FFMpegWriter(fps=fps, metadata={'artist': 'F1 Telemetry Analysis'})
        anim.save(video_path, writer=writer, dpi=200)
        plt.close(fig)
        logger.info(f"Channel animation saved: {video_path} (~{duration_sec:.1f}s, fps {fps})")
    
    def calculate_time_delta(self, driver1: str, driver2: str, time_scaled: bool = False) -> Tuple[List[float], List[float]]:
        """
        Calculate time delta between two drivers based on distance percentage (driver1 - driver2).
        
        Args:
            driver1: First driver abbreviation (reference)
            driver2: Second driver abbreviation
            time_scaled: If True, use time as percentage of lap time (0-100%)
            
        Returns:
            Tuple of (time_data, delta_data)
        """
        # Get position data for both drivers
        x1 = self.driver_data[driver1]['position_data']['X']
        y1 = self.driver_data[driver1]['position_data']['Y']
        x2 = self.driver_data[driver2]['position_data']['X']
        y2 = self.driver_data[driver2]['position_data']['Y']
        
        # Get time data for both drivers
        time1 = self.driver_data[driver1]['telemetry']['Time']
        time2 = self.driver_data[driver2]['telemetry']['Time']
        
        # Convert time strings to relative seconds
        def parse_time_to_seconds(time_data):
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
            
            return time_seconds
        
        time1_seconds = parse_time_to_seconds(time1)
        time2_seconds = parse_time_to_seconds(time2)
        
        # Calculate distance percentage for each driver
        def calculate_distance_percentage(x_coords, y_coords):
            if not x_coords or not y_coords:
                return []
            
            # Calculate cumulative distance
            distances = [0.0]
            for i in range(1, len(x_coords)):
                if (x_coords[i] is not None and y_coords[i] is not None and 
                    x_coords[i-1] is not None and y_coords[i-1] is not None):
                    dx = x_coords[i] - x_coords[i-1]
                    dy = y_coords[i] - y_coords[i-1]
                    distance = (dx**2 + dy**2)**0.5
                    distances.append(distances[-1] + distance)
                else:
                    distances.append(distances[-1])
            
            # Convert to percentage
            total_distance = max(distances)
            if total_distance == 0:
                return [0.0] * len(distances)
            
            return [(d / total_distance) * 100 for d in distances]
        
        # Calculate distance percentages
        dist_pct1 = calculate_distance_percentage(x1, y1)
        dist_pct2 = calculate_distance_percentage(x2, y2)
        
        # Ensure all arrays have the same length
        min_length = min(len(x1), len(y1), len(time1_seconds), len(dist_pct1))
        x1 = x1[:min_length]
        y1 = y1[:min_length]
        time1_seconds = time1_seconds[:min_length]
        dist_pct1 = dist_pct1[:min_length]
        
        min_length2 = min(len(x2), len(y2), len(time2_seconds), len(dist_pct2))
        x2 = x2[:min_length2]
        y2 = y2[:min_length2]
        time2_seconds = time2_seconds[:min_length2]
        dist_pct2 = dist_pct2[:min_length2]
        
        # Create common distance percentage grid (0-100%)
        import numpy as np
        common_dist_pct = np.linspace(0, 100, 1000)
        
        # Interpolate times to common distance percentage grid
        time1_interp = np.interp(common_dist_pct, dist_pct1, 
                                [t if t is not None else 0 for t in time1_seconds])
        time2_interp = np.interp(common_dist_pct, dist_pct2, 
                                [t if t is not None else 0 for t in time2_seconds])
        
        # Calculate time deltas at each distance percentage
        time_deltas = time1_interp - time2_interp
        
        # Convert to time-scaled if requested
        if time_scaled:
            # Use distance percentage as time percentage for consistency
            time_data = common_dist_pct.tolist()
        else:
            # Use interpolated time data
            time_data = time1_interp.tolist()
        
        return time_data, time_deltas.tolist()
    
    def get_corner_positions(self, time_scaled: bool = False) -> List[Tuple[float, str]]:
        """
        Get corner positions as percentages of lap time or distance.
        
        Args:
            time_scaled: If True, return as percentage of lap time, else as percentage of distance
            
        Returns:
            List of (position_percentage, corner_name) tuples
        """
        if not self.session_data or 'session_data' not in self.session_data:
            return []
        
        session_data = self.session_data['session_data']
        if 'circuit_info' not in session_data or 'corners' not in session_data['circuit_info']:
            return []
        
        circuit_info = session_data['circuit_info']
        if not circuit_info['corners']:
            return []
        
        corners = circuit_info['corners']
        corner_positions = []
        
        if corners:
            # Get the total track length from the last corner's distance
            total_distance = corners[-1]['Distance']
            
            # Use a set to track unique corner numbers to avoid duplicates
            seen_corners = set()
            
            for corner in corners:
                if 'Distance' in corner and 'Number' in corner and corner['Distance'] is not None:
                    # Convert distance to percentage of total track length
                    distance_percentage = (corner['Distance'] / total_distance) * 100
                    
                    # Handle duplicate corner numbers by using a counter
                    corner_number = int(corner['Number'])
                    if corner_number in seen_corners:
                        corner_name = f"T{corner_number}a"
                    else:
                        corner_name = f"T{corner_number}"
                        seen_corners.add(corner_number)
                    
                    corner_positions.append((distance_percentage, corner_name))
        
        return corner_positions
    
    def plot_telemetry_comparison(self, driver1: str, driver2: str, channel: str, save_path: Optional[str] = None, time_scaled: bool = False) -> None:
        """
        Create a telemetry comparison plot between two drivers.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            channel: Telemetry channel to compare
            save_path: Optional path to save the plot
            time_scaled: If True, use time as percentage of lap time (0-100%)
        """
        # Get telemetry data for both drivers
        if channel == 'TimeDelta':
            time1, data1 = self.calculate_time_delta(driver1, driver2, time_scaled)
            time2, data2 = time1, data1  # TimeDelta is a single line
        else:
            time1, data1 = self.get_telemetry_data(driver1, channel, time_scaled)
            time2, data2 = self.get_telemetry_data(driver2, channel, time_scaled)
        
        # Get driver info
        info1 = self.driver_data[driver1]
        info2 = self.driver_data[driver2]
        session_info = self.session_data['session_info']
        
        # Get channel properties
        channel_info = self.telemetry_channels.get(channel, {'unit': '', 'color': '#1f77b4', 'description': channel})
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot telemetry traces
        if channel == 'TimeDelta':
            # TimeDelta shows continuous line: + = pole sitter ahead, - = second driver ahead
            plt.plot(time1, data1, linewidth=2, color='#FF8C00')  # Orange line
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.fill_between(time1, data1, 0, where=(np.array(data1) > 0), alpha=0.3, color='#FF8C00', label=f'{driver1} ahead')  # Vibrant orange
            plt.fill_between(time1, data1, 0, where=(np.array(data1) < 0), alpha=0.3, color='#404040', label=f'{driver2} ahead')  # Dark grey
        else:
            plt.plot(time2, data2, label=f"{driver2} ({info2['team']})", linewidth=2, color='#404040')  # Dark grey for Piastri (drawn first)
            plt.plot(time1, data1, label=f"{driver1} ({info1['team']})", linewidth=2, color='#FF8C00')  # Orange for Norris (drawn on top)
        
        # Customize the plot
        plt.xlabel('', fontsize=12)  # Remove x-axis label
        plt.ylabel(f'{channel} ({channel_info["unit"]})', fontsize=12)
        title_suffix = ' (Time-Scaled)' if time_scaled else ''
        plt.title(f'{channel} Comparison: {driver1} vs {driver2}{title_suffix}\n'
                 f'{session_info["event"]} - {session_info["session"]}', 
                 fontsize=14, fontweight='bold')
        
        # Remove x-axis tick labels
        plt.xticks([])
        
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
        
        # Channel-specific zones removed for cleaner look
        
        # Add corner markers (skip for Throttle, nGear, and Brake)
        if channel not in ['Throttle', 'nGear', 'Brake']:
            corner_positions = self.get_corner_positions(time_scaled)
            if corner_positions:
                # Get the y-axis range for positioning labels
                y_min, y_max = plt.ylim()
                y_range = y_max - y_min
                label_y = y_min - (y_range * 0.05)  # Position labels below the plot
                
                for corner_pos, corner_name in corner_positions:
                    plt.axvline(x=corner_pos, color='red', linestyle=':', alpha=0.7, linewidth=1)
                    plt.text(corner_pos, label_y, corner_name, 
                            fontsize=9, ha='center', va='top', rotation=0,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8, edgecolor='red'))
        
        # Remove box around graph for Throttle, nGear, and Brake
        if channel in ['Throttle', 'nGear', 'Brake']:
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        # Add legend (grid removed)
        plt.legend(fontsize=11)
        
        # Set axis limits
        if time_scaled:
            plt.xlim(0, 100)  # 0-100% of lap time
        else:
            plt.xlim(0, max(max(time1), max(time2)))
        if channel in ['Speed', 'RPM', 'Throttle']:
            plt.ylim(0, max(max(data1), max(data2)) * 1.05)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_telemetry_with_sectors(self, driver1: str, driver2: str, channel: str, save_path: Optional[str] = None, time_scaled: bool = False) -> None:
        """
        Create a telemetry comparison plot with sector information.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            channel: Telemetry channel to compare
            save_path: Optional path to save the plot
            time_scaled: If True, use time as percentage of lap time (0-100%)
        """
        # Get telemetry data for both drivers
        if channel == 'TimeDelta':
            time1, data1 = self.calculate_time_delta(driver1, driver2, time_scaled)
            time2, data2 = time1, data1  # TimeDelta is a single line
        else:
            time1, data1 = self.get_telemetry_data(driver1, channel, time_scaled)
            time2, data2 = self.get_telemetry_data(driver2, channel, time_scaled)
        
        # Get driver info
        info1 = self.driver_data[driver1]
        info2 = self.driver_data[driver2]
        session_info = self.session_data['session_info']
        
        # Get channel properties
        channel_info = self.telemetry_channels.get(channel, {'unit': '', 'color': '#1f77b4', 'description': channel})
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        
        # Plot telemetry traces
        if channel == 'TimeDelta':
            # TimeDelta shows the difference between drivers
            ax1.plot(time1, data1, label=f"{driver1} - {driver2}", linewidth=2, color='#e377c2')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax1.fill_between(time1, data1, 0, where=(np.array(data1) > 0), alpha=0.3, color='blue', label=f'{driver1} ahead')
            ax1.fill_between(time1, data1, 0, where=(np.array(data1) < 0), alpha=0.3, color='orange', label=f'{driver2} ahead')
        else:
            ax1.plot(time1, data1, label=f"{driver1} ({info1['team']})", linewidth=2, color='#1f77b4')
            ax1.plot(time2, data2, label=f"{driver2} ({info2['team']})", linewidth=2, color='#ff7f0e')
        
        # Add sector information
        sector_times1 = info1['sector_times']
        sector_times2 = info2['sector_times']
        
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
                    ax1.text(avg_sector_boundary, max(max(data1), max(data2)) * 0.95, f'S{i+1}', 
                            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                except:
                    pass
        
        # Add corner markers to the first subplot
        corner_positions = self.get_corner_positions(time_scaled)
        if corner_positions:
            # Get the y-axis range for positioning labels
            y_min, y_max = ax1.get_ylim()
            y_range = y_max - y_min
            label_y = y_min - (y_range * 0.05)  # Position labels below the plot
            
            for corner_pos, corner_name in corner_positions:
                ax1.axvline(x=corner_pos, color='red', linestyle=':', alpha=0.7, linewidth=1)
                ax1.text(corner_pos, label_y, corner_name, 
                        fontsize=9, ha='center', va='top', rotation=0,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8, edgecolor='red'))
        
        # Customize the first subplot
        ax1.set_ylabel(f'{channel} ({channel_info["unit"]})', fontsize=12)
        title_suffix = ' (Time-Scaled)' if time_scaled else ''
        ax1.set_title(f'{channel} Comparison: {driver1} vs {driver2}{title_suffix}\n'
                     f'{session_info["event"]} - {session_info["session"]}', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)  # Grid removed
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
        # Interpolate data to common time grid
        if time_scaled:
            common_time = np.linspace(0, 100, 1000)  # 0-100% of lap time
        else:
            common_time = np.linspace(0, min(max(time1), max(time2)), 1000)
        data1_interp = np.interp(common_time, time1, data1)
        data2_interp = np.interp(common_time, time2, data2)
        
        # Calculate delta (driver1 - driver2)
        delta = data1_interp - data2_interp
        
        ax2.plot(common_time, delta, linewidth=2, color='purple', label=f'{driver1} - {driver2}')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(common_time, delta, 0, where=(delta > 0), alpha=0.3, color='blue', label=f'{driver1} higher')
        ax2.fill_between(common_time, delta, 0, where=(delta < 0), alpha=0.3, color='orange', label=f'{driver2} higher')
        
        xlabel = 'Time (% of lap)' if time_scaled else 'Time (seconds)'
        ax2.set_xlabel(xlabel, fontsize=12)
        ax2.set_ylabel(f'{channel} Delta ({channel_info["unit"]})', fontsize=12)
        ax2.set_title(f'{channel} Difference', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)  # Grid removed
        
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
    
    def plot_multiple_channels(self, driver1: str, driver2: str, channels: List[str], save_path: Optional[str] = None, time_scaled: bool = False) -> None:
        """
        Create a multi-panel plot comparing multiple telemetry channels.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            channels: List of telemetry channels to compare
            save_path: Optional path to save the plot
            time_scaled: If True, use time as percentage of lap time (0-100%)
        """
        n_channels = len(channels)
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 4 * n_channels), sharex=True)
        
        if n_channels == 1:
            axes = [axes]
        
        for i, channel in enumerate(channels):
            # Get telemetry data for both drivers
            if channel == 'TimeDelta':
                time1, data1 = self.calculate_time_delta(driver1, driver2, time_scaled)
                time2, data2 = time1, data1  # TimeDelta is a single line
            else:
                time1, data1 = self.get_telemetry_data(driver1, channel, time_scaled)
                time2, data2 = self.get_telemetry_data(driver2, channel, time_scaled)
            
            # Get channel properties
            channel_info = self.telemetry_channels.get(channel, {'unit': '', 'color': '#1f77b4', 'description': channel})
            
            # Plot telemetry traces
            if channel == 'TimeDelta':
                # TimeDelta shows continuous line: + = pole sitter ahead, - = second driver ahead
                axes[i].plot(time1, data1, linewidth=2, color='#FF8C00')  # Orange line
                axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[i].fill_between(time1, data1, 0, where=(np.array(data1) > 0), alpha=0.3, color='#FF8C00', label=f'{driver1} ahead')  # Vibrant orange
                axes[i].fill_between(time1, data1, 0, where=(np.array(data1) < 0), alpha=0.3, color='#404040', label=f'{driver2} ahead')  # Dark grey
            else:
                axes[i].plot(time2, data2, label=f"{driver2}", linewidth=2, color='#404040')  # Dark grey for Piastri (drawn first)
                axes[i].plot(time1, data1, label=f"{driver1}", linewidth=2, color='#FF8C00')  # Orange for Norris (drawn on top)
            
            # Add corner markers to each subplot (skip for Throttle, nGear, and Brake)
            if channel not in ['Throttle', 'nGear', 'Brake']:
                corner_positions = self.get_corner_positions(time_scaled)
                if corner_positions:
                    for corner_pos, corner_name in corner_positions:
                        axes[i].axvline(x=corner_pos, color='red', linestyle=':', alpha=0.7, linewidth=1)
                        if i == 0:  # Only add corner labels to the first subplot to avoid clutter
                            # Get the y-axis range for positioning labels
                            y_min, y_max = axes[i].get_ylim()
                            y_range = y_max - y_min
                            label_y = y_min - (y_range * 0.05)  # Position labels below the plot
                            
                            axes[i].text(corner_pos, label_y, corner_name, 
                                    fontsize=9, ha='center', va='top', rotation=0,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8, edgecolor='red'))
            
            # Remove box around graph for Throttle, nGear, and Brake
            if channel in ['Throttle', 'nGear', 'Brake']:
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
                axes[i].spines['bottom'].set_visible(False)
                axes[i].spines['left'].set_visible(False)
            
            # Customize the subplot
            axes[i].set_ylabel(f'{channel} ({channel_info["unit"]})', fontsize=10)
            axes[i].set_title(f'{channel} Comparison', fontsize=12, fontweight='bold')
            axes[i].legend(fontsize=9)  # Grid removed
            
            # Set axis limits
            if channel in ['Speed', 'RPM', 'Throttle']:
                axes[i].set_ylim(0, max(max(data1), max(data2)) * 1.05)
        
        # Remove x-axis label and tick labels
        axes[-1].set_xlabel('', fontsize=12)
        for ax in axes:
            ax.set_xticks([])
        
        # Add main title
        title_suffix = ' (Time-Scaled)' if time_scaled else ''
        fig.suptitle(f'Telemetry Comparison: {driver1} vs {driver2}{title_suffix}\n'
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
    parser = argparse.ArgumentParser(description='F1 Telemetry Comparison Visualization')
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
    parser.add_argument('--time-scaled', action='store_true', 
                       help='Use time as percentage of lap time (0-100%) instead of absolute time')
    parser.add_argument('--list-channels', action='store_true', help='List available channels for driver1')
    parser.add_argument('--animate-channel', type=str, help='Animate a single channel to MP4 (e.g., Speed, Throttle, Brake, nGear, TimeDelta)')
    parser.add_argument('--video', type=str, help='Output MP4 path for animation')
    parser.add_argument('--duration', type=float, default=28.0, help='Animation duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize the comparison tool
    comparison = F1TelemetryComparison(args.data_dir)
    
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
    
    # Animation path
    if args.animate_channel:
        if not args.video:
            logger.error('Please provide --video output path when using --animate-channel')
            return
        try:
            comparison.animate_channel(args.driver1, args.driver2, args.animate_channel, args.video, args.time_scaled, args.duration)
            logger.info('Channel animation created successfully!')
        except Exception as e:
            logger.error(f"Error creating animation: {e}")
            raise
        return

    # Create the comparison plot
    try:
        if args.channels:
            # Multiple channels comparison
            comparison.plot_multiple_channels(args.driver1, args.driver2, args.channels, args.save, args.time_scaled)
        elif args.sectors:
            # Single channel with sectors
            comparison.plot_telemetry_with_sectors(args.driver1, args.driver2, args.channel, args.save, args.time_scaled)
        else:
            # Single channel basic comparison
            comparison.plot_telemetry_comparison(args.driver1, args.driver2, args.channel, args.save, args.time_scaled)
        
        logger.info("Telemetry comparison plot created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        raise


if __name__ == "__main__":
    main()

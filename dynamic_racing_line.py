"""
F1 Dynamic Racing Line Tool
===========================

This script creates a single racing line where the color changes based on 
which driver is ahead in time delta at each point on the track.

Usage:
    python dynamic_racing_line.py --data-dir "./f1_data_comprehensive" --driver1 NOR --driver2 PIA
    python dynamic_racing_line.py --data-dir "./f1_data_comprehensive" --driver1 VER --driver2 HAM --save "./ver_vs_ham_dynamic_line.png"
"""

import json
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import argparse
import os
from typing import Dict, List, Optional, Tuple
import logging
from matplotlib.colors import LinearSegmentedColormap

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1DynamicRacingLine:
    """
    Tool for creating dynamic racing lines based on time delta.
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
            True if successful, False otherwise
        """
        try:
            # Look for telemetry directory
            telemetry_dir = None
            for item in os.listdir(self.data_dir):
                if item.endswith('_telemetry'):
                    telemetry_dir = os.path.join(self.data_dir, item)
                    break
            
            if not telemetry_dir:
                logger.error(f"Could not find telemetry directory in {self.data_dir}")
                return False
            
            driver_file = os.path.join(telemetry_dir, f"{driver}_telemetry.json")
            with open(driver_file, 'r') as f:
                self.driver_data[driver] = json.load(f)
            logger.info(f"Loaded telemetry data for {driver}")
            return True
        except Exception as e:
            logger.error(f"Failed to load driver data for {driver}: {e}")
            return False
    
    def get_racing_line_data(self, driver: str) -> Tuple[List[float], List[float]]:
        """
        Extract X,Y position data for a driver's racing line.
        
        Args:
            driver: Driver abbreviation
            
        Returns:
            Tuple of (x_coordinates, y_coordinates)
        """
        if driver not in self.driver_data:
            raise ValueError(f"Driver {driver} data not loaded")
        
        driver_data = self.driver_data[driver]
        
        # Get position data
        if 'position_data' not in driver_data:
            raise ValueError(f"No position data available for {driver}")
        
        position_data = driver_data['position_data']
        x_coords = position_data.get('X', [])
        y_coords = position_data.get('Y', [])
        
        if not x_coords or not y_coords:
            raise ValueError(f"No X,Y coordinates available for {driver}")
        
        # Filter out None values and create clean arrays
        clean_x = []
        clean_y = []
        
        for x, y in zip(x_coords, y_coords):
            if x is not None and y is not None:
                clean_x.append(x)
                clean_y.append(y)
        
        return clean_x, clean_y
    
    def calculate_time_delta_along_track(self, driver1: str, driver2: str) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate time delta between two drivers based on distance percentage.
        
        Args:
            driver1: First driver abbreviation (reference)
            driver2: Second driver abbreviation
            
        Returns:
            Tuple of (x_coords, y_coords, time_deltas)
        """
        # Get position data for both drivers
        x1, y1 = self.get_racing_line_data(driver1)
        x2, y2 = self.get_racing_line_data(driver2)
        
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
        common_dist_pct = np.linspace(0, 100, 1000)
        
        # Interpolate times to common distance percentage grid
        time1_interp = np.interp(common_dist_pct, dist_pct1, 
                                [t if t is not None else 0 for t in time1_seconds])
        time2_interp = np.interp(common_dist_pct, dist_pct2, 
                                [t if t is not None else 0 for t in time2_seconds])
        
        # Interpolate positions to common distance percentage grid
        x1_interp = np.interp(common_dist_pct, dist_pct1, x1)
        y1_interp = np.interp(common_dist_pct, dist_pct1, y1)
        
        # Calculate time deltas at each distance percentage
        time_deltas = time1_interp - time2_interp
        
        return x1_interp.tolist(), y1_interp.tolist(), time_deltas.tolist()
    
    def plot_dynamic_racing_line(self, driver1: str, driver2: str, save_path: Optional[str] = None) -> None:
        """
        Create a dynamic racing line plot where color changes based on time delta.
        
        Args:
            driver1: First driver abbreviation (reference)
            driver2: Second driver abbreviation
            save_path: Optional path to save the plot
        """
        # Get racing line data with time deltas
        x_coords, y_coords, time_deltas = self.calculate_time_delta_along_track(driver1, driver2)
        
        # Get driver info
        info1 = self.driver_data[driver1]
        info2 = self.driver_data[driver2]
        session_info = self.session_data['session_info']
        
        # Create the plot
        plt.figure(figsize=(15, 12))
        
        # Create a colormap: vibrant orange for driver1 ahead, dark grey for driver2 ahead
        colors = ['#FF8C00' if delta >= 0 else '#404040' for delta in time_deltas]
        
        # Plot the dynamic racing line
        for i in range(len(x_coords) - 1):
            plt.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                    color=colors[i], linewidth=8, alpha=0.9)
        
        # Connect the finish line back to the start line to complete the loop
        if len(x_coords) > 1:
            # Use the color of the last segment to connect back to start
            plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 
                    color=colors[-1], linewidth=8, alpha=0.9)
        
        # Add start/finish markers
        plt.scatter(x_coords[0], y_coords[0], color='green', s=100, marker='o', 
                   label='Start/Finish', zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], color='red', s=100, marker='s', 
                   label='End of Lap', zorder=5)
        
        # Add corner markers if available
        corner_positions = self.get_corner_positions()
        if corner_positions:
            for corner_x, corner_y, corner_name in corner_positions:
                plt.scatter(corner_x, corner_y, color='orange', s=50, marker='^', zorder=4)
                plt.annotate(corner_name, (corner_x, corner_y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Customize the plot
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        plt.title(f'Dynamic Racing Line: {driver1} vs {driver2}\n'
                 f'Blue = {driver1} ahead, Red = {driver2} ahead\n'
                 f'{session_info["event"]} - {session_info["session"]}', 
                 fontsize=14, fontweight='bold')
        
        # Add lap time information
        lap_time1 = info1['lap_time_seconds']
        lap_time2 = info2['lap_time_seconds']
        time_diff = lap_time2 - lap_time1
        
        # Add lap time info to the plot
        plt.text(0.02, 0.98, f'{driver1}: {lap_time1:.3f}s', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.text(0.02, 0.92, f'{driver2}: {lap_time2:.3f}s', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        if time_diff > 0:
            plt.text(0.02, 0.86, f'{driver1} faster by {time_diff:.3f}s', 
                    transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            plt.text(0.02, 0.86, f'{driver2} faster by {abs(time_diff):.3f}s', 
                    transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label=f'{driver1} ahead'),
            Patch(facecolor='red', label=f'{driver2} ahead'),
            Patch(facecolor='green', label='Start/Finish'),
            Patch(facecolor='orange', label='Corners')
        ]
        plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
        
        # Grid removed for cleaner appearance
        
        # Set equal aspect ratio for proper track representation
        plt.axis('equal')
        
        # Add some padding around the track
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def animate_dynamic_racing_line(self, driver1: str, driver2: str, video_path: str, duration_sec: float = 30.0) -> None:
        """
        Create an animated dynamic racing line video.
        Draw a thin light grey outline of the track, then animate orange/grey progression
        over the track based on who is ahead on time delta at each distance percentage.
        
        Args:
            driver1: First driver abbreviation (reference; orange when ahead)
            driver2: Second driver abbreviation (dark grey when ahead)
            video_path: Output MP4 path
            duration_sec: Target video duration in seconds (20–40 recommended)
        """
        # Compute track and delta data
        x_coords, y_coords, time_deltas = self.calculate_time_delta_along_track(driver1, driver2)

        # Build segments and per-segment colors
        segments = []
        colors = []
        for i in range(len(x_coords) - 1):
            segments.append([[x_coords[i], y_coords[i]], [x_coords[i+1], y_coords[i+1]]])
            colors.append('#FF8C00' if time_deltas[i] >= 0 else '#404040')
        # Close the loop (use last segment color)
        if len(x_coords) > 1:
            segments.append([[x_coords[-1], y_coords[-1]], [x_coords[0], y_coords[0]]])
            colors.append('#FF8C00' if time_deltas[-1] >= 0 else '#404040')

        # Figure and axes
        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        session_info = self.session_data['session_info'] if self.session_data else {}
        title_line = f"Dynamic Racing Line (Animated): {driver1} vs {driver2}"
        subtitle = f"{session_info.get('event','')} - {session_info.get('session','')}"
        ax.set_title(f"{title_line}\n{subtitle}", fontsize=14, fontweight='bold')

        # Light grey outline of the full track path
        outline_x = x_coords + [x_coords[0]]
        outline_y = y_coords + [y_coords[0]]
        ax.plot(outline_x, outline_y, color='#d3d3d3', linewidth=1.5, alpha=0.9, zorder=1)

        # Prepare animated line as a LineCollection-like incremental plot
        # We'll manage a list of Line2D segments and update visibility progressively
        animated_lines = []
        for (p0, p1), c in zip(segments, colors):
            ln, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, linewidth=8, alpha=0.9, zorder=2)
            ln.set_visible(False)
            animated_lines.append(ln)

        # Limits with padding
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Remove grid for cleaner appearance
        ax.grid(False)

        # Animation timing
        total_frames = len(animated_lines)
        if total_frames <= 0:
            logger.warning("No segments to animate; skipping video export.")
            return
        # Constrain duration to 20–40s if not already
        duration_sec = max(20.0, min(40.0, duration_sec))
        interval_ms = (duration_sec * 1000.0) / total_frames
        fps = max(1, int(round(1000.0 / interval_ms)))

        def init():
            for ln in animated_lines:
                ln.set_visible(False)
            return animated_lines

        def update(frame_idx: int):
            # Reveal segments up to current frame
            if frame_idx < len(animated_lines):
                animated_lines[frame_idx].set_visible(True)
            return [animated_lines[frame_idx]] if frame_idx < len(animated_lines) else []

        anim = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=total_frames,
            interval=interval_ms,
            blit=True,
            repeat=False
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)

        # Save as MP4 using ffmpeg
        try:
            writer = animation.FFMpegWriter(fps=fps, metadata={'artist': 'F1 Telemetry Analysis'})
            anim.save(video_path, writer=writer, dpi=200)
        except Exception as e:
            logger.error(f"Failed to write MP4 with ffmpeg: {e}")
            raise
        finally:
            plt.close(fig)
        logger.info(f"Video saved to: {video_path} (duration ~{duration_sec:.1f}s, fps {fps})")
    
    def get_corner_positions(self) -> List[Tuple[float, float, str]]:
        """
        Get corner positions as X,Y coordinates.
        
        Returns:
            List of (x, y, corner_name) tuples
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
        
        # Use a set to track unique corner numbers to avoid duplicates
        seen_corners = set()
        
        for corner in corners:
            if 'X' in corner and 'Y' in corner and 'Number' in corner:
                x = corner['X']
                y = corner['Y']
                corner_number = int(corner['Number'])
                
                # Handle duplicate corner numbers
                if corner_number in seen_corners:
                    corner_name = f"T{corner_number}a"
                else:
                    corner_name = f"T{corner_number}"
                    seen_corners.add(corner_number)
                
                corner_positions.append((x, y, corner_name))
        
        return corner_positions


def main():
    """Main function to run the dynamic racing line tool."""
    parser = argparse.ArgumentParser(description='Create dynamic F1 racing line based on time delta')
    parser.add_argument('--data-dir', type=str, default='./f1_data_comprehensive',
                       help='Directory containing F1 telemetry data')
    parser.add_argument('--driver1', type=str, required=True,
                       help='First driver abbreviation (e.g., VER, HAM)')
    parser.add_argument('--driver2', type=str, required=True,
                       help='Second driver abbreviation (e.g., VER, HAM)')
    parser.add_argument('--save', type=str, help='Path to save the plot (optional)')
    parser.add_argument('--animate', action='store_true', help='Export an animated MP4 of the dynamic racing line')
    parser.add_argument('--video', type=str, help='Path to save the MP4 when using --animate')
    parser.add_argument('--duration', type=float, default=30.0, help='Video duration in seconds (20–40 recommended)')
    
    args = parser.parse_args()
    
    # Initialize the dynamic racing line tool
    dynamic_line = F1DynamicRacingLine(args.data_dir)
    
    # Find session JSON file
    session_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json') and 'telemetry' not in f]
    if not session_files:
        logger.error("Could not find session JSON file")
        return
    
    session_file = os.path.join(args.data_dir, session_files[0])
    
    # Load session data
    if not dynamic_line.load_session_data(session_file):
        logger.error("Failed to load session data")
        return
    
    # Load driver data
    if not dynamic_line.load_driver_data(args.driver1):
        logger.error(f"Failed to load data for {args.driver1}")
        return
    
    if not dynamic_line.load_driver_data(args.driver2):
        logger.error(f"Failed to load data for {args.driver2}")
        return
    
    # Create the dynamic racing line plot or animation
    try:
        if args.animate:
            if not args.video:
                logger.error('Please provide --video path when using --animate')
                return
            dynamic_line.animate_dynamic_racing_line(args.driver1, args.driver2, args.video, duration_sec=args.duration)
            logger.info("Dynamic racing line video created successfully!")
        else:
            dynamic_line.plot_dynamic_racing_line(args.driver1, args.driver2, args.save)
            logger.info("Dynamic racing line plot created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        raise


if __name__ == "__main__":
    main()

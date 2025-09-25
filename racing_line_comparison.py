"""
F1 Racing Line Comparison Tool
=============================

This script creates a top-down view of the racing lines taken by two drivers
using their X,Y position data from the F1 telemetry.

Usage:
    python racing_line_comparison.py --data-dir "./f1_data_comprehensive" --driver1 NOR --driver2 PIA
    python racing_line_comparison.py --data-dir "./f1_data_comprehensive" --driver1 VER --driver2 HAM --save "./ver_vs_ham_racing_lines.png"
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1RacingLineComparison:
    """
    Tool for comparing racing lines between two F1 drivers.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the racing line comparison tool.
        
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
    
    def plot_racing_line_comparison(self, driver1: str, driver2: str, save_path: Optional[str] = None) -> None:
        """
        Create a racing line comparison plot between two drivers.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            save_path: Optional path to save the plot
        """
        # Get racing line data for both drivers
        x1, y1 = self.get_racing_line_data(driver1)
        x2, y2 = self.get_racing_line_data(driver2)
        
        # Get driver info
        info1 = self.driver_data[driver1]
        info2 = self.driver_data[driver2]
        session_info = self.session_data['session_info']
        
        # Create the plot
        plt.figure(figsize=(15, 12))
        
        # Plot racing lines
        plt.plot(x1, y1, label=f"{driver1} ({info1['team']})", linewidth=3, color='#1f77b4', alpha=0.8)
        plt.plot(x2, y2, label=f"{driver2} ({info2['team']})", linewidth=3, color='#ff7f0e', alpha=0.8)
        
        # Add start/finish markers
        plt.scatter(x1[0], y1[0], color='green', s=100, marker='o', label='Start/Finish', zorder=5)
        plt.scatter(x1[-1], y1[-1], color='red', s=100, marker='s', label='End of Lap', zorder=5)
        
        # Add corner markers if available
        corner_positions = self.get_corner_positions()
        if corner_positions:
            for corner_x, corner_y, corner_name in corner_positions:
                plt.scatter(corner_x, corner_y, color='red', s=50, marker='^', zorder=4)
                plt.annotate(corner_name, (corner_x, corner_y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Customize the plot
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        plt.title(f'Racing Line Comparison: {driver1} vs {driver2}\n'
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
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='upper right')
        
        # Set equal aspect ratio for proper track representation
        plt.axis('equal')
        
        # Add some padding around the track
        x_min, x_max = min(min(x1), min(x2)), max(max(x1), max(x2))
        y_min, y_max = min(min(y1), min(y2)), max(max(y1), max(y2))
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
    """Main function to run the racing line comparison tool."""
    parser = argparse.ArgumentParser(description='Compare F1 racing lines between two drivers')
    parser.add_argument('--data-dir', type=str, default='./f1_data_comprehensive',
                       help='Directory containing F1 telemetry data')
    parser.add_argument('--driver1', type=str, required=True,
                       help='First driver abbreviation (e.g., VER, HAM)')
    parser.add_argument('--driver2', type=str, required=True,
                       help='Second driver abbreviation (e.g., VER, HAM)')
    parser.add_argument('--save', type=str, help='Path to save the plot (optional)')
    
    args = parser.parse_args()
    
    # Initialize the comparison tool
    comparison = F1RacingLineComparison(args.data_dir)
    
    # Find session JSON file
    session_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json') and 'telemetry' not in f]
    if not session_files:
        logger.error("Could not find session JSON file")
        return
    
    session_file = os.path.join(args.data_dir, session_files[0])
    
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
    
    # Create the racing line comparison plot
    try:
        comparison.plot_racing_line_comparison(args.driver1, args.driver2, args.save)
        logger.info("Racing line comparison plot created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        raise


if __name__ == "__main__":
    main()

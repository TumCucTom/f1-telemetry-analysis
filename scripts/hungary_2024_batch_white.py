"""
Hungary 2024 Batch Telemetry Comparison - White Background
========================================================

This script generates individual telemetry graphs and dynamic racing lines for Hungary 2024
with white backgrounds, creating separate files for each of the 7 telemetry channels.

Usage:
    python hungary_2024_batch_white.py
"""

import subprocess
import os
import sys
import logging
from typing import List, Tuple
from datetime import datetime
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Hungary2024BatchWhite:
    """Batch telemetry comparison tool for Hungary 2024 with white backgrounds."""
    
    def __init__(self):
        """Initialize the Hungary 2024 batch comparison tool."""
        self.data_dir = "./f1_data_comprehensive"
        self.output_dir = "./hungary_2024_white_graphs"
        
        # Create output directory structure
        self._create_output_structure()
        
        # Define the 7 telemetry channels
        self.channels = ['Speed', 'RPM', 'Throttle', 'Brake', 'nGear', 'DRS', 'TimeDelta']
        
    def _create_output_structure(self):
        """Create the output directory structure."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/telemetry_graphs", exist_ok=True)
        os.makedirs(f"{self.output_dir}/racing_lines", exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
        
        logger.info(f"📁 Created output directory structure in: {self.output_dir}")
    
    def get_available_drivers(self) -> List[str]:
        """
        Get list of available drivers from the telemetry directory.
        
        Returns:
            List of driver abbreviations
        """
        # Find the telemetry directory dynamically
        telemetry_dir = None
        for item in os.listdir(self.data_dir):
            if item.endswith("_telemetry") and os.path.isdir(f"{self.data_dir}/{item}"):
                telemetry_dir = f"{self.data_dir}/{item}"
                break
        
        if not telemetry_dir or not os.path.exists(telemetry_dir):
            logger.error(f"❌ Telemetry directory not found in: {self.data_dir}")
            return []
        
        drivers = []
        for filename in os.listdir(telemetry_dir):
            if filename.endswith("_telemetry.json"):
                driver_abbr = filename.replace("_telemetry.json", "")
                drivers.append(driver_abbr)
        
        logger.info(f"🏎️ Found {len(drivers)} drivers: {', '.join(sorted(drivers))}")
        return sorted(drivers)
    
    def get_all_driver_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all possible driver pairs from available drivers.
        
        Returns:
            List of (driver1, driver2) tuples
        """
        drivers = self.get_available_drivers()
        if not drivers:
            return []
        
        pairs = []
        # Generate all possible pairs (excluding self-comparisons)
        for i in range(len(drivers)):
            for j in range(i + 1, len(drivers)):
                pairs.append((drivers[i], drivers[j]))
        
        logger.info(f"📊 Generated {len(pairs)} driver pairs")
        return pairs
    
    def run_command(self, command: str, description: str) -> bool:
        """
        Run a command and log the result.
        
        Args:
            command: Command to run
            description: Description for logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🔄 {description}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"   ✅ {description} - SUCCESS")
                return True
            else:
                logger.error(f"   ❌ {description} - ERROR: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ {description} - ERROR: {str(e)}")
            return False
    
    def generate_individual_telemetry_graphs(self, driver1: str, driver2: str, pair_index: int, total_pairs: int) -> bool:
        """
        Generate individual telemetry channel graphs for a driver pair.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            pair_index: Current pair index (for progress tracking)
            total_pairs: Total number of pairs (for progress tracking)
            
        Returns:
            True if successful, False otherwise
        """
        success_count = 0
        
        for channel in self.channels:
            filename = f"{driver1}_vs_{driver2}_{channel.lower()}_white.png"
            output_path = f"{self.output_dir}/telemetry_graphs/{filename}"
            
            command = f'python telemetry_comparison_white.py --data-dir "{self.data_dir}" --driver1 {driver1} --driver2 {driver2} --channel {channel} --time-scaled --save "{output_path}"'
            
            if self.run_command(command, f"[{pair_index}/{total_pairs}] {channel} comparison: {driver1} vs {driver2}"):
                success_count += 1
        
        # Return True if at least half the channels succeeded
        return success_count >= len(self.channels) // 2
    
    def generate_dynamic_racing_line(self, driver1: str, driver2: str, pair_index: int, total_pairs: int) -> bool:
        """
        Generate dynamic racing line for a driver pair.
        
        Args:
            driver1: First driver abbreviation
            driver2: Second driver abbreviation
            pair_index: Current pair index (for progress tracking)
            total_pairs: Total number of pairs (for progress tracking)
            
        Returns:
            True if successful, False otherwise
        """
        filename = f"{driver1}_vs_{driver2}_racing_line_white.png"
        output_path = f"{self.output_dir}/racing_lines/{filename}"
        
        command = f'python dynamic_racing_line_white.py --data-dir "{self.data_dir}" --driver1 {driver1} --driver2 {driver2} --time-scaled --save "{output_path}"'
        
        return self.run_command(command, f"[{pair_index}/{total_pairs}] Dynamic racing line: {driver1} vs {driver2}")
    
    def run_batch_comparison(self):
        """Run the complete batch comparison for all driver pairs."""
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("🏁 HUNGARY 2024 BATCH TELEMETRY COMPARISON - WHITE BACKGROUND")
        logger.info("=" * 80)
        
        # Get all driver pairs
        driver_pairs = self.get_all_driver_pairs()
        if not driver_pairs:
            logger.error("❌ No driver pairs found")
            return
        
        total_pairs = len(driver_pairs)
        telemetry_success = 0
        telemetry_failed = 0
        racing_line_success = 0
        racing_line_failed = 0
        
        logger.info(f"📊 Processing {total_pairs} driver pairs...")
        logger.info("=" * 80)
        
        for i, (driver1, driver2) in enumerate(driver_pairs, 1):
            logger.info(f"\n🏎️ Processing pair {i}/{total_pairs}: {driver1} vs {driver2}")
            
            # Generate individual telemetry channel comparisons
            if self.generate_individual_telemetry_graphs(driver1, driver2, i, total_pairs):
                telemetry_success += 1
            else:
                telemetry_failed += 1
            
            # Generate dynamic racing line
            if self.generate_dynamic_racing_line(driver1, driver2, i, total_pairs):
                racing_line_success += 1
            else:
                racing_line_failed += 1
        
        # Summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 HUNGARY 2024 BATCH COMPARISON COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total time: {duration:.1f} seconds")
        logger.info(f"📊 Telemetry comparisons: {telemetry_success} success, {telemetry_failed} failed")
        logger.info(f"🏁 Dynamic racing lines: {racing_line_success} success, {racing_line_failed} failed")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        # Create summary report
        self._create_summary_report(telemetry_success, telemetry_failed, racing_line_success, racing_line_failed, duration)
        
        # List generated files
        logger.info("\n📁 Generated files:")
        for root, dirs, files in os.walk(self.output_dir):
            level = root.replace(self.output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                logger.info(f"{subindent}{file}")
    
    def _create_summary_report(self, telemetry_success: int, telemetry_failed: int, 
                             racing_line_success: int, racing_line_failed: int, duration: float):
        """Create a summary report of the batch comparison."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "race": "Hungary 2024",
            "background": "white",
            "total_driver_pairs": telemetry_success + telemetry_failed,
            "telemetry_success": telemetry_success,
            "telemetry_failed": telemetry_failed,
            "racing_line_success": racing_line_success,
            "racing_line_failed": racing_line_failed,
            "duration_seconds": duration,
            "channels_generated": self.channels,
            "output_directory": self.output_dir
        }
        
        # Save JSON summary
        summary_file = f"{self.output_dir}/logs/hungary_2024_white_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save text summary
        text_summary = f"""Hungary 2024 Batch Telemetry Comparison - White Background
================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.1f} seconds

Results:
--------
Total Driver Pairs: {telemetry_success + telemetry_failed}
Telemetry Comparisons: {telemetry_success} success, {telemetry_failed} failed
Dynamic Racing Lines: {racing_line_success} success, {racing_line_failed} failed

Channels Generated:
------------------
{', '.join(self.channels)}

Output Directory: {self.output_dir}
"""
        
        text_file = f"{self.output_dir}/logs/hungary_2024_white_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_file, 'w') as f:
            f.write(text_summary)
        
        logger.info(f"📄 Summary reports saved: {summary_file}")

def main():
    """Main function to run the Hungary 2024 batch comparison."""
    batch = Hungary2024BatchWhite()
    batch.run_batch_comparison()

if __name__ == "__main__":
    main()

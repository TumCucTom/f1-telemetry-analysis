#!/usr/bin/env python3
"""
Generate Missing F1 2025 Comprehensive Graphs
============================================

This script generates the missing comparison graphs for all races in f1_2025_comprehensive
that have data but are missing graphs.

Usage:
    python generate_missing_graphs.py
"""

import subprocess
import os
import sys
import logging
from typing import List, Dict
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_races_with_data_but_no_graphs() -> List[Dict[str, str]]:
    """
    Get list of races that have data but are missing comparison graphs.
    
    Returns:
        List of race dictionaries with data_dir and comparison_dir
    """
    base_data_dir = "./f1_2025_comprehensive/data"
    base_comparison_dir = "./f1_2025_comprehensive/comparisons"
    
    races = []
    
    # Check each race directory
    for race_dir in os.listdir(base_data_dir):
        if race_dir.startswith("f1_data_2025_"):
            race_name = race_dir.replace("f1_data_2025_", "")
            data_dir = f"{base_data_dir}/{race_dir}"
            comparison_dir = f"{base_comparison_dir}/{race_name}"
            
            # Check if data exists
            telemetry_dir = None
            for item in os.listdir(data_dir):
                if item.endswith("_telemetry") and os.path.isdir(f"{data_dir}/{item}"):
                    telemetry_dir = f"{data_dir}/{item}"
                    break
            
            if telemetry_dir and os.path.exists(telemetry_dir):
                # Check if graphs exist
                telemetry_graphs_dir = f"{comparison_dir}/telemetry_graphs"
                racing_lines_dir = f"{comparison_dir}/racing_lines"
                
                has_graphs = (os.path.exists(telemetry_graphs_dir) and 
                            len(os.listdir(telemetry_graphs_dir)) > 0 and
                            os.path.exists(racing_lines_dir) and 
                            len(os.listdir(racing_lines_dir)) > 0)
                
                if not has_graphs:
                    races.append({
                        "race_name": race_name.replace("_", " ").title(),
                        "data_dir": data_dir,
                        "comparison_dir": comparison_dir
                    })
                    logger.info(f"📊 {race_name}: Data ✅, Graphs ❌")
                else:
                    logger.info(f"📊 {race_name}: Data ✅, Graphs ✅")
            else:
                logger.info(f"📊 {race_name}: Data ❌")
    
    return races

def run_comprehensive_batch_comparison(data_dir: str, output_dir: str) -> bool:
    """
    Run the comprehensive batch comparison for a specific race.
    
    Args:
        data_dir: Directory containing the race data
        output_dir: Directory to output the comparison graphs
        
    Returns:
        True if successful, False otherwise
    """
    command = [
        "python", "comprehensive_batch_comparison.py",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--all-pairs"
    ]
    
    logger.info(f"🚀 Running comprehensive batch comparison for {os.path.basename(data_dir)}")
    logger.info(f"   Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully generated graphs for {os.path.basename(data_dir)}")
            return True
        else:
            logger.error(f"❌ Failed to generate graphs for {os.path.basename(data_dir)}")
            logger.error(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout generating graphs for {os.path.basename(data_dir)}")
        return False
    except Exception as e:
        logger.error(f"💥 Exception generating graphs for {os.path.basename(data_dir)}: {e}")
        return False

def main():
    """Main function to generate missing graphs for all races."""
    logger.info("🔍 F1 2025 Comprehensive - Generating Missing Graphs")
    logger.info("=" * 60)
    
    # Get races that need graphs
    races_to_process = get_races_with_data_but_no_graphs()
    
    if not races_to_process:
        logger.info("🎉 All races already have graphs generated!")
        return
    
    logger.info(f"📋 Found {len(races_to_process)} races that need graphs:")
    for race in races_to_process:
        logger.info(f"   - {race['race_name']}")
    
    # Process each race
    successful = 0
    failed = 0
    
    for i, race in enumerate(races_to_process, 1):
        logger.info(f"\n🏁 Processing {i}/{len(races_to_process)}: {race['race_name']}")
        logger.info("-" * 50)
        
        if run_comprehensive_batch_comparison(race['data_dir'], race['comparison_dir']):
            successful += 1
        else:
            failed += 1
        
        # Add a small delay between races to avoid overwhelming the system
        if i < len(races_to_process):
            logger.info("⏳ Waiting 5 seconds before next race...")
            time.sleep(5)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Total: {len(races_to_process)}")
    
    if successful > 0:
        logger.info(f"\n🎉 Successfully generated graphs for {successful} races!")
        logger.info("📁 Check the f1_2025_comprehensive/comparisons/ directory for results.")
    
    if failed > 0:
        logger.info(f"\n⚠️  {failed} races failed to generate graphs.")
        logger.info("Check the logs above for error details.")

if __name__ == "__main__":
    main()



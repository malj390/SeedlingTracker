#!/usr/bin/env python3
"""
Generate Dummy Data for Seedling Tracker
========================================

This script generates a synthetic TIF stack to test the Seedling Tracker pipeline.
It creates a multi-frame TIF file in the '1_Data' folder simulating a growing seedling.

Usage:
    python generate_sample_data.py
"""

import numpy as np
import tifffile
from pathlib import Path

def generate_dummy_tif(filename="dummy.tif", num_frames=5, width=512, height=512):
    """
    Generates a synthetic TIF stack with a moving object.
    
    Args:
        filename (str): Name of the output file.
        num_frames (int): Number of timeframes.
        width (int): Image width.
        height (int): Image height.
    """
    print(f"Generating dummy data: {filename} ({num_frames} frames, {width}x{height})")
    
    # Create the output folder if it doesn't exist
    data_folder = Path("1_Data")
    data_folder.mkdir(exist_ok=True)
    output_path = data_folder / filename
    
    # Initialize stack (Time, Height, Width)
    stack = np.zeros((num_frames, height, width), dtype=np.uint8)
    
    # Parameters for the "seedling"
    start_x = width // 2
    start_y = height - 50
    growth_speed = 20  # pixels per frame
    radius = 10
    
    for t in range(num_frames):
        # Add some noise
        noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)
        stack[t] = noise
        
        # Draw a "growing" line/stem
        current_height = start_y - (t * growth_speed)
        
        # Simple raster circle drawing for the tip
        y, x = np.ogrid[:height, :width]
        mask = ((x - start_x)**2 + (y - current_height)**2) <= radius**2
        stack[t][mask] = 200  # Bright object
        
        # Draw the "stem" trailing behind
        if t > 0:
            stem_mask = (np.abs(x - start_x) < radius // 2) & (y > current_height) & (y < start_y)
            stack[t][stem_mask] = 150

    # Save using tifffile
    tifffile.imwrite(output_path, stack)
    print(f"✓ Saved to {output_path}")
    print("You can now run 'SeedlingNew.py' and select this file in Step 1.")

if __name__ == "__main__":
    generate_dummy_tif()

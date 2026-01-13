#!/usr/bin/env python3
"""
Seedling Tracking Pipeline
============================

Simplified workflow for processing time-lapse TIF stacks of seedlings:
- Step 1: Load TIF stack from 1_Data/, align timeframes, save to 2_Processed/
- Step 2: User draws rectangles to separate seedlings, save to 2_Processed/
- Step 3: User tracks growth points, calculate metrics, save to 3_Results/

Folder structure:
    1_Data/       - Raw TIF stacks (input)
    2_Processed/  - Aligned stacks and separated seedlings
    3_Results/    - Excel files with tracking data and plots

Author: Miguel
Version: 5.0 (Simplified)
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from skimage import filters, registration
from scipy.spatial import distance
from collections import defaultdict
import math
from tqdm import tqdm

# GUI imports
from tkinter import filedialog, Tk

# napari for interactive viewing
try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
    warnings.warn("napari not available. Interactive features will be disabled.")

# Custom tools (optional)
try:
    from Miguel_scripts import miguel_tools as mt
    CUSTOM_TOOLS_AVAILABLE = True
except ImportError:
    CUSTOM_TOOLS_AVAILABLE = False


# ============================================================================
# FOLDER SETUP
# ============================================================================

def setup_folders(base_path: Path) -> Tuple[Path, Path, Path]:
    """
    Create 1_Data/, 2_Processed/, and 3_Results/ folders if they don't exist.
    
    Args:
        base_path: Base directory (project root)
    
    Returns:
        Tuple of (data_folder, processed_folder, results_folder)
    """
    data_folder = base_path / "1_Data"
    processed_folder = base_path / "2_Processed"
    results_folder = base_path / "3_Results"
    
    data_folder.mkdir(exist_ok=True)
    processed_folder.mkdir(exist_ok=True)
    results_folder.mkdir(exist_ok=True)
    
    return data_folder, processed_folder, results_folder


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ask_file(title: str = "Select a TIF file") -> Path:
    """Open a GUI dialog to select a file."""
    root = Tk()
    root.withdraw()
    file_selected = filedialog.askopenfilename(
        title=title,
        filetypes=[("TIF files", "*.tif *.tiff"), ("All files", "*.*")]
    )
    root.destroy()
    if not file_selected:
        raise ValueError("No file selected")
    return Path(file_selected)


def calculate_shift(im1: np.ndarray, im2: np.ndarray) -> List[int]:
    """
    Calculate the shift between two images using phase cross-correlation.
    
    Args:
        im1: First image (reference)
        im2: Second image (to be registered)
    
    Returns:
        List of [shift_y, shift_x] in pixels
    """
    shift, error, diffphase = registration.phase_cross_correlation(im1, im2)
    return [int(i) for i in shift]


def shift_image(im: np.ndarray, shift: List[int]) -> np.ndarray:
    """
    Apply a shift to an image using numpy roll.
    
    Args:
        im: Input image
        shift: [shift_y, shift_x] to apply
    
    Returns:
        Shifted image
    """
    Xshifted = np.roll(im, shift=shift[1], axis=1)
    Yshifted = np.roll(Xshifted, shift=shift[0], axis=0)
    return Yshifted


# ============================================================================
# STEP 1: LOAD AND ALIGN TIF STACK
# ============================================================================

def load_and_align_stack(tif_path: Path, processed_folder: Path, 
                        reference_frame: int = 0) -> Tuple[Path, np.ndarray]:
    """
    Load a TIF stack, align all frames to a reference, and save to 2_Processed/.
    
    Args:
        tif_path: Path to input TIF stack
        processed_folder: Folder to save aligned stack
        reference_frame: Frame index to use as alignment reference (default: 0)
    
    Returns:
        Tuple of (aligned_tif_path, aligned_stack)
    """
    print("\n" + "="*70)
    print("STEP 1: LOADING AND ALIGNING TIF STACK")
    print("="*70 + "\n")
    
    print(f"Loading: {tif_path.name}")
    stack = tifffile.imread(tif_path)
    
    print(f"Stack shape: {stack.shape}")
    print(f"Data type: {stack.dtype}")
    
    # Determine if stack is 3D (t, y, x) or 4D (t, y, x, c)
    if stack.ndim == 4:
        print("Detected 4D stack (time, y, x, channels) - using first channel")
        stack = stack[:, :, :, 0]
    elif stack.ndim != 3:
        raise ValueError(f"Expected 3D or 4D stack, got {stack.ndim}D")
    
    num_frames = stack.shape[0]
    print(f"Number of timeframes: {num_frames}")
    
    if reference_frame >= num_frames:
        print(f"Warning: reference_frame {reference_frame} out of range, using frame 0")
        reference_frame = 0
    
    print(f"\nAligning all frames to frame {reference_frame}...")
    reference_image = stack[reference_frame]
    aligned_stack = np.zeros_like(stack)
    aligned_stack[reference_frame] = reference_image
    
    # Align all other frames
    for i in tqdm(range(num_frames), desc="Aligning"):
        if i == reference_frame:
            continue
        
        shift = calculate_shift(reference_image, stack[i])
        aligned_stack[i] = shift_image(stack[i], shift)
    
    # Save aligned stack
    output_path = processed_folder / f"{tif_path.stem}_aligned.tif"
    print(f"\nSaving aligned stack to: {output_path.name}")
    
    tifffile.imwrite(output_path, aligned_stack.astype(stack.dtype))
    
    print("✓ Step 1 complete!")
    return output_path, aligned_stack


# ============================================================================
# STEP 2: SEPARATE SEEDLINGS
# ============================================================================

def separate_seedlings_interactive(aligned_stack: np.ndarray, aligned_path: Path,
                                   processed_folder: Path) -> List[Path]:
    """
    Use napari to let user draw rectangles around seedlings and extract them.
    
    Args:
        aligned_stack: Aligned TIF stack
        aligned_path: Path to aligned TIF file
        processed_folder: Folder to save separated seedlings
    
    Returns:
        List of paths to separated seedling TIF files
    """
    print("\n" + "="*70)
    print("STEP 2: SEPARATING SEEDLINGS")
    print("="*70 + "\n")
    
    if not NAPARI_AVAILABLE:
        raise RuntimeError("napari is required for interactive seedling separation")
    
    print("Opening napari viewer...")
    print("\nInstructions:")
    print("  1. Use the 'Shapes' layer to draw rectangles around each seedling")
    print("  2. Draw ONE rectangle per seedling")
    print("  3. Close the viewer when done")
    print("\nNavigation:")
    print("  - Pan: Click and drag with mouse (or press/hold Space + drag)")
    print("  - Zoom: Scroll wheel up/down")
    print("  - Draw rectangle: Click and drag (tool is selected by default)")
    print("  - Select/Move rectangles: Press 'D' to switch to select mode")
    print("  - Delete rectangles: Select shape and press Delete/Backspace")
    print("  - Draw new rectangle: Press 'R' to switch back to rectangle mode")
    print()
    
    # Show max projection or middle frame for reference
    display_frame = np.max(aligned_stack, axis=0)
    
    viewer = napari.Viewer()
    viewer.add_image(display_frame, name='max_projection')
    viewer.add_image(aligned_stack, name='timeseries')
    shapes_layer = viewer.add_shapes(
        name='seedling_rectangles',
        shape_type='rectangle',
        edge_color='red',
        edge_width=3,
        face_color='transparent'
    )
    # Set rectangle tool as default
    shapes_layer.mode = 'add_rectangle'
    
    napari.run()
    
    # Extract rectangles
    rectangles = shapes_layer.data
    
    if len(rectangles) == 0:
        print("No rectangles drawn. Skipping seedling separation.")
        return []
    
    print(f"\nFound {len(rectangles)} seedlings")
    
    seedling_paths = []
    
    for idx, rect in enumerate(rectangles, start=1):
        # Rectangle format: [[y1, x1], [y2, x1], [y2, x2], [y1, x2]]
        coords = rect.astype(int)
        y_min = coords[:, 0].min()
        y_max = coords[:, 0].max()
        x_min = coords[:, 1].min()
        x_max = coords[:, 1].max()
        
        print(f"  Seedling {idx}: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")
        
        # Extract seedling
        seedling_stack = aligned_stack[:, y_min:y_max, x_min:x_max]
        
        # Save
        output_path = processed_folder / f"{aligned_path.stem}_seedling_{idx}.tif"
        tifffile.imwrite(output_path, seedling_stack.astype(aligned_stack.dtype))
        seedling_paths.append(output_path)
    
    print(f"\n✓ Step 2 complete! Saved {len(seedling_paths)} seedlings to 2_Processed/")
    return seedling_paths


# ============================================================================
# STEP 3: TRACK SEEDLING GROWTH
# ============================================================================

def track_seedling_interactive(seedling_path: Path, results_folder: Path,
                               pixel_size_mm: float = 0.036,
                               use_edge_enhancement: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Track seedling growth by having user click points in each timeframe.
    
    Args:
        seedling_path: Path to seedling TIF file
        results_folder: Folder to save results
        pixel_size_mm: Size of each pixel in mm (for metric calculations)
        use_edge_enhancement: Whether to apply Sobel edge enhancement for visualization
    
    Returns:
        Tuple of (raw_data_df, metrics_df)
    """
    print("\n" + "="*70)
    print(f"STEP 3: TRACKING SEEDLING - {seedling_path.stem}")
    print("="*70 + "\n")
    
    if not NAPARI_AVAILABLE:
        raise RuntimeError("napari is required for interactive tracking")
    
    # Load seedling
    print(f"Loading: {seedling_path.name}")
    seedling_stack = tifffile.imread(seedling_path)
    
    # Apply edge enhancement if requested
    if use_edge_enhancement:
        print("Applying Sobel edge enhancement...")
        enhanced_stack = filters.sobel(seedling_stack)
        # Rotate 90 degrees for lateral view
        display_stack = np.rot90(enhanced_stack, axes=(1, 2))
    else:
        display_stack = np.rot90(seedling_stack, axes=(1, 2))
    
    print("\nOpening napari viewer...")
    print("\nInstructions:")
    print("  1. Click ONE point per timeframe following the growth tip")
    print("  2. Points should track a consistent landmark (e.g., root tip, shoot apex)")
    print("  3. Work through frames in order (frame 0, 1, 2, ...)")
    print("  4. Close the viewer when done")
    print("\nNavigation:")
    print("  - Pan: Click and drag with mouse (or press/hold Space + drag)")
    print("  - Zoom: Scroll wheel up/down")
    print("  - Add points: Tool is selected by default (or press '2')")
    print("  - Move/Select points: Press '3' to switch to select mode")
    print("  - Delete points: Select point(s) and press Delete/Backspace")
    print("  - Switch back to add mode: Press '2'")
    print()
    
    viewer = napari.Viewer()
    viewer.add_image(display_stack, name='seedling_enhanced')
    points_layer = viewer.add_points(
        name='tracking_points',
        face_color='red',
        symbol='disc',
        size=5,
        ndim=3
    )
    # Set add points tool as default
    points_layer.mode = 'add'
    
    napari.run()
    
    # Get tracking points
    tracking_data = points_layer.data
    
    if len(tracking_data) == 0:
        print("No points tracked. Skipping analysis.")
        return None, None
    
    print(f"\nTracked {len(tracking_data)} points")
    
    # Create DataFrame with tracking data
    data_df = pd.DataFrame(tracking_data, columns=['timepoint', 'y', 'x'])
    data_df['timepoint'] = data_df['timepoint'].astype(int)
    data_df = data_df.sort_values('timepoint').reset_index(drop=True)
    
    data_df.insert(0, 'seedling_name', seedling_path.stem)
    data_df['x_mm'] = data_df['x'] * pixel_size_mm
    data_df['y_mm'] = data_df['y'] * pixel_size_mm
    
    # Calculate metrics between consecutive points
    print("Calculating growth metrics...")
    metrics = calculate_growth_metrics(data_df)
    
    # Save results
    excel_path = results_folder / f"{seedling_path.stem}.xlsx"
    print(f"\nSaving results to: {excel_path.name}")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        data_df.to_excel(writer, sheet_name='raw_data', index=False)
        metrics.to_excel(writer, sheet_name='metrics', index=False)
    
    # Create visualization
    print("Creating visualization...")
    plot_tracking_results(data_df, metrics, seedling_path.stem, pixel_size_mm, results_folder)
    
    print(f"✓ Step 3 complete for {seedling_path.stem}!")
    return data_df, metrics


def calculate_growth_metrics(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate angles, distances, and speeds between consecutive tracking points.
    
    Args:
        data_df: DataFrame with columns ['timepoint', 'y', 'x', 'y_mm', 'x_mm']
    
    Returns:
        DataFrame with calculated metrics
    """
    metrics = []
    
    for i in range(len(data_df) - 1):
        p1 = data_df.iloc[i]
        p2 = data_df.iloc[i + 1]
        
        # Calculate angle (degrees)
        angle = math.degrees(math.atan2(p2['y'] - p1['y'], p2['x'] - p1['x']))
        
        # Calculate Euclidean distance (mm)
        dist_mm = distance.euclidean([p1['y_mm'], p1['x_mm']], [p2['y_mm'], p2['x_mm']])
        
        # Middle point for labeling
        mid_y = (p1['y'] + p2['y']) / 2
        mid_x = (p1['x'] + p2['x']) / 2
        
        metrics.append({
            'from_timepoint': int(p1['timepoint']),
            'to_timepoint': int(p2['timepoint']),
            'segment': f"{int(p1['timepoint'])}-{int(p2['timepoint'])}",
            'angle_degrees': angle,
            'distance_mm': dist_mm,
            'mid_y': mid_y,
            'mid_x': mid_x
        })
    
    return pd.DataFrame(metrics)


def plot_tracking_results(data_df: pd.DataFrame, metrics_df: pd.DataFrame,
                          seedling_name: str, pixel_size: float, 
                          results_folder: Path):
    """
    Create and save a plot showing tracking trajectory with angle annotations.
    
    Args:
        data_df: Raw tracking data
        metrics_df: Calculated metrics
        seedling_name: Name of the seedling
        pixel_size: Pixel size in mm
        results_folder: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot trajectory
    ax.plot(data_df['x'], data_df['y'], 'o-', color='blue', markersize=8, linewidth=2)
    
    # Add point labels
    for idx, row in data_df.iterrows():
        ax.text(row['x'] + 2, row['y'] + 2, f"t{int(row['timepoint'])}", 
                fontsize=10, color='darkblue')
    
    # Add angle annotations
    for _, row in metrics_df.iterrows():
        ax.text(row['mid_x'], row['mid_y'], f"{int(row['angle_degrees'])}°",
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Formatting
    ax.set_xlabel("X distance (mm)", fontsize=12)
    ax.set_ylabel("Y distance (mm)", fontsize=12)
    ax.set_title(f"{seedling_name} - Growth Tracking", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Invert Y axis for image coordinates
    
    # Convert axes to mm
    xlocs = ax.get_xticks()
    xnewlabels = [f"{int(x * pixel_size)}" for x in xlocs]
    ax.set_xticklabels(xnewlabels)
    
    ylocs = ax.get_yticks()
    ynewlabels = [f"{int(y * pixel_size)}" for y in ylocs]
    ax.set_yticklabels(ynewlabels)
    
    plt.tight_layout()
    
    # Save
    plot_path_png = results_folder / f"{seedling_name}_tracking.png"
    plot_path_svg = results_folder / f"{seedling_name}_tracking.svg"
    
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_svg, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main workflow execution."""
    print("\n" + "="*70)
    print("SEEDLING TRACKING PIPELINE")
    print("="*70)
    
    try:
        # Setup folders
        base_path = Path(__file__).parent
        data_folder, processed_folder, results_folder = setup_folders(base_path)
        
        print(f"\nFolder structure:")
        print(f"  Data:      {data_folder}")
        print(f"  Processed: {processed_folder}")
        print(f"  Results:   {results_folder}")
        
        # Ask user what to do
        print("\n" + "="*70)
        print("SELECT WORKFLOW:")
        print("  1 - Load and align TIF stack")
        print("  2 - Separate seedlings from aligned stack")
        print("  3 - Track individual seedling")
        print("  4 - Run full pipeline (1 → 2 → 3)")
        print("="*70)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # Step 1 only
            print("\nSelect TIF stack from Data/ folder...")
            tif_path = ask_file("Select TIF stack to align")
            aligned_path, aligned_stack = load_and_align_stack(tif_path, processed_folder)
            print(f"\n✓ Aligned stack saved: {aligned_path}")
        
        elif choice == '2':
            # Step 2 only
            print("\nSelect aligned TIF stack from Processed/ folder...")
            aligned_path = ask_file("Select aligned TIF stack")
            aligned_stack = tifffile.imread(aligned_path)
            seedling_paths = separate_seedlings_interactive(aligned_stack, aligned_path, processed_folder)
            print(f"\n✓ Separated {len(seedling_paths)} seedlings")
        
        elif choice == '3':
            # Step 3 only
            print("\nSelect seedling TIF from Processed/ folder...")
            seedling_path = ask_file("Select seedling TIF to track")
            
            pixel_size = input("\nEnter pixel size in mm (default 0.036): ").strip()
            pixel_size = float(pixel_size) if pixel_size else 0.036
            
            enhance = input("Use edge enhancement? (y/n, default y): ").strip().lower()
            use_enhancement = enhance != 'n'
            
            track_seedling_interactive(seedling_path, results_folder, pixel_size, use_enhancement)
        
        elif choice == '4':
            # Full pipeline
            print("\n" + "="*70)
            print("RUNNING FULL PIPELINE")
            print("="*70)
            
            # Step 1
            print("\nSelect TIF stack from Data/ folder...")
            tif_path = ask_file("Select TIF stack to process")
            aligned_path, aligned_stack = load_and_align_stack(tif_path, processed_folder)
            
            # Step 2
            seedling_paths = separate_seedlings_interactive(aligned_stack, aligned_path, processed_folder)
            
            if len(seedling_paths) == 0:
                print("\nNo seedlings to track. Exiting.")
                return
            
            # Step 3 - track each seedling
            pixel_size = input("\nEnter pixel size in mm (default 0.036): ").strip()
            pixel_size = float(pixel_size) if pixel_size else 0.036
            
            enhance = input("Use edge enhancement? (y/n, default y): ").strip().lower()
            use_enhancement = enhance != 'n'
            
            for seedling_path in seedling_paths:
                track_seedling_interactive(seedling_path, results_folder, pixel_size, use_enhancement)
            
            print("\n" + "="*70)
            print("✓ FULL PIPELINE COMPLETE!")
            print("="*70)
        
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

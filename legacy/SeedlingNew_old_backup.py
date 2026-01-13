#!/usr/bin/env python3
"""
Seedling Tracking Pipeline
============================

A streamlined script for processing time-lapse TIF stacks of seedlings:
- Step 1: Load TIF stack, align timeframes, save to Processed/
- Step 2: User draws rectangles to separate seedlings, save to Processed/
- Step 3: User tracks points in each seedling, calculate metrics, save to Results/

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
import datetime
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
    CUSTOM_TOOLS_AVAILABLE = True
except ImportError:
    CUSTOM_TOOLS_AVAILABLE = False
    warnings.warn("miguel_tools or iTools not available. Some features may be limited.")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ask_folder() -> Path:
    """Open a GUI dialog to select a folder."""
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    root.destroy()
    if not folder_selected:
        raise ValueError("No folder selected")
    return Path(folder_selected)


def ask_file() -> Path:
    """Open a GUI dialog to select a file."""
    root = Tk()
    root.withdraw()
    file_selected = filedialog.askopenfilename()
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


def registration_checkpoint(im1: np.ndarray, im2: np.ndarray, 
                           shift: List[int], savein: Optional[Path] = None):
    """
    Create a visual checkpoint comparing raw and shifted image alignment.
    
    Args:
        im1: Reference image
        im2: Image to be aligned
        shift: Calculated shift values
        savein: Path to save the checkpoint image (optional)
    """
    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    
    ax[0].set_title("Raw")
    ax[0].imshow(im1, alpha=0.5, cmap='Reds')
    ax[0].imshow(im2, alpha=0.5, cmap='Greens')
    
    im2_shifted = shift_image(im2, shift)
    ax[1].set_title("Shifted")
    ax[1].imshow(im1, alpha=0.5, cmap='Reds')
    ax[1].imshow(im2_shifted, alpha=0.5, cmap='Greens')
    
    for i in ax:
        i.axis('off')
    
    if savein is not None:
        plt.savefig(savein, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================================
# STEP A: PLATE PROCESSING AND SEEDLING EXTRACTION
# ============================================================================

def load_images_from_folder(folder_tif: Path) -> Tuple[np.ndarray, List[str], Path]:
    """
    Load all JPG images from a folder and prepare metadata.
    
    Args:
        folder_tif: Path to folder containing TIF time-lapse images
    
    Returns:
        Tuple of (stacked_images, timepoints, save_folder)
    """
    tif_paths = sorted(list(folder_tif.glob("*.tif")))
    
    if not tif_paths:
        raise ValueError(f"No TIF files found in {folder_tif}")
    
    print(f"Found {len(tif_paths)} images")
    
    # Create results folder
    save_folder = folder_tif.parent / f"{folder_tif.name}_results"
    save_folder.mkdir(exist_ok=True)
    
    # Extract timepoints from filenames
    timepoints = [i.stem.rstrip("W") for i in tif_paths]
    
    # Read images (take only red channel)
    print("Loading images...")
    ims = [tf.imread(i)[:, :, 0] for i in tqdm(tif_paths)]
    
    return ims, timepoints, save_folder


def center_crop_images(ims: List[np.ndarray], plate_size: float = 120.0,
                       pixel_plate_dim: Tuple[int, int] = (3324, 3312)) -> Tuple[np.ndarray, float]:
    """
    Crop images to center square based on plate dimensions.
    
    Args:
        ims: List of images
        plate_size: Physical plate size in mm
        pixel_plate_dim: Pixel dimensions of the plate
    
    Returns:
        Tuple of (cropped_images_stack, pixel_size_in_mm)
    """
    shapes = ims[0].shape
    min_dim = np.min(pixel_plate_dim)
    
    pixel_size = plate_size / min_dim
    print(f"\nPlate size: {plate_size} mm")
    print(f"Pixel width/height of the plate: {pixel_plate_dim}")
    print(f"Pixel size: {round(pixel_size, 3)} mm/pixel --> {round(pixel_size * 1000, 2)} µm/pixel\n")
    
    height = int((shapes[0] - min_dim) / 2)
    width = int((shapes[1] - min_dim) / 2)
    
    print("Cropping images to center square...")
    imscut = np.stack([i[height:height + min_dim, width:width + min_dim] 
                      for i in tqdm(ims)], axis=0)
    
    return imscut, pixel_size


def correct_rotation_and_shift(plate: np.ndarray, rot_from: int, 
                               save_folder: Path, folder_name: str) -> np.ndarray:
    """
    Correct rotation and shift in time-series images.
    
    Args:
        plate: 3D array of images (time, y, x)
        rot_from: Frame index where rotation starts
        save_folder: Folder to save checkpoint images
        folder_name: Name for saved files
    
    Returns:
        Corrected image stack
    """
    print(f"Correcting rotation from frame {rot_from}...")
    
    # Apply rotation
    for i in tqdm(range(rot_from, plate.shape[0]), desc="Rotating"):
        plate[i] = np.rot90(plate[i], axes=(1, 0))
    
    # Calculate shift
    shift = calculate_shift(plate[rot_from - 1], plate[rot_from])
    
    # Save checkpoint
    registration_checkpoint(
        plate[rot_from - 1],
        plate[rot_from],
        shift=shift,
        savein=save_folder / f"{folder_name}_registration_check.png"
    )
    
    # Apply shift
    print("Applying shift correction...")
    first_part = np.copy(plate[:rot_from])
    second_part = []
    for i in tqdm(range(rot_from, plate.shape[0]), desc="Shifting"):
        second_part.append(shift_image(plate[i], shift))
    
    second_part = np.stack(second_part, axis=0)
    combined = np.concatenate([first_part, second_part], axis=0)
    
    return combined


def extract_seedlings(ims_cut_corrected: np.ndarray, imscut: np.ndarray,
                     coordinates: np.ndarray) -> List[np.ndarray]:
    """
    Extract individual seedlings from the plate based on coordinates.
    
    Args:
        ims_cut_corrected: Corrected full plate images
        imscut: Original cropped images
        coordinates: Array of corner coordinates for each seedling (pairs of y,x)
    
    Returns:
        List of seedling image stacks
    """
    coordinates = coordinates.astype(int)
    # Reshape to pairs (assuming format: [y1, x1, y2, x2, ...])
    coordinates = coordinates.reshape(-1, 2, 2)
    
    seedling_names = [f"seedling_{i+1}" for i in range(len(coordinates))]
    print(f"Extracting {len(seedling_names)} seedlings:\n{', '.join(seedling_names)}\n")
    
    plants_separated = []
    for coord in tqdm(coordinates, desc="Extracting seedlings"):
        y1, x1 = coord[0]
        y2, x2 = coord[1]
        plant_cut = imscut[:, y1:y2, x1:x2]
        plants_separated.append(plant_cut)
    
    return plants_separated, seedling_names


def realign_seedlings(plants_separated: List[np.ndarray], rot_from: int) -> List[np.ndarray]:
    """
    Realign each extracted seedling individually.
    
    Args:
        plants_separated: List of seedling image stacks
        rot_from: Frame where rotation was applied
    
    Returns:
        List of realigned seedling stacks
    """
    plants_separated_aligned = []
    
    for ix, each_plant in enumerate(tqdm(plants_separated, desc="Realigning seedlings")):
        first_part = each_plant[:rot_from]
        second_part = []
        
        im1, im2 = each_plant[rot_from - 1], each_plant[rot_from]
        # Trim borders to avoid wrong alignment
        shift_value = calculate_shift(im1[200:, 200:], im2[200:, 200:])
        
        for sl in range(rot_from, each_plant.shape[0]):
            second_part.append(shift_image(each_plant[sl], shift_value))
        
        second_part = np.stack(second_part, axis=0)
        plant_combined = np.concatenate([first_part, second_part], axis=0)
        plants_separated_aligned.append(plant_combined)
    
    return plants_separated_aligned


def save_seedling_tiles(plants_separated: List[np.ndarray], 
                       save_folder: Path, folder_name: str):
    """
    Save a tile view of all extracted seedlings.
    
    Args:
        plants_separated: List of seedling image stacks
        save_folder: Folder to save the tile image
        folder_name: Name for the saved file
    """
    print("Saving tile of extracted seedlings...")
    n = math.ceil(np.sqrt(len(plants_separated)))
    fig, ax = plt.subplots(nrows=n, ncols=n, figsize=(10, 10))
    ax = ax.ravel()
    
    for ix, a in enumerate(ax):
        if ix < len(plants_separated):
            a.imshow(plants_separated[ix][-1])
            a.set_title(f"seedling {ix+1}")
        a.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_folder / f"{folder_name}_seedlings_extracted.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def step_a_process_plate(folder_jpg: Optional[Path] = None, 
                         rot_from: Optional[int] = None,
                         seedling_coords: Optional[np.ndarray] = None,
                         interactive: bool = True) -> Tuple[Path, List[str]]:
    """
    STEP A: Complete plate processing pipeline.
    
    Args:
        folder_jpg: Path to folder with JPG images (if None, opens dialog)
        rot_from: Frame where rotation starts (if None and interactive, uses napari)
        seedling_coords: Coordinates for seedling extraction (if None and interactive, uses napari)
        interactive: Whether to use interactive napari selection
    
    Returns:
        Tuple of (save_folder, seedling_names)
    """
    print("\n" + "="*70)
    print("STEP A: PLATE PROCESSING AND SEEDLING EXTRACTION")
    print("="*70 + "\n")
    
    # Load images
    if folder_jpg is None:
        print("Select the folder containing JPG images...")
        folder_jpg = ask_folder()
    
    ims, timepoints, save_folder = load_images_from_folder(folder_jpg)
    
    # Crop to center
    imscut, pixel_size = center_crop_images(ims)
    del ims  # Free memory
    
    # Metadata
    metadata = {
        'timepoints': timepoints,
        'pixel_size_mm': pixel_size
    }
    
    # Determine rotation frame
    if rot_from is None:
        if interactive and NAPARI_AVAILABLE:
            print("\n>>> Opening napari: Click ONE point on the first rotated frame")
            print(">>> Then close the viewer window")
            viewer = napari.Viewer()
            viewer.add_image(imscut, name="plate_timeseries")
            rot_layer = viewer.add_points(
                name='rotation_start',
                face_color='red',
                size=50,
                ndim=3
            )
            napari.run()
            
            if len(rot_layer.data) > 0:
                rot_from = int(rot_layer.data[0][0])
                print(f"Rotation starts at frame: {rot_from}")
            else:
                raise ValueError("No point was marked for rotation frame")
        else:
            raise ValueError("rot_from must be provided in non-interactive mode")
    
    metadata['slide_rotated'] = rot_from
    
    # Correct rotation and shift
    ims_cut_corrected = correct_rotation_and_shift(
        imscut.copy(), rot_from, save_folder, folder_jpg.name
    )
    
    # Save corrected plate
    print("\nSaving corrected plate...")
    tifffile.imsave(
        file=save_folder / f"{folder_jpg.name}_rotated_reg.tif",
        data=np.uint8(ims_cut_corrected),
        imagej=True,
        metadata=metadata
    )
    
    # Extract seedlings
    if seedling_coords is None:
        if interactive and NAPARI_AVAILABLE:
            print("\n>>> Opening napari: Mark seedling corners (upper-left, lower-right pairs)")
            print(">>> Then close the viewer window")
            viewer = napari.Viewer()
            viewer.add_image(ims_cut_corrected[-1], name="last_frame")
            plants_layer = viewer.add_points(
                name='seedling_corners',
                face_color='red',
                size=50
            )
            napari.run()
            
            if len(plants_layer.data) > 0:
                seedling_coords = plants_layer.data
            else:
                raise ValueError("No points were marked for seedlings")
        else:
            raise ValueError("seedling_coords must be provided in non-interactive mode")
    
    plants_separated, seedling_names = extract_seedlings(
        ims_cut_corrected, imscut, seedling_coords
    )
    
    # Save tile
    save_seedling_tiles(plants_separated, save_folder, folder_jpg.name)
    
    # Realign individual seedlings
    plants_separated_aligned = realign_seedlings(plants_separated, rot_from)
    
    # Save individual seedlings
    print("\nSaving individual seedling TIF files...")
    for ix, plant in enumerate(tqdm(plants_separated_aligned, desc="Saving")):
        meta_plant = metadata.copy()
        meta_plant['seedling'] = seedling_names[ix]
        
        tifffile.imsave(
            file=save_folder / f"{folder_jpg.name}_seedling_{ix+1}.tif",
            data=np.uint8(plant),
            imagej=True,
            metadata=meta_plant
        )
    
    print(f"\n✓ Step A complete! Results saved to: {save_folder}")
    return save_folder, seedling_names


# ============================================================================
# STEP B: INDIVIDUAL SEEDLING TRACKING
# ============================================================================

def extract_metadata_from_tiff(tiff_path: Path) -> dict:
    """
    Extract metadata from a TIFF file's ImageDescription.
    
    Args:
        tiff_path: Path to TIFF file
    
    Returns:
        Dictionary of metadata
    """
    with tifffile.TiffFile(tiff_path) as tif:
        image_description = tif.pages[0].tags.get('ImageDescription')
        if image_description is None:
            return {}
        
        description_str = image_description.value
        metadata = {}
        
        for line in description_str.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                metadata[key.strip()] = value.strip()
        
        return metadata


def align_seedling(im: np.ndarray, rot: int, 
                  alignment_coords: Optional[np.ndarray] = None,
                  interactive: bool = True) -> np.ndarray:
    """
    Fine-tune alignment of a single seedling around rotation point.
    
    Args:
        im: Seedling image stack
        rot: Frame where rotation was applied
        alignment_coords: Two points defining a region for alignment calculation
        interactive: Whether to use napari for interactive selection
    
    Returns:
        Aligned image stack
    """
    if alignment_coords is None:
        if interactive and NAPARI_AVAILABLE:
            print("\n>>> Mark TWO points to define alignment region")
            viewer = napari.Viewer()
            viewer.add_image(im[rot-1], opacity=1, colormap='red', name='before_rot')
            viewer.add_image(im[rot], opacity=0.5, colormap='yellow', name='after_rot')
            points = viewer.add_points(name='subregion', face_color='red', size=25)
            napari.run()
            
            alignment_coords = points.data.astype(int)
        else:
            raise ValueError("alignment_coords required in non-interactive mode")
    
    y1, x1, y2, x2 = alignment_coords.ravel()
    reg1 = im[rot-1][y1:y2, x1:x2]
    reg2 = im[rot][y1:y2, x1:x2]
    shift = calculate_shift(reg1, reg2)
    
    first_part = im[:rot]
    second_part = []
    
    for sl in range(rot, im.shape[0]):
        second_part.append(shift_image(im[sl], shift))
    
    second_part = np.stack(second_part, axis=0)
    im_aligned = np.concatenate([first_part, second_part], axis=0)
    
    return im_aligned


def extract_tracking_points(plant_highlited_rot: np.ndarray,
                            tracking_coords: Optional[np.ndarray] = None,
                            interactive: bool = True) -> np.ndarray:
    """
    Extract tracking points from seedling time-series.
    
    Args:
        plant_highlited_rot: Rotated and edge-enhanced seedling stack
        tracking_coords: Pre-defined tracking points
        interactive: Whether to use napari
    
    Returns:
        Array of tracking points (time, y, x)
    """
    if tracking_coords is None:
        if interactive and NAPARI_AVAILABLE:
            print("\n>>> Mark tracking points (one per frame, following growth)")
            viewer = napari.Viewer()
            viewer.add_image(plant_highlited_rot, name='seedling')
            path_layer = viewer.add_points(
                name='path',
                face_color='red',
                symbol='square',
                size=1,
                ndim=3
            )
            napari.run()
            
            tracking_coords = path_layer.data
        else:
            raise ValueError("tracking_coords required in non-interactive mode")
    
    return tracking_coords


def calculate_tracking_metrics(data_df: pd.DataFrame, pixel_size: float) -> pd.DataFrame:
    """
    Calculate angles, distances, and speeds between tracking points.
    
    Args:
        data_df: DataFrame with columns ['time', 'y', 'x', 'y_mm', 'x_mm']
        pixel_size: Size of each pixel in mm
    
    Returns:
        DataFrame with calculated metrics
    """
    angle_of_line = lambda p1, p2: math.degrees(
        math.atan2(p2['y'] - p1['y'], p2['x'] - p1['x'])
    )
    
    middle_point = lambda p1, p2: (
        (p1['y'] + (p2['y'] - p1['y']) / 2),
        (p1['x'] + (p2['x'] - p1['x']) / 2)
    )
    
    euclidean_dist = lambda p1, p2: distance.euclidean(
        (p1['y_mm'], p1['x_mm']),
        (p2['y_mm'], p2['x_mm'])
    )
    
    middles = defaultdict(list)
    
    for i in data_df.index[:-1]:
        p1 = data_df.loc[i]
        p2 = data_df.loc[i+1]
        
        middle = middle_point(p1, p2)
        middles['caps'].append(f"{int(p1['cap'])}-{int(p2['cap'])}")
        middles['y_degrees_label'].append(middle[0])
        middles['x_degrees_label'].append(middle[1])
        
        time_dif = (p2['time'] - p1['time']).total_seconds()
        degree = angle_of_line(p1, p2)
        euclidean = euclidean_dist(p1, p2)
        
        middles['degree'].append(degree)
        middles['euclidean (mm)'].append(euclidean)
        middles['time_dif (s)'].append(time_dif)
        middles['speed (mm/s)'].append(euclidean / time_dif if time_dif > 0 else 0)
    
    df_results = pd.DataFrame(middles)
    df_results['speed (mm/h)'] = df_results['speed (mm/s)'] * 3600
    
    return df_results


def plot_tracking_degrees(data_df: pd.DataFrame, df_results: pd.DataFrame,
                         seedling_name: str, pixel_size: float, 
                         save_folder: Path):
    """
    Create and save a plot showing tracking points with angle annotations.
    
    Args:
        data_df: Raw tracking data
        df_results: Calculated metrics
        seedling_name: Name of the seedling
        pixel_size: Pixel size in mm
        save_folder: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot tracking points
    timepoints = data_df['time'].tolist()
    for name, group in data_df.groupby(['cap']):
        plt.scatter(group.x, group.y, label=timepoints[int(name)])
    
    plt.plot(data_df.x, data_df.y, color='grey', linestyle='--')
    
    # Add degree annotations
    for ix, row in df_results.iterrows():
        plt.text(
            x=row['x_degrees_label'] - 2,
            y=row['y_degrees_label'] + 2,
            s=f"{math.ceil(row['degree'])}°",
            size=12
        )
    
    plt.ylabel("distance (mm)")
    plt.xlabel("distance (mm)")
    plt.ylim(data_df.y.max() + 10, data_df.y.min() - 10)
    
    # Convert axes to mm
    xlocs, _ = plt.xticks()
    xnewlabels = [round(i * pixel_size) for i in xlocs]
    plt.xticks(ticks=xlocs, labels=xnewlabels, rotation=0)
    
    ylocs, _ = plt.yticks()
    ynewlabels = [round(i * pixel_size) for i in ylocs]
    plt.yticks(ticks=ylocs, labels=ynewlabels, rotation=0)
    
    # Legend (first and last timepoint only)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[0], handles[-1]],
        [labels[0], labels[-1]],
        loc="upper left",
        bbox_to_anchor=(1.04, 1)
    )
    
    plt.title(seedling_name)
    plt.grid(axis='both')
    
    plt.savefig(save_folder / f"{seedling_name}_lateral_view_degrees.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(save_folder / f"{seedling_name}_lateral_view_degrees.svg",
                dpi=300, bbox_inches='tight')
    plt.close()


def step_b_track_seedling(seedling_tifpath: Optional[Path] = None,
                         alignment_coords: Optional[np.ndarray] = None,
                         tracking_coords: Optional[np.ndarray] = None,
                         interactive: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    STEP B: Track a single seedling and extract growth metrics.
    
    Args:
        seedling_tifpath: Path to seedling TIF file
        alignment_coords: Coordinates for alignment (optional)
        tracking_coords: Pre-defined tracking points (optional)
        interactive: Whether to use napari
    
    Returns:
        Tuple of (raw_data_df, results_df)
    """
    print("\n" + "="*70)
    print("STEP B: SEEDLING TRACKING")
    print("="*70 + "\n")
    
    # Select file
    if seedling_tifpath is None:
        print("Select a seedling TIF file...")
        seedling_tifpath = ask_file()
    
    seedling_name = seedling_tifpath.stem
    save_folder = seedling_tifpath.parent
    
    # Load image
    print(f"Loading {seedling_name}...")
    seedling_im = tifffile.imread(seedling_tifpath)
    
    # Extract metadata
    metadata = extract_metadata_from_tiff(seedling_tifpath)
    rot = int(metadata.get('slide_rotated', 0))
    pixel_size = float(metadata.get('pixel_size_mm', 0.036))
    
    timepoints_str = metadata.get('timepoints', '').replace("[", "").replace("]", "")
    timepoints_str = timepoints_str.replace(" ", "").replace("'", "").split(",")
    timepoints = [datetime.datetime.strptime(t, "%Y%m%d_%H%M%S") for t in timepoints_str]
    
    # Fine alignment
    print("Performing fine alignment...")
    seedling_im = align_seedling(seedling_im, rot, alignment_coords, interactive)
    
    # Highlight edges and rotate
    print("Highlighting edges...")
    plant_highlighted = filters.sobel(seedling_im)
    plant_highlighted_rot = np.rot90(plant_highlighted, axes=(1, 2))
    
    # Extract tracking points
    tracking_data = extract_tracking_points(plant_highlighted_rot, tracking_coords, interactive)
    
    # Create data DataFrame
    data_df = pd.DataFrame(tracking_data, columns=['point', 'y', 'x'])
    data_df.insert(0, 'cap', data_df.index)
    data_df.insert(0, 'time', pd.Series(timepoints))
    data_df.insert(0, 'seedling_name', seedling_name)
    data_df['x_mm'] = data_df.x * pixel_size
    data_df['y_mm'] = data_df.y * pixel_size
    
    # Calculate metrics
    print("Calculating metrics...")
    df_results = calculate_tracking_metrics(data_df, pixel_size)
    
    # Save results
    print(f"Saving results to {save_folder / seedling_name}.xlsx")
    
    if CUSTOM_TOOLS_AVAILABLE:
        mt.writting_excel(
            DF=[data_df, df_results],
            pathname=save_folder / f"{seedling_name}.xlsx",
            sheet_name=['raw_data', 'degrees']
        )
    else:
        # Fallback: use pandas ExcelWriter
        with pd.ExcelWriter(save_folder / f"{seedling_name}.xlsx") as writer:
            data_df.to_excel(writer, sheet_name='raw_data')
            df_results.to_excel(writer, sheet_name='degrees')
    
    # Plot
    print("Creating visualization...")
    plot_tracking_degrees(data_df, df_results, seedling_name, pixel_size, save_folder)
    
    print(f"\n✓ Step B complete! Results saved to: {save_folder}")
    return data_df, df_results


# ============================================================================
# STEP C: BATCH ANALYSIS
# ============================================================================

def load_all_seedling_results(results_folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine all seedling Excel results from a folder.
    
    Args:
        results_folder: Folder containing seedling result Excel files
    
    Returns:
        Tuple of (combined_raw_data, combined_degrees_data)
    """
    excel_files = sorted(list(results_folder.glob("**/*.xlsx")))
    
    if not excel_files:
        raise ValueError(f"No Excel files found in {results_folder}")
    
    print(f"Found {len(excel_files)} result files")
    
    names = [f"{f.parent.name}_{f.stem}" for f in excel_files]
    
    dfs_degrees = []
    dfs_data = []
    
    for idx, excel_file in enumerate(tqdm(excel_files, desc="Loading results")):
        try:
            df_deg = pd.read_excel(excel_file, sheet_name="degrees", index_col=0)
            df_raw = pd.read_excel(excel_file, sheet_name="raw_data", index_col=0)
            
            df_deg.insert(0, 'experiment', names[idx])
            df_raw.insert(0, 'experiment', names[idx])
            
            dfs_degrees.append(df_deg)
            dfs_data.append(df_raw)
        except Exception as e:
            print(f"Warning: Could not load {excel_file}: {e}")
    
    df_degrees_combined = pd.concat(dfs_degrees, ignore_index=True)
    df_data_combined = pd.concat(dfs_data, ignore_index=True)
    
    # Format caps with leading zeros
    df_degrees_combined['caps'] = df_degrees_combined['caps'].apply(
        lambda x: "-".join([x.split("-")[0].zfill(3), x.split("-")[1].zfill(3)])
    )
    df_data_combined['cap'] = df_data_combined['cap'].apply(lambda x: str(x).zfill(3))
    
    return df_data_combined, df_degrees_combined


def create_batch_visualizations(df_data: pd.DataFrame, df_degrees: pd.DataFrame,
                                save_folder: Path):
    """
    Create comprehensive visualizations for batch analysis.
    
    Args:
        df_data: Combined raw tracking data
        df_degrees: Combined degree metrics
        save_folder: Where to save plots
    """
    print("\nCreating batch visualizations...")
    
    # Add timepoints to degrees dataframe
    timepoints = df_data[
        df_data.cap.isin(df_degrees.caps.apply(lambda x: x.split("-")[-1]))
    ].reset_index()['time']
    df_degrees.insert(2, 'timepoints', timepoints.values)
    
    # Extract experiment type
    df_degrees.insert(0, 'type', df_degrees.experiment.apply(lambda x: x.split("_")[0]))
    df_data.insert(0, 'type', df_data.experiment.apply(lambda x: x.split("_")[0]))
    
    # 1. Heatmap by caps
    plt.figure(figsize=(20, 10))
    pivot_caps = df_degrees.pivot("experiment", "caps", "degree")
    sns.heatmap(pivot_caps, cmap='viridis')
    plt.title("Growth Angles Heatmap (by capture)")
    plt.tight_layout()
    plt.savefig(save_folder / "heatmap_caps.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap by timepoints
    plt.figure(figsize=(20, 10))
    pivot_time = df_degrees.pivot("experiment", "timepoints", "degree")
    sns.heatmap(pivot_time, cmap='viridis')
    plt.title("Growth Angles Heatmap (by time)")
    plt.tight_layout()
    plt.savefig(save_folder / "heatmap_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Density plot
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.kdeplot(data=df_degrees, x="degree", hue='type', fill=True)
    plt.title("Density Distribution of Growth Angles")
    plt.xlabel("Angle (degrees)")
    plt.tight_layout()
    plt.savefig(save_folder / "density_degrees.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Point plot
    g = sns.catplot(
        kind='point',
        data=df_degrees,
        x='caps',
        y='degree',
        row='type',
        aspect=3,
        height=4
    )
    g.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig(save_folder / "pointplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Visualizations saved")


def step_c_batch_analysis(results_folder: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    STEP C: Combine and analyze multiple seedling results.
    
    Args:
        results_folder: Folder containing multiple seedling Excel results
    
    Returns:
        Tuple of (combined_raw_data, combined_degrees_data)
    """
    print("\n" + "="*70)
    print("STEP C: BATCH ANALYSIS")
    print("="*70 + "\n")
    
    if results_folder is None:
        print("Select the folder containing seedling results...")
        results_folder = ask_folder()
    
    # Load all results
    df_data, df_degrees = load_all_seedling_results(results_folder)
    
    # Save combined data
    save_path = results_folder.parent / f"{results_folder.stem.replace(' ', '_')}.xlsx"
    print(f"\nSaving combined data to {save_path}")
    
    if CUSTOM_TOOLS_AVAILABLE:
        mt.writting_excel(
            [df_data, df_degrees],
            pathname=save_path,
            sheet_name=['data', 'degrees']
        )
    else:
        with pd.ExcelWriter(save_path) as writer:
            df_data.to_excel(writer, sheet_name='data')
            df_degrees.to_excel(writer, sheet_name='degrees')
    
    # Create visualizations
    create_batch_visualizations(df_data, df_degrees, results_folder)
    
    print(f"\n✓ Step C complete! Results saved to: {results_folder}")
    return df_data, df_degrees


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Seedling Segmentation and Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline interactively
  python SeedlingNew.py --step all
  
  # Run only Step A (plate processing)
  python SeedlingNew.py --step A --folder /path/to/jpg/folder
  
  # Run only Step B (single seedling tracking)
  python SeedlingNew.py --step B --file /path/to/seedling.tif
  
  # Run only Step C (batch analysis)
  python SeedlingNew.py --step C --folder /path/to/results
  
  # Non-interactive mode (requires all parameters)
  python SeedlingNew.py --step A --folder /path/to/jpg --rot-frame 119 --no-interactive
        """
    )
    
    parser.add_argument(
        '--step',
        choices=['A', 'B', 'C', 'all'],
        required=True,
        help='Which step to run (A: plate processing, B: tracking, C: batch analysis, all: run all)'
    )
    
    parser.add_argument(
        '--folder',
        type=Path,
        help='Input folder path (for steps A and C)'
    )
    
    parser.add_argument(
        '--file',
        type=Path,
        help='Input file path (for step B)'
    )
    
    parser.add_argument(
        '--rot-frame',
        type=int,
        help='Frame where rotation starts (for step A)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=True,
        help='Use interactive napari selection (default: True)'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_false',
        dest='interactive',
        help='Disable interactive mode'
    )
    
    args = parser.parse_args()
    
    # Check napari availability
    if args.interactive and not NAPARI_AVAILABLE:
        print("Warning: napari not available. Falling back to non-interactive mode.")
        args.interactive = False
    
    try:
        if args.step == 'A' or args.step == 'all':
            save_folder, seedling_names = step_a_process_plate(
                folder_jpg=args.folder,
                rot_from=args.rot_frame,
                interactive=args.interactive
            )
            
            if args.step == 'all':
                print("\nContinuing to Step B...")
        
        if args.step == 'B' or args.step == 'all':
            data_df, results_df = step_b_track_seedling(
                seedling_tifpath=args.file,
                interactive=args.interactive
            )
            
            if args.step == 'all':
                print("\nContinuing to Step C...")
        
        if args.step == 'C' or args.step == 'all':
            df_data, df_degrees = step_c_batch_analysis(
                results_folder=args.folder
            )
        
        print("\n" + "="*70)
        print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
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

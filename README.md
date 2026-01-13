# Seedling Tracking Pipeline

A streamlined Python script for processing time-lapse TIF stacks of seedlings and tracking growth dynamics.

## Overview

This simplified pipeline processes TIF stacks containing multiple seedlings over time:

- **Step 1**: Load TIF stack, align all timeframes, save to `2_Processed/`
- **Step 2**: User draws rectangles to separate individual seedlings, save to `2_Processed/`
- **Step 3**: User tracks growth by clicking points in each timeframe, save results to `3_Results/`

## Features

- 🔄 Automatic timeframe alignment using phase cross-correlation
- 🌱 Interactive seedling separation with napari rectangles
- 📊 Growth angle and distance calculations
- 📈 Visual tracking plots with angle annotations
- 💾 Export to Excel with tracking data and metrics
- 🖥️ Simple menu-driven interface

## Requirements

### System Requirements

- Python 3.9+
- Linux, macOS, or Windows
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Python Dependencies

```toml
dependencies = [
    "numpy>=2.0",
    "pandas>=2.0",
    "matplotlib>=3.8",
    "scikit-image>=0.26",
    "scipy>=1.10",
    "tifffile>=2023",
    "tqdm>=4.65",
    "napari[all]>=0.6",  # For interactive mode
    "openpyxl>=3.1",     # For Excel export
]
```

## Folder Structure

The script expects and creates the following folder structure:

```
YourProject/
├── 1_Data/         # Place your raw TIF stacks here (INPUT)
├── 2_Processed/    # Aligned stacks and separated seedlings (AUTO-CREATED)
├── 3_Results/      # Excel files and plots (AUTO-CREATED)
└── SeedlingNew.py  # The script
```

## Installation

### Ubuntu (Linux)

#### One-Step Installation (Fastest)

If you have `uv` installed, you can set up the entire environment with a single command:

```bash
uv sync
source .venv/bin/activate
```

#### Manual Installation with uv

1. **Install uv** (if not already installed):
   
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Navigate to project directory:**
   
   ```bash
   cd /path/to/your/project
   ```

3. **Create and activate virtual environment:**
   
   ```bash
   uv venv
   source .venv/bin/activate
   ```

4. **Install dependencies:**
   
   ```bash
   uv pip install numpy pandas matplotlib scikit-image scipy tifffile tqdm "napari[all]" openpyxl
   ```

#### Using pip

1. **Navigate to project directory:**
   
   ```bash
   cd /path/to/your/project
   ```

2. **Create virtual environment:**
   
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   
   ```bash
   pip install numpy pandas matplotlib scikit-image scipy tifffile tqdm "napari[all]" openpyxl
   ```

### Windows

#### One-Step Installation (Fastest)

If you have `uv` installed, you can set up the entire environment with a single command:

```powershell
uv sync
.venv\Scripts\activate
```

#### Manual Installation with uv

1. **Install uv** (if not already installed):
   Open PowerShell and run:
   
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Navigate to project directory:**
   
   ```powershell
   cd \path\to\your\project
   ```

3. **Create and activate virtual environment:**
   
   ```powershell
   uv venv
   .venv\Scripts\activate
   ```

4. **Install dependencies:**
   
   ```powershell
   uv pip install numpy pandas matplotlib scikit-image scipy tifffile tqdm "napari[all]" openpyxl
   ```

#### Using pip

1. **Navigate to project directory:**
   
   ```powershell
   cd \path\to\your\project
   ```

2. **Create virtual environment:**
   
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   
   ```powershell
   pip install numpy pandas matplotlib scikit-image scipy tifffile tqdm "napari[all]" openpyxl
   ```

## Usage

### Quick Start

**Using uv (Recommended):**

This works on all platforms (Linux, macOS, Windows) and automatically handles the environment.

```bash
uv run SeedlingNew.py
```

**Standard Python (Alternative):**

If you prefer manual activation:

**Ubuntu/Linux/macOS:**

```bash
# Activate your environment
source .venv/bin/activate

# Run the script
python SeedlingNew.py
```

**Windows:**

```powershell
# Activate your environment
.venv\Scripts\activate

# Run the script
python SeedlingNew.py
```

The script will present a menu:

### Trying it out with Dummy Data

If you don't have your own TIF stacks yet, you can generate synthetic data to test the pipeline:

1. **Generate the dummy data:**
   
   ```bash
   uv run generate_sample_data.py
   ```
   
   This creates `1_Data/dummy.tif` simulating a growing seedling.

2. **Run the pipeline:**
   
   ```bash
   uv run SeedlingNew.py
   ```
   
   - Select **Option 4** (Full Pipeline)
   - When prompted for a file, select `1_Data/dummy.tif`
   - Follow the instructions to separate and track the seedling.

```
======================================================================
SELECT WORKFLOW:
  1 - Load and align TIF stack
  2 - Separate seedlings from aligned stack
  3 - Track individual seedling
  4 - Run full pipeline (1 → 2 → 3)
======================================================================
```

### Workflow Options

#### Option 1: Load and Align TIF Stack

Loads a TIF stack from `1_Data/` and aligns all timeframes.

```bash
# Select option 1
# Browse to select your TIF stack from 1_Data/ folder
# Script will save: 2_Processed/{filename}_aligned.tif
```

**What it does:**

1. Prompts you to select a TIF file from `1_Data/`
2. Detects if stack is 3D (t, y, x) or 4D (t, y, x, channels)
3. Uses frame 0 as reference for alignment
4. Aligns all other frames using phase cross-correlation
5. Saves aligned stack to `2_Processed/`

---

#### Option 2: Separate Seedlings

Opens napari to draw rectangles around seedlings in an aligned stack.

```bash
# Select option 2
# Browse to select an aligned TIF from 2_Processed/
# Draw rectangles in napari
# Close napari when done
```

**napari Instructions:**

1. Use the **Shapes** layer (should be active by default)
2. Draw ONE rectangle around each seedling
3. You can draw as many rectangles as you have seedlings
4. Close the napari window when finished

**What it does:**

1. Opens napari with max projection and timeseries
2. Extracts each rectangle region
3. Saves separate TIF files: `2_Processed/{filename}_aligned_seedling_1.tif`, etc.

---

#### Option 3: Track Individual Seedling

Opens napari to track growth by clicking points in each timeframe.

```bash
# Select option 3
# Browse to select a seedling TIF from 2_Processed/
# Enter pixel size (e.g., 0.036 mm)
# Choose whether to use edge enhancement (recommended: y)
# Click tracking points in napari
# Close napari when done
```

**napari Instructions:**

1. Click ONE point per timeframe
2. Track a consistent landmark (e.g., root tip, shoot apex)
3. Work through frames sequentially (0, 1, 2, ...)
4. Be as consistent as possible with point placement

**What it does:**

1. Loads seedling TIF
2. Optionally applies Sobel edge enhancement
3. Rotates image 90° for lateral view
4. Calculates angles and distances between consecutive points
5. Saves Excel file: `3_Results/{seedling_name}.xlsx`
6. Creates plots: `3_Results/{seedling_name}_tracking.png` and `.svg`

**Excel Output:**

- **raw_data sheet**: timepoint, y, x, y_mm, x_mm
- **metrics sheet**: from_timepoint, to_timepoint, segment, angle_degrees, distance_mm

---

#### Option 4: Full Pipeline

Runs all three steps sequentially on one TIF stack.

```bash
# Select option 4
# Follow prompts for each step:
#   1. Select TIF from 1_Data/
#   2. Draw rectangles in napari
#   3. Enter pixel size
#   4. Track each seedling in napari
```

---

### Example Session

```bash
$ uv run SeedlingNew.py

======================================================================
SEEDLING TRACKING PIPELINE
======================================================================

Folder structure:
  Data:      /path/to/project/1_Data
  Processed: /path/to/project/2_Processed
  Results:   /path/to/project/3_Results

======================================================================
SELECT WORKFLOW:
  1 - Load and align TIF stack
  2 - Separate seedlings from aligned stack
  3 - Track individual seedling
  4 - Run full pipeline (1 → 2 → 3)
======================================================================

Enter choice (1-4): 4

======================================================================
RUNNING FULL PIPELINE
======================================================================

Select TIF stack from 1_Data/ folder...
# [File dialog opens - select Exp_Plate.tif]

======================================================================
STEP 1: LOADING AND ALIGNING TIF STACK
======================================================================

Loading: Exp_Plate.tif
Stack shape: (50, 1024, 1024)
Data type: uint8
Number of timeframes: 50

Aligning all frames to frame 0...
Aligning: 100%|████████████████████| 50/50 [00:15<00:00,  3.2it/s]

Saving aligned stack to: Exp_Plate_aligned.tif
✓ Step 1 complete!

======================================================================
STEP 2: SEPARATING SEEDLINGS
======================================================================

Opening napari viewer...

Instructions:
  1. Use the 'Shapes' layer to draw rectangles around each seedling
  2. Draw ONE rectangle per seedling
  3. Close the viewer when done

# [napari opens - draw 3 rectangles]
# [Close napari]

Found 3 seedlings
  Seedling 1: y=[100:300], x=[50:200]
  Seedling 2: y=[100:300], x=[250:400]
  Seedling 3: y=[350:550], x=[50:200]

✓ Step 2 complete! Saved 3 seedlings to 2_Processed/

Enter pixel size in mm (default 0.036): 0.036
Use edge enhancement? (y/n, default y): y

======================================================================
STEP 3: TRACKING SEEDLING - Exp_Plate_aligned_seedling_1
======================================================================

Loading: Exp_Plate_aligned_seedling_1.tif
Applying Sobel edge enhancement...

Opening napari viewer...
# [Click points in each frame]
# [Close napari]

Tracked 50 points
Calculating growth metrics...

Saving results to: Exp_Plate_aligned_seedling_1.xlsx
Creating visualization...
✓ Step 3 complete for Exp_Plate_aligned_seedling_1!

# [Repeats for seedling_2 and seedling_3]

======================================================================
✓ FULL PIPELINE COMPLETE!
======================================================================
```

---

## File Naming Convention

### After Step 1 (Alignment):

```
2_Processed/
└── {original_name}_aligned.tif
```

### After Step 2 (Separation):

```
2_Processed/
├── {original_name}_aligned.tif
├── {original_name}_aligned_seedling_1.tif
├── {original_name}_aligned_seedling_2.tif
└── ...
```

### After Step 3 (Tracking):

```
3_Results/
├── {seedling_name}.xlsx
├── {seedling_name}_tracking.png
└── {seedling_name}_tracking.svg
```

---

## Tips and Best Practices

### Image Acquisition

- Use consistent lighting across timepoints
- Keep camera position fixed
- Capture at regular time intervals
- Avoid shadows and reflections

### Alignment (Step 1)

- The script uses frame 0 as reference by default
- Works best when most of the field of view is stable
- If alignment fails, check for extreme movements or rotation

### Seedling Separation (Step 2)

- Draw rectangles generously - include some margin around each seedling
- Make sure seedlings don't overlap in the rectangles
- You can draw rectangles in any order
- The max projection helps identify where seedlings are located

#### napari Rectangle Tool

The **Rectangle** tool is selected by default when the separation window opens. Here's how to use it:

- **Pan/Move Image**: Click and drag with the mouse (or hold Space + drag)
- **Zoom In/Out**: Use the scroll wheel (up = zoom in, down = zoom out)
- **Draw Rectangle**: Click and drag to create a rectangle (tool is active by default)
  - Keyboard shortcut: Press `R` to reactivate rectangle mode
- **Select/Edit Rectangles**: Press `D` to switch to direct/select mode
  - Click on a rectangle to select it
  - Drag corners or edges to resize
  - Drag the center to move the entire rectangle
  - Press Delete or Backspace to remove selected rectangles
- **Switch Between Tools**: Use keyboard shortcuts `R` (rectangle) and `D` (select) to switch between modes

**Workflow tip**: Use the max projection view to identify all seedlings first, then draw rectangles around each one. You can zoom/pan and switch between drawing and selecting as needed.

### Tracking (Step 3)

- **Edge enhancement**: Highly recommended for better visibility of growth tips
- **Consistent landmarks**: Always click the same anatomical feature (e.g., root tip)
- **Order matters**: Click points in chronological order (frame 0, 1, 2, ...)
- **Accuracy**: Take your time - accurate clicking = better angle calculations
- **If you mess up**: Just close napari and restart Step 3 for that seedling

#### napari Navigation and Tools

The **Add Points** tool is selected by default when the tracking window opens. Here's how to navigate:

- **Pan/Move Image**: Click and drag with the mouse (or hold Space + drag)
- **Zoom In/Out**: Use the scroll wheel (up = zoom in, down = zoom out)
- **Add Points**: Tool is active by default - simply click to add points
  - Keyboard shortcut: Press `2` to reactivate add mode
- **Select/Move Points**: Press `3` to switch to select mode
  - Drag points to reposition them
  - Press Delete or Backspace to remove selected points
- **Switch Between Tools**: Use keyboard shortcuts `2` (add) and `3` (select) to switch between modes

**Workflow tip**: Navigate between frames using the slider, zoom/pan to the growth point, ensure add mode is active (press `2` if needed), then click to mark the landmark. Repeat for each timeframe.

### Pixel Size

- Default is 0.036 mm/pixel
- Measure your actual pixel size using a calibration ruler
- This affects distance calculations in mm

---

## Troubleshooting

### "napari not available"

```bash
# Install Qt backend
uv pip install pyqt5

# Or try PySide6
uv pip install pyside6

# Then reinstall napari
uv pip install "napari[all]"
```

### napari window is blank or frozen

```bash
# Try a different Qt backend
pip uninstall pyqt5 pyside6
pip install pyside6
```

### "Expected 3D or 4D stack" error

Your TIF file might not be a time-series stack. Check:

```python
import tifffile
stack = tifffile.imread("your_file.tif")
print(stack.shape)  # Should be (time, height, width) or (time, height, width, channels)
```

### Alignment produces weird results

- Check that your stack actually needs alignment (maybe it's already aligned?)
- Try aligning to a different reference frame (edit the script)
- Some stacks have too much movement for phase cross-correlation

### Excel export fails

```bash
uv pip install openpyxl
```

### Out of memory errors

- Process seedlings one at a time (use options 1, 2, 3 separately)
- Reduce TIF stack size before processing
- Close other applications

---

## Advanced Usage

### Processing Multiple Files

```bash
#!/bin/bash
# process_all.sh

for tif in 1_Data/*.tif; do
    echo "Processing $tif"
    uv run SeedlingNew.py << EOF
1
EOF
done
```

### Customizing Pixel Size

Edit the script to change the default:

```python
# Around line 300 in track_seedling_interactive():
pixel_size_mm: float = 0.036,  # Change this value
```

### Changing Reference Frame

Edit the script to align to a different frame:

```python
# In load_and_align_stack():
reference_frame: int = 0,  # Change to your desired frame number
```

---

## Output Format

### Excel File Structure

**raw_data sheet:**
| seedling_name | timepoint | y | x | x_mm | y_mm |
|---------------|-----------|-----|-----|------|------|
| seedling_1    | 0         | 150 | 80  | 2.88 | 5.40 |
| seedling_1    | 1         | 148 | 82  | 2.95 | 5.33 |
| ...           | ...       | ... | ... | ...  | ...  |

**metrics sheet:**
| from_timepoint | to_timepoint | segment | angle_degrees | distance_mm | mid_y | mid_x |
|----------------|--------------|---------|---------------|-------------|-------|-------|
| 0              | 1            | 0-1     | 35.5          | 0.082       | 149.0 | 81.0  |
| 1              | 2            | 1-2     | 42.3          | 0.095       | 147.5 | 82.5  |
| ...            | ...          | ...     | ...           | ...         | ...   | ...   |

### Plot Format

- PNG: High-resolution bitmap (300 DPI) for presentations
- SVG: Vector format for publications (scalable, editable)
- Shows trajectory with points labeled by timepoint
- Angle annotations at midpoints between consecutive points
- Axes in mm (converted from pixels)

---

## Citation

If you use this pipeline in your research, please cite:

```
[Your Lab/Publication]
Seedling Tracking Pipeline v5.0
```

---

## Changelog

### Version 5.0 (Simplified - Current)

- Completely redesigned for simpler workflow
- Direct TIF stack processing (no JPG conversion needed)
- Menu-driven interface
- Integrated alignment, separation, and tracking
- Automatic folder management (1_Data/, 2_Processed/, 3_Results/)
- Removed batch analysis (focus on individual seedlings)
- Removed rotation correction (not needed for most TIF stacks)

### Previous Versions

- v4.0: Consolidated three scripts (A, B, C)
- v3.A: Plate processing
- v3.B: Seedling tracking
- v3.C: Batch analysis

---

## Support

For questions or issues:

- Check the troubleshooting section above
- Review the example session
- Ensure all dependencies are installed correctly

---

## License

[Specify your license]

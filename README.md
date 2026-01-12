# BigTank

Multi-camera calibration pipeline for overhead camera arrays using ChArUco boards. Built on aniposelib with enhanced 
error metrics, ground-plane reorientation, and comprehensive visualization tools. Calibration outputs in aniposelib
format for compatibility with downstream tools. Allows for easy manipulation of certain hard-coded aniposelib parameters
that, when left at the defaults, can cause calibration failure (especially for setups where the calibration board 
cannot be seen by every camera simultaneously). 

## Features

- **Robust calibration**: Enhanced intrinsic calibration requiring ≥12 detected corners (vs. anipose's ≥9) for more stable results (see supplementary section)
- **Ground-plane reorientation**: Automatically reorients coordinate frame so origin is at board center with XY plane as ground plane and Z-axis pointing upward
- **Dual-metric validation**: Bundle adjustment error + per-frame RANSAC validation for comprehensive quality assessment
- **Rich visualizations**: 3D camera extrinsics plots and per-frame reprojection error heatmaps

## Installation

```bash
pip install aniposelib opencv-python numpy matplotlib toml
```

For Python 3.11+, `tomllib` is built-in. For earlier versions, install `toml`.

## Quick Start

```bash
# Run calibration with default settings
python main.py projects/calibration_121525

# Skip visualizations for faster processing
python main.py projects/calibration_121525 --no-visualizations

# Adjust frame sampling for error analysis
python main.py projects/calibration_121525 --frame-step 5

# Quiet mode with no board image
python main.py projects/calibration_121525 --quiet --no-board-image
```

## Usage

### Command Line Interface

```bash
python main.py <project_folder> [options]
```

**Arguments:**
- `project_folder`: Path to project folder (e.g., `projects/calibration_121525`)

**Options:**
- `--no-board-image`: Skip generating board image PNG
- `--no-visualizations`: Skip generating visualization plots (faster)
- `--quiet`: Suppress verbose output during calibration
- `--video-extension`: Video file extension (default: from config or "avi")
- `--frame-step N`: Sample every Nth frame for error analysis (default: 10). Lower = more thorough but slower
- `--frustum-scale N`: Camera frustum scale for 3D visualization (default: 200)

### Project Structure

A completed calibration project will have the following structure. Files/folders marked required must be
present prior to attempting calibration:

```
projects/calibration_121525/
├── config.toml                      # Configuration file (required)
├── videos/                          # Calibration videos (required)
│   ├── e3v8250-20251215T133038-133806.avi
│   ├── e3v829d-20251215T133038-133806.avi
│   ├── e3v82e0-20251215T133038-133806.avi
│   └── ...
└── output/                          # Generated outputs (created automatically)
    ├── calibration.toml             # Standard calibration (world frame arbitrary)
    ├── calibration_reoriented.toml  # Ground-plane calibration (Z-up, origin at board)
    ├── calibration_summary.txt      # Human-readable calibration summary
    ├── board.png                    # Generated board image for printing
    └── visualizations/
        ├── camera_extrinsics.png    # 3D plot of camera positions
        └── reprojection_error_heatmap.png  # Per-frame error heatmap
```

### Configuration File

Create a `config.toml` file in your project folder. This is the same config.toml format used by anipose, so see
the anipose documentation for more information.

```toml
# name of the project
project = "calibration_121525"
model_folder = ''

nesting = 1

video_extension = 'avi'

[calibration]
# Board type: checkerboard / charuco / aruco
board_type = "charuco"

# Grid dimensions (width x height)
board_size = [12, 9]

# ArUco dictionary parameters (if using charuco/aruco)
board_marker_bits = 5           # Number of bits in markers
board_marker_dict_number = 100  # Dictionary size (50, 100, 250, 1000)
board_marker_length = 45        # Marker side length (mm)

# Square/separation length
board_square_side_length = 60   # Square side length for charuco/checkerboard (mm)
# board_marker_separation_length = 1  # Marker separation for aruco (mm)

# Camera model
fisheye = false

[triangulation]
cam_regex = '([^-]+)'

[manual_verification]
manually_verify = true
```

**Key parameters:**

- `board_type`: Type of calibration board (`"charuco"`, `"checkerboard"`, or `"aruco"`)
- `board_size`: [width, height] of the board grid
- `board_marker_length`: Physical size of ArUco markers in mm
- `board_square_side_length`: Physical size of checkerboard squares in mm
- `fisheye`: Set to `true` for fisheye lens models
- `cam_regex`: Regex pattern to extract camera name from video filename

### Video Requirements

- For reorientation to work, end the video with a period of at least three seconds where the board is placed flat on the ground plane, centered where you want the origin
- Videos should show the calibration board from multiple angles
- the board does NOT need to be in frame for every camera at all times, but must enter the FOV of each camera at some point in the video.
- All cameras should be synchronized and capture the same calibration sequence
- Video filenames should contain camera identifiers (extracted via `cam_regex`)

## Calibration Process

The pipeline performs the following steps:

1. **Configuration loading**: Parse `config.toml` and validate project structure
2. **Board generation**: Create calibration board image for printing (optional)
3. **Video processing**: Detect ChArUco corners in all frames across all cameras
4. **Intrinsic calibration**: Calibrate each camera individually using detected corners
   - **Enhanced filtering**: Requires ≥12 detected corners per frame (vs. anipose's ≥9)
5. **Extrinsic calibration**: Estimate relative camera positions via bundle adjustment
6. **Per-frame validation**: Compute RANSAC-validated reprojection errors across video frames
7. **Ground-plane reorientation**: Transform coordinate frame to board-centric axes
   - Analyzes final 3 seconds of videos to locate board on ground
   - Reorients so origin is at board center, XY plane is ground, Z points upward
8. **Visualization generation**: Create 3D plots and error heatmaps (optional)

### Coordinate Frames

The calibration produces two outputs with different coordinate systems:

#### Standard Calibration (`calibration.toml`)
- World frame origin and orientation determined by bundle adjustment
- Camera poses expressed relative to this arbitrary world frame
- Standard aniposelib output format

#### Reoriented Calibration (`calibration_reoriented.toml`)
- **Origin**: Center of calibration board on ground
- **X-axis**: Right along board surface (ground plane)
- **Y-axis**: Forward along board surface (ground plane)
- **Z-axis**: Upward from ground (perpendicular to board)
- Cameras at positive Z heights, looking downward (-Z direction)

This reorientation is ideal for overhead tracking systems where you want world coordinates aligned with the ground plane.

## Differences from Standard Anipose

This implementation extends aniposelib with several enhancements:

### 1. Enhanced Intrinsic Calibration Stability
- **Modification**: Monkey-patches `CameraGroup.calibrate_rows()` to require ≥12 detected corners
- **Rationale**: Some cameras produce NaN intrinsics with anipose's default ≥9 threshold
- **Result**: More stable intrinsic parameter initialization
- **Location**: `bigtank/calibrate_cameras.py:37-96`

### 2. Ground-Plane Coordinate Reorientation
- **Feature**: Automatic transformation to board-centric coordinate frame
- **Process**:
  1. Detect board in final 3 seconds of calibration videos
  2. Transform board poses from camera coords to calibration world coords
  3. Fit plane to board centers to define ground plane
  4. Compute rotation to align Z-axis with plane normal (pointing upward)
  5. Translate origin to board center on ground
  6. Apply transformation to all camera extrinsics
- **Output**: Separate `calibration_reoriented.toml` file
- **Use case**: Overhead tracking systems where ground-plane alignment is critical
- **Location**: `bigtank/calibrate_cameras.py:2017-2140`

### 3. Dual Error Metrics
- **Bundle adjustment error**: Standard aniposelib optimization error
- **Per-frame validation error**: Independent RANSAC-based validation
  - Uses `cv2.solvePnPRansac()` with ITERATIVE solver
  - Requires ≥6 inlier points per frame
  - 8-pixel reprojection error threshold for RANSAC
  - Provides frame-by-frame quality assessment
- **Benefit**: Detects overfitting and validates calibration quality on held-out data
- **Location**: `bigtank/calibrate_cameras.py:1060-1227`

### 4. Comprehensive Visualizations
- **3D camera extrinsics plot**: Shows camera positions, orientations (frustums), and board location
- **Reprojection error heatmap**: Per-camera, per-frame error visualization with statistics
- **Location**: `bigtank/calibrate_cameras.py:1230-1395` (visualizations)

### 5. Detailed Summary Reports
- **calibration_summary.txt**: Focal lengths, image sizes, error metrics, quality assessment
- Human-readable format for quick quality checks

## Output Files

### Calibration Files (TOML)

**calibration.toml**: Standard multi-camera calibration
```toml
[cam_0]
name = "e3v8250"
size = [1600, 1200]
matrix = [[1940.21, 0.0, 799.5], [0.0, 1940.21, 599.5], [0.0, 0.0, 1.0]]
distortions = [-0.447, 0.0, 0.0, 0.0, 0.0]
rotation = [-2.636, 1.596, 0.033]       # Rodriguez vector
translation = [-287.56, 93.46, 1707.24]  # Arbitrary world frame
```

**calibration_reoriented.toml**: Ground-plane aligned calibration (same format, different extrinsics)

### Summary Files (TXT)

**calibration_summary.txt**:
- Board parameters
- Camera information (names, focal lengths, image sizes)
- Error metrics (bundle adjustment + per-frame validation)
- Quality assessment

### Visualizations (PNG)

**board.png**: Generated calibration board image for printing

**camera_extrinsics.png**: 3D visualization showing:
- Camera positions as colored points
- Camera viewing frustums
- Calibration board location and orientation
- Coordinate axes

**reprojection_error_heatmap.png**:
- Heatmap of reprojection errors (cameras × frames)
- Error distribution histogram
- Statistical summary (mean, std, min, max, percentiles)

## Error Metrics Explained

### Bundle Adjustment Error
- Reported by aniposelib after calibration optimization
- Measures reprojection error on corners used during bundle adjustment
- Optimistic metric (calibration was optimized to minimize this)
- Typical values: 0.5-1.5 pixels for good calibrations

### Per-Frame Validation Error
- Independent validation using RANSAC + ITERATIVE solver
- Estimates board pose from corner detections, reprojects, measures error
- Not influenced by bundle adjustment optimization
- Uses robust RANSAC to filter outlier corner detections
- More conservative metric that detects overfitting
- Typical values: 0.5-2.0 pixels for good calibrations

**Quality Assessment:**
- **Excellent**: Both errors < 0.8 pixels
- **Good**: Both errors < 1.5 pixels
- **Acceptable**: Both errors < 2.5 pixels
- **Poor**: Either error > 2.5 pixels (review calibration)

## Tips for Best Results

1. **Board movement**: Move the board slowly and smoothly through the entire field of view
2. **Multiple angles**: Tilt the board at various angles (0-45°) to calibrate distortion
3. **Coverage**: Ensure board reaches edges and corners of the camera views
4. **Lighting**: Use even, diffuse lighting to maximize corner detection
5. **Focus**: Ensure all cameras are properly focused on the board
6. **Synchronization**: Videos should be temporally synchronized across cameras

## Troubleshooting

**NaN intrinsics**: If cameras produce NaN intrinsic parameters, try:
- Improving corner detection by adjusting lighting or board position
- Adjusting the corner threshold above or below 12
- Check that board parameters in config match physical board

**High reprojection errors**: If validation errors exceed 2.5 pixels:
- Review video quality and corner detection
- Ensure board is moved through sufficient variety of angles
- Check for motion blur or focus issues
- Verify board dimensions in config match physical board

**Reorientation failure**: If ground-plane reorientation fails:
- Ensure final 3 seconds of videos show board on ground
- Board should be clearly visible to multiple cameras
- Adjust `reorient_duration_seconds` parameter if needed

**Missing visualizations**: If plots aren't generated:
- Check that matplotlib is installed
- Run without `--no-visualizations` flag
- Check console output for specific error messages

## Citation

This implementation builds on aniposelib:

```
@software{karashchuk2021anipose,
  author = {Karashchuk, Pierre and Tuthill, John C and Brunton, Steven L and Dickinson, Michael H and Tuthill, John and Murphy, James T},
  title = {Anipose: a toolkit for robust markerless 3D pose estimation},
  url = {https://github.com/lambdaloop/anipose},
  year = {2021}
}
```

## License

See LICENSE file for details.

## Supplement

This section contains details on the development of this repository. It is not necessary reading for using this repo,
but may help you understand the logic of my decisions and avoid pitfalls I've already explored if you wish to 
reuse or modify this code. 

# Scenario
I created this pipeline because anipose was struggling to calibrate the camera array that I was using in a particular
project. The array consisted of 7 cameras arranged in a coplanar array, with 6 cameras at the vertices of a regular hexagon
and the final camera in its center. While every part of the scene was visible to 2+ cameras (theoretically sufficient
for calibration and 3d reconstruction) little (if any) of the scene was visible to all 7 cameras simultaneously.
So when recording the calibration video by moving a charuco board through the capture volume, each camera experiences
periods where the board is partially/completely out of frame.

# Development Notes
Using the default anipose calibration approach (aniposelib.cameras.CameraGroup.calibrate_videos), I encountered the
following error in aniposelib's get_calibration_graph function. 

```
  Could not build calibration graph.                                                                                                                                                                                   
  Some group of cameras could not be paired by simultaneous calibration board detections.                                                                                                                              
  Check which cameras have different group numbers below to see the missing edges.                                                                                                                                     
  {'e3v8250': 5, 'e3v829d': 1, 'e3v82e0': 5, 'e3v82f9': 5, 'e3v831e': 5, 'e3v832e': 5, 'e3v8334': 5}  
```

I first attempted re-recording the calibration videos being careful to move the board over the entire scene
at multiple angles and rotations. I got even worse results, with 2 cameras isolated. I then confirmed 
that (1) the cameras were properly synchronized, (2) the isolated camera(s) has similar detection quality
to the non-isolated cameras, and (3) there were plenty of frames where the isolated camera(s) and one or more non-isolated
cameras had simultaneous detections. 

Closer examination and debugging revealed that the root problem was that, during the earlier
intrinsic calibration step, the function cv2.initCameraMatrix2D() was returning NaN values for the isolated 
cameras. Digging into the aniposelib code, I found that in the CameraGroup.calibrate_rows method, there was a 
hard-coded filter where only frames with >=9 detections were used for intrinsic calibration. I wondered first if this 
might be too strict a threshold, and if lowering it might give the solver the extra data it needed to find a solution,
but lowering the threshold to >=6 still yielded NaN values. I then wondered if it might be too lenient, as I was using
a relatively large (12x9) calibration grid, meaning a frame with 9 detections might have just one visible column. I 
therefore bumped the threshold to >=12, and got non-NaN results for the first time. I then ran the extrinsic portion
of calibration and plotted the camera positions in 3D space. While the orientation was a bit unusual in the default 
coordinate system, the overall geometry clearly resembled the real-world layout.

Next, I implemented methods for quantifying the calibration quality. The bundle adjustment error is already returned 
by CameraGroup.calibrate_rows (functionality that I preserved in my monkey-patch of that method), so that gives us
one metric. For a more easily interpretable metric, I also implemented a reprojection error calculation. In my first
attempt, I found that while most of my reprojection errors were very low, there were some extreme outliers (some
many times larger than the original image dimensions). In my original approach, I used cv2.SOLVEPNP_EPNP when only 4-6 
points were available, and cv2.SOLVEPNP_ITERATIVE for ≥6 points (as the latter requires 6 or more points). 
I found that the outliers came exclusively from the 4-6 point cases. This struck me more as a numerical instability
than an actual result. I therefore opted to focus on the ≥6 case exclusively. I also switched to v2.solvePnPRansac
with cv2.SOLVEPNP_ITERATIVE as the underlying solver. For my calibration data, this yielded a mean per-frame 
reprojection error of 0.6146 pixels, just slightly lower than the bundle adjustment error of 0.8069 pixels. Both
metrics suggest a "good" quality calibration, which is sufficient for my purposes. 

I also tested how increasing the min_corners_intrinsic threshold above 12 affected the results. I tried this after 
noticing that, even at a threshold of >=12, I was still getting 2000-3000 valid frames per camera. Since the frames
with 9 to 11 detections were so problematic, it seemed reasonable that an even higher threshold might be even better.
I tested thresholds of 54 (i.e., half of all corners in my 12*9 board) and 33 (halfway between 54 and 12). Results
in table below. While the threshold of 12 seems to perform marginally better (lowest bundle adjustment and 
mean reprojection errors) the overall difference is minimal. Note that the reoriented position accuracy
section represents results calculated after reorientation, which I cover in the next paragraph.

● ## Calibration Threshold Comparison: Summary Table

  | Metric                                          | Threshold=12 (Best) | Threshold=33 | Δ from Best | Threshold=54 | Δ from Best |
  |-------------------------------------------------|---------------------|--------------|-------------|--------------|-------------|
  | **Quality Metrics**                             |
  | Bundle Adjustment Error (px)                    | 0.8068 | 0.8092 | +0.0024 | 0.8089 | +0.0021 |
  | Mean reprojection error (px)                    | 0.6146 | 0.6153 | +0.0007 | 0.6156 | +0.0010 |
  | **Focal Length Statistics (pixels)**            |
  | Mean Focal Length                               | 1929.28 | 1927.60 | -1.68 | 1925.26 | -4.02 |
  | Std Dev Focal Length                            | 26.45 | 25.67 | -0.78 | 25.25 | -1.20 |
  | Max Focal Length (e3v8334)                      | 1970.74 | 1964.72 | -6.02 | 1966.22 | -4.52 |
  | Min Focal Length (e3v829d)                      | 1887.02 | 1887.78 | +0.76 | 1887.09 | +0.07 |
  | RMS Focal Length Error                          | - | 2.38 | - | 3.02 | - |
  | **Reoriented Position Accuracy (mm)**           |
  | Mean Abs X Error                                | - | 0.04 | - | 0.05 | - |
  | Max Abs X Error                                 | - | 0.10 | - | 0.10 | - |
  | Mean Abs Y Error                                | - | 0.46 | - | 0.58 | - |
  | Max Abs Y Error                                 | - | 0.79 | - | 0.80 | - |
  | Mean Abs Z Error                                | - | 1.25 | - | 2.20 | - |
  | Max Abs Z Error                                 | - | 5.07 | - | 5.55 | - |
  | 3D RMS Position Error                           | - | 1.71 | - | 2.55 | - |
  | **Per-Camera Focal Length Deviations (pixels)** |
  | e3v8250                                         | 1940.21 | 1939.19 | -1.02 | 1937.54 | -2.67 |
  | e3v829d                                         | 1887.02 | 1887.78 | +0.76 | 1887.09 | +0.07 |
  | e3v82e0                                         | 1907.36 | 1907.76 | +0.40 | 1906.25 | -1.11 |
  | e3v82f9                                         | 1925.75 | 1922.23 | -3.52 | 1925.64 | -0.11 |
  | e3v831e                                         | 1925.04 | 1924.76 | -0.28 | 1921.32 | -3.72 |
  | e3v832e                                         | 1948.82 | 1946.49 | -2.33 | 1942.77 | -6.05 |
  | e3v8334                                         | 1970.74 | 1964.72 | -6.02 | 1966.22 | -4.52 |

Once I was confident in the overall quality and validity of the calibration pipeline, I 
turned towards reorientation. I figured that, rather than accepting the somewhat arbitrary default coordinate
system, it would be nice if the origin was on the ground plane, in the center of my arena, with the
z axis increasing upwards. This turned out to be a real rabbit hole, due to the many different frames
of reference involved and the fact that opencv changed certain conventions about default axis orientations in
version 4.6.0. Because of this, and because linear algebra isn't my strongest suit, I did lean on 
generative AI tools more heavily than I would usually in this section of the code. See the "coordinate frames reference"
block comment towards the beginning of calibrate_cameras.py for more details on the reorientation process,
but approach with a healthy dose of skepticism. 







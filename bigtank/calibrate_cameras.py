"""
Multi-camera calibration module using aniposelib.

This module provides modular functions for performing multi-camera calibration
using ChArUco boards. Functions are designed to be imported and composed in
other scripts.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib_fallback
    tomllib = None

from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup


# ============================================================================
# Monkey-patch: Fix intrinsic calibration threshold
# ============================================================================
# Issue: Some cameras fail intrinsic initialization with NaN when using the
# default threshold of ≥9 corners. Investigation showed that using ≥12 corners
# filters out lower-quality detections and produces valid results.
# This patch modifies aniposelib's calibrate_rows method to use ≥12 threshold.

_original_calibrate_rows = CameraGroup.calibrate_rows

def _patched_calibrate_rows(self, all_rows, board,
                            init_intrinsics=True, init_extrinsics=True, verbose=True,
                            min_corners_intrinsic=12,  # New parameter with fixed default
                            **kwargs):
    """
    Patched version of CameraGroup.calibrate_rows with configurable corner threshold.

    Parameters
    ----------
    min_corners_intrinsic : int, optional
        Minimum number of corners required for intrinsic calibration (default: 12)
        Original aniposelib default was 9, but 12 produces more stable results.
    """
    from aniposelib.cameras import merge_rows, extract_points, extract_rtvecs
    from aniposelib.cameras import get_connections, get_initial_extrinsics
    from pprint import pprint

    assert len(all_rows) == len(self.cameras), \
        "Number of camera detections does not match number of cameras"

    for rows, camera in zip(all_rows, self.cameras):
        size = camera.get_size()

        assert size is not None, \
            "Camera with name {} has no specified frame size".format(camera.get_name())

        if init_intrinsics:
            objp, imgp = board.get_all_calibration_points(rows)
            # PATCHED: Use configurable threshold (default 12 instead of 9)
            mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= min_corners_intrinsic]
            objp, imgp = zip(*mixed)
            matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
            camera.set_camera_matrix(matrix.copy())
            camera.zero_distortions()

    print(self.get_dicts())

    for i, (row, cam) in enumerate(zip(all_rows, self.cameras)):
        all_rows[i] = board.estimate_pose_rows(cam, row)

    new_rows = [[r for r in rows if r['ids'].size >= 8] for rows in all_rows]
    merged = merge_rows(new_rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)

    if init_extrinsics:
        rtvecs = extract_rtvecs(merged)
        if verbose:
            pprint(get_connections(rtvecs, self.get_names()))
        rvecs, tvecs = get_initial_extrinsics(rtvecs, self.get_names())
        self.set_rotations(rvecs)
        self.set_translations(tvecs)

    error = self.bundle_adjust_iter(imgp, extra, verbose=verbose, **kwargs)

    return error

# Apply the monkey-patch
CameraGroup.calibrate_rows = _patched_calibrate_rows


# ============================================================================
# COORDINATE FRAMES REFERENCE
# ============================================================================
"""
This module works with multiple coordinate frames during calibration and reorientation.
Understanding these frames is essential for working with the coordinate transformation code.

COORDINATE FRAMES
-----------------

1. CAMERA FRAME (OpenCV Convention):
   - Origin: Camera optical center
   - X-axis: Right (along image columns, toward positive column indices)
   - Y-axis: Down (along image rows, toward positive row indices)
   - Z-axis: Forward (along optical axis, into the scene)
   - Camera looks in +Z direction

2. CHARUCO BOARD FRAME (OpenCV Convention):
   - Origin: One corner of the board
   - X-axis: Right along board surface
   - Y-axis: Down along board surface
   - Z-axis: Depends on OpenCV version:
     * OpenCV 4.6.0+ with CW ordering: Z points INTO board (perpendicular to board, away from viewer)
     * Pre-4.6.0: Z pointed OUT of board (toward viewer)
   - For boards on ground with overhead cameras (OpenCV 4.6.0+):
     Z points downward (into ground, away from cameras above)

3. CALIBRATION WORLD FRAME:
   - Arbitrary frame established during calibration
   - Usually centered near first camera position or first board detection
   - Orientation determined by bundle adjustment process

4. DESIRED BOARD-CENTRIC WORLD FRAME:
   - Origin: Center of calibration board on ground
   - X-axis: Right along board surface (ground plane)
   - Y-axis: Forward along board surface (ground plane)
   - Z-axis: Upward from ground (perpendicular to board, away from ground)
   - For overhead cameras: cameras at positive Z, looking in -Z direction (downward)

CAMERA EXTRINSICS (OpenCV Convention)
-------------------------------------

Camera extrinsics (cam.rvec, cam.tvec) represent the transformation:
    p_cam = R @ p_world + tvec
    where R = cv2.Rodrigues(cam.rvec)

This is a WORLD-TO-CAMERA transformation:
    - R rotates vectors from world frame to camera frame
    - tvec is the position of the world origin expressed in camera coordinates

IMPORTANT: This is NOT the camera position in world coordinates!

To get CAMERA-TO-WORLD (camera position and orientation in world):
    Camera position in world: -R.T @ tvec
    Camera orientation in world: R.T (columns are camera X, Y, Z axes in world)

Why the negative sign?
    At camera center, p_cam = 0:
        0 = R @ p_world_cam_center + tvec
        p_world_cam_center = -R^-1 @ tvec = -R.T @ tvec

    The negative accounts for the inverse relationship between camera position
    and the translation vector in the world-to-camera transformation.

COORDINATE FRAME TRANSFORMATIONS
--------------------------------

When changing from old world frame to new world frame, we must transform both
points and camera extrinsics.

Point Transformation:
    p_new = R_transform @ (p_old - new_origin_in_old)
    p_new = R_transform @ p_old + t_transform

    where: t_transform = -R_transform @ new_origin_in_old

Camera Extrinsics Transformation:
    R_cam_new = R_cam_old @ R_transform.T
    tvec_new = R_transform @ tvec_old + t_transform

Why transpose for R_cam?
    Camera extrinsics represent world-to-camera transforms. When we change
    the "world" frame definition, the camera's rotation must be composed with
    the inverse of the frame change rotation. Since R_transform rotates from
    old world to new world, R_transform.T rotates from new world to old world.

    The camera rotation is "left-multiplied" by the inverse frame change:
        p_cam = R_cam_old @ p_old
              = R_cam_old @ (R_transform.T @ p_new)
              = (R_cam_old @ R_transform.T) @ p_new
              = R_cam_new @ p_new

TRANSFORMATION MATRIX NOTATION
-----------------------------

Throughout this module:
    {frame1}_to_{frame2}_rotation : Rotation matrix from frame1 to frame2
    {object}_position_in_{frame}  : Position of object in given frame
    {frame}_origin_in_{otherframe}: Origin of one frame expressed in another

Example:
    world_to_cam_rotation : Rotates world coordinates to camera coordinates (R)
    cam_to_world_rotation : Rotates camera coordinates to world coordinates (R.T)
    cam_position_in_world : Camera center position in world coordinates

REORIENTATION PROCESS
--------------------

The reorientation pipeline transforms the calibration frame to a board-centric frame:

1. Detect board poses in final video frames (cameras see board on ground)
2. Transform board poses from camera coordinates to calibration world coordinates
3. Average multiple board detections to get mean board pose
4. Rotate board frame 180° around X-axis (OpenCV 4.6.0+: board Z downward → world Z upward)
5. Compute coordinate frame transformation parameters
6. Apply transformation to all camera extrinsics

After reorientation:
    - Origin at board center on ground
    - XY plane is the ground plane
    - Z-axis points upward
    - Cameras at positive Z heights, looking in -Z direction (downward)
"""


# ============================================================================
# 1. Configuration Management
# ============================================================================

def load_config(config_path: str) -> dict:
    """
    Load and parse anipose config.toml file.

    Parameters
    ----------
    config_path : str
        Path to config.toml file

    Returns
    -------
    dict
        Parsed configuration dictionary
    """
    config_path = Path(config_path)

    if tomllib is not None:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
    else:
        with open(config_path, 'r') as f:
            config = tomllib_fallback.load(f)

    return config


def extract_calibration_params(config: dict) -> dict:
    """
    Extract and normalize calibration-specific parameters.

    Parameters
    ----------
    config : dict
        Full configuration dictionary from load_config()

    Returns
    -------
    dict
        Normalized calibration parameters
    """
    calib = config.get('calibration', {})

    params = {
        'board_type': calib.get('board_type', 'charuco'),
        'board_size': calib.get('board_size', [7, 10]),
        'square_length': calib.get('board_square_side_length', 60),
        'marker_length': calib.get('board_marker_length', 45),
        'marker_bits': calib.get('board_marker_bits', 4),
        'dict_size': calib.get('board_marker_dict_number', 50),
        'fisheye': calib.get('fisheye', False),
        'manually_verify': calib.get('manually_verify', False)
    }

    return params


def get_config_value(config: dict, key_path: str, default=None):
    """
    Safely extract nested config values.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    key_path : str
        Dot-separated path (e.g., "calibration.board_type")
    default : optional
        Default value if key not found

    Returns
    -------
        Config value or default
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


# ============================================================================
# 2. Board Creation
# ============================================================================

def create_board_from_config(config: dict):
    """
    Create CharucoBoard object from config parameters.

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config()

    Returns
    -------
    CharucoBoard
        Board object for calibration
    """
    params = extract_calibration_params(config)

    board = CharucoBoard(
        squaresX=params['board_size'][0],
        squaresY=params['board_size'][1],
        square_length=params['square_length'],
        marker_length=params['marker_length'],
        marker_bits=params['marker_bits'],
        dict_size=params['dict_size'],
        manually_verify=params['manually_verify']
    )

    return board


def save_board_image(board, output_path: str, size: tuple = None) -> None:
    """
    Generate and save PNG image of the calibration board.

    Parameters
    ----------
    board : CharucoBoard
        Board object to render
    output_path : str
        Path to save PNG file (e.g., 'project_folder/output/board.png')
    size : tuple, optional
        Image size in pixels (width, height). If None, auto-calculated based on board dimensions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate size if not provided
    if size is None:
        # Use board dimensions to calculate appropriate image size
        # Add margin and scale up for print quality
        margin = 2  # squares of margin
        total_squares_x = board.squaresX + margin * 2
        total_squares_y = board.squaresY + margin * 2
        pixels_per_square = 100  # pixels per square for good resolution
        size = (total_squares_x * pixels_per_square, total_squares_y * pixels_per_square)

    # Generate board image using OpenCV's generateImage method
    # Note: board.board accesses the underlying cv2.aruco.CharucoBoard object
    board_img = board.board.generateImage(size)

    # Save as PNG
    cv2.imwrite(str(output_path), board_img)


# ============================================================================
# 3. Video Discovery and Organization
# ============================================================================

def discover_calibration_videos(project_folder: str, video_extension: str = 'avi') -> List[str]:
    """
    Find all calibration videos in project folder's videos/ subfolder.

    Parameters
    ----------
    project_folder : str
        Path to project folder (e.g., 'projects/calibration_121125')
    video_extension : str, optional
        Video file extension (default: 'avi')

    Returns
    -------
    List[str]
        Sorted list of absolute video file paths
    """
    project_path = Path(project_folder)
    videos_dir = project_path / 'videos'

    # Find all videos with given extension
    video_paths = sorted(videos_dir.glob(f'*.{video_extension}'))

    return [str(p.absolute()) for p in video_paths]


def extract_camera_names(video_paths: List[str], cam_regex: str) -> List[str]:
    """
    Extract camera identifiers from filenames using regex pattern.

    Parameters
    ----------
    video_paths : List[str]
        List of video file paths
    cam_regex : str
        Regex pattern from config['triangulation']['cam_regex']

    Returns
    -------
    List[str]
        List of camera names in same order as video_paths
    """
    pattern = re.compile(cam_regex)
    cam_names = []

    for video_path in video_paths:
        filename = Path(video_path).name
        match = pattern.search(filename)
        if match:
            cam_names.append(match.group(1))
        else:
            # Fallback to filename without extension
            cam_names.append(Path(video_path).stem)

    return cam_names


def organize_videos_by_camera(video_paths: List[str]) -> List[List[str]]:
    """
    Organize video paths into per-camera lists for aniposelib.

    Parameters
    ----------
    video_paths : List[str]
        List of video file paths

    Returns
    -------
    List[List[str]]
        Nested list format: [[cam1_vid], [cam2_vid], [cam3_vid]]
    """
    return [[vid] for vid in video_paths]


# ============================================================================
# 4. Camera Group Management
# ============================================================================

def create_camera_group(cam_names: List[str], fisheye: bool = False) -> CameraGroup:
    """
    Create empty CameraGroup object ready for calibration.

    Parameters
    ----------
    cam_names : List[str]
        List of camera names
    fisheye : bool, optional
        Whether cameras use fisheye lenses (default: False)

    Returns
    -------
    CameraGroup
        Empty camera group from aniposelib
    """
    cgroup = CameraGroup.from_names(cam_names, fisheye=fisheye)
    return cgroup


# ============================================================================
# 5. Calibration Execution
# ============================================================================

def calibrate_cameras(
    video_lists: List[List[str]],
    board,
    cam_names: List[str],
    fisheye: bool = False,
    init_intrinsics: bool = True,
    init_extrinsics: bool = True,
    verbose: bool = True
) -> Tuple[CameraGroup, float]:
    """
    Perform multi-camera calibration using aniposelib.

    Parameters
    ----------
    video_lists : List[List[str]]
        List of video lists (one per camera)
    board : CharucoBoard
        Board object for detection
    cam_names : List[str]
        List of camera names
    fisheye : bool, optional
        Fisheye lens flag (default: False)
    init_intrinsics : bool, optional
        Initialize intrinsics during calibration (default: True)
    init_extrinsics : bool, optional
        Initialize extrinsics during calibration (default: True)
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    Tuple[CameraGroup, float]
        - Calibrated camera group
        - Bundle adjustment reprojection error (pixels)
    """
    # Create camera group
    cgroup = create_camera_group(cam_names, fisheye=fisheye)

    # Perform calibration and capture bundle adjustment error
    bundle_error = cgroup.calibrate_videos(
        video_lists,
        board,
        init_intrinsics=init_intrinsics,
        init_extrinsics=init_extrinsics,
        verbose=verbose
    )

    # calibrate_videos returns the final bundle adjustment error
    # Handle both single value and tuple return (depending on aniposelib version)
    if bundle_error is None:
        bundle_error = 0.0
    elif isinstance(bundle_error, tuple):
        bundle_error = bundle_error[0]  # Extract first element if tuple

    return cgroup, bundle_error


# ============================================================================
# 6. Calibration Persistence
# ============================================================================

def save_calibration(camera_group: CameraGroup, project_folder: str, metadata: Optional[dict] = None) -> None:
    """
    Save calibrated CameraGroup to TOML file.

    Parameters
    ----------
    camera_group : CameraGroup
        Calibrated camera group
    project_folder : str
        Path to project folder (saves to {project_folder}/calibration.toml)
    metadata : dict, optional
        Optional metadata to include in calibration file
    """
    project_path = Path(project_folder)
    output_path = project_path / 'calibration.toml'

    if metadata is not None:
        camera_group.metadata = metadata

    camera_group.dump(str(output_path))


def load_calibration(project_folder: str) -> CameraGroup:
    """
    Load calibration from TOML file.

    Parameters
    ----------
    project_folder : str
        Path to project folder (loads from {project_folder}/calibration.toml)

    Returns
    -------
    CameraGroup
        Camera group with loaded calibration
    """
    project_path = Path(project_folder)
    calib_path = project_path / 'calibration.toml'

    cgroup = CameraGroup.load(str(calib_path))
    return cgroup


# ============================================================================
# 7. Validation and Metrics
# ============================================================================

def validate_calibration(camera_group: CameraGroup) -> dict:
    """
    Compute calibration quality metrics.

    Parameters
    ----------
    camera_group : CameraGroup
        Calibrated camera group to validate

    Returns
    -------
    dict
        Dictionary with validation metrics
    """
    metrics = {
        'n_cameras': len(camera_group.cameras),
        'camera_names': [cam.get_name() for cam in camera_group.cameras],
        'all_intrinsics_initialized': all(cam.matrix is not None for cam in camera_group.cameras),
        'all_extrinsics_initialized': all(cam.rvec is not None for cam in camera_group.cameras),
    }

    # Extract focal lengths and image sizes
    focal_lengths = {}
    image_sizes = {}

    for cam in camera_group.cameras:
        name = cam.get_name()
        if cam.matrix is not None:
            fx = cam.matrix[0, 0]
            fy = cam.matrix[1, 1]
            focal_lengths[name] = (fx, fy)
        if cam.size is not None:
            image_sizes[name] = tuple(cam.size)

    metrics['focal_lengths'] = focal_lengths
    metrics['image_sizes'] = image_sizes

    return metrics


def compute_reprojection_error(camera_group: CameraGroup, points_2d, points_3d) -> float:
    """
    Calculate reprojection error to assess calibration quality.

    Parameters
    ----------
    camera_group : CameraGroup
        Calibrated camera group
    points_2d : array-like
        2D detected points (CxNx2 array)
    points_3d : array-like
        Triangulated 3D points (Nx3 array)

    Returns
    -------
    float
        Mean reprojection error in pixels
    """
    error = camera_group.reprojection_error(points_3d, points_2d, mean=True)
    return error


def save_calibration_summary(
    camera_group: CameraGroup,
    config: dict,
    output_path: str,
    bundle_adjustment_error: Optional[float] = None,
    per_frame_validation_error: Optional[float] = None
) -> None:
    """
    Save formatted calibration summary to text file.

    Parameters
    ----------
    camera_group : CameraGroup
        Calibrated camera group
    config : dict
        Configuration dictionary used for calibration
    output_path : str
        Path to save summary file (e.g., 'project_folder/output/calibration_summary.txt')
    bundle_adjustment_error : float, optional
        Bundle adjustment reprojection error from calibration (pixels)
    per_frame_validation_error : float, optional
        Per-frame validation reprojection error using RANSAC (pixels)
    """
    from datetime import datetime

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get calibration parameters
    calib_params = extract_calibration_params(config)

    # Get validation metrics
    metrics = validate_calibration(camera_group)

    # Build summary text
    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-CAMERA CALIBRATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Calibration parameters
    lines.append("-" * 70)
    lines.append("CALIBRATION PARAMETERS")
    lines.append("-" * 70)
    lines.append(f"Board type:           {calib_params['board_type']}")
    lines.append(f"Board size:           {calib_params['board_size'][0]} x {calib_params['board_size'][1]}")
    lines.append(f"Square length:        {calib_params['square_length']} mm")
    lines.append(f"Marker length:        {calib_params['marker_length']} mm")
    lines.append(f"Marker bits:          {calib_params['marker_bits']}")
    lines.append(f"Dictionary size:      {calib_params['dict_size']}")
    lines.append(f"Fisheye model:        {calib_params['fisheye']}")
    lines.append(f"Manual verification:  {calib_params['manually_verify']}")
    lines.append("")

    # Camera information
    lines.append("-" * 70)
    lines.append("CAMERA INFORMATION")
    lines.append("-" * 70)
    lines.append(f"Number of cameras:    {metrics['n_cameras']}")
    lines.append(f"Camera names:         {', '.join(metrics['camera_names'])}")
    lines.append("")

    # Calibration status
    lines.append("-" * 70)
    lines.append("CALIBRATION STATUS")
    lines.append("-" * 70)
    lines.append(f"Intrinsics calibrated:  {metrics['all_intrinsics_initialized']}")
    lines.append(f"Extrinsics calibrated:  {metrics['all_extrinsics_initialized']}")
    lines.append("")

    # Focal lengths
    if metrics['focal_lengths']:
        lines.append("-" * 70)
        lines.append("FOCAL LENGTHS (pixels)")
        lines.append("-" * 70)
        lines.append(f"{'Camera':<20} {'fx':>12} {'fy':>12}")
        lines.append("-" * 70)
        for cam_name, (fx, fy) in sorted(metrics['focal_lengths'].items()):
            lines.append(f"{cam_name:<20} {fx:>12.2f} {fy:>12.2f}")
        lines.append("")

    # Image sizes
    if metrics['image_sizes']:
        lines.append("-" * 70)
        lines.append("IMAGE SIZES (pixels)")
        lines.append("-" * 70)
        lines.append(f"{'Camera':<20} {'Width':>12} {'Height':>12}")
        lines.append("-" * 70)
        for cam_name, (width, height) in sorted(metrics['image_sizes'].items()):
            lines.append(f"{cam_name:<20} {width:>12} {height:>12}")
        lines.append("")

    # Reprojection errors
    if bundle_adjustment_error is not None or per_frame_validation_error is not None:
        lines.append("-" * 70)
        lines.append("CALIBRATION QUALITY")
        lines.append("-" * 70)

        if bundle_adjustment_error is not None:
            lines.append(f"Bundle adjustment error:      {bundle_adjustment_error:.4f} pixels")
            lines.append("  (Optimization error from multi-camera calibration)")

        if per_frame_validation_error is not None:
            lines.append(f"Per-frame validation error:   {per_frame_validation_error:.4f} pixels")
            lines.append("  (RANSAC validation using ITERATIVE solver, >=6 points)")

        lines.append("")

        # Quality assessment based on bundle adjustment error (primary metric)
        if bundle_adjustment_error is not None:
            if bundle_adjustment_error < 0.5:
                quality = "Excellent"
            elif bundle_adjustment_error < 1.0:
                quality = "Good"
            elif bundle_adjustment_error < 2.0:
                quality = "Acceptable"
            else:
                quality = "Poor - consider recalibration"
            lines.append(f"Quality assessment:           {quality}")
            lines.append("")

    lines.append("=" * 70)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# ============================================================================
# 8. Visualization Functions
# ============================================================================

class Arrow3D(FancyArrowPatch):
    """
    Helper class for drawing 3D arrows in matplotlib.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (in radians).

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix

    Returns
    -------
    Tuple[float, float, float]
        Euler angles (rx, ry, rz) in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return x, y, z


def _draw_camera_frustum(ax, position: np.ndarray, rotation: np.ndarray,
                         scale: float = 100, color: str = 'blue', alpha: float = 0.3):
    """
    Draw a camera frustum (pyramid) showing the camera's field of view.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes
    position : np.ndarray
        Camera position (3,) array
    rotation : np.ndarray
        Camera rotation matrix (3x3)
    scale : float
        Size of the frustum
    color : str
        Color of the frustum
    alpha : float
        Transparency of the frustum
    """
    # Define frustum vertices in camera coordinate system
    # Camera looks down the -Z axis in OpenCV convention
    frustum_points = np.array([
        [0, 0, 0],           # Camera center
        [-1, -1, 2],         # Bottom-left
        [1, -1, 2],          # Bottom-right
        [1, 1, 2],           # Top-right
        [-1, 1, 2]           # Top-left
    ]) * scale / 2

    # Transform to world coordinates
    world_points = position + (rotation @ frustum_points.T).T

    # Draw frustum edges
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # From camera center to corners
        (1, 2), (2, 3), (3, 4), (4, 1)   # Rectangle at the image plane
    ]

    for start, end in edges:
        ax.plot3D(
            [world_points[start, 0], world_points[end, 0]],
            [world_points[start, 1], world_points[end, 1]],
            [world_points[start, 2], world_points[end, 2]],
            color=color, alpha=alpha, linewidth=1.5
        )

    # Fill the frustum faces with transparency
    # Image plane rectangle
    plane_points = world_points[[1, 2, 3, 4], :]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection([plane_points], alpha=alpha * 0.3, facecolor=color, edgecolor='none')
    ax.add_collection3d(poly)


def plot_camera_extrinsics(
    camera_group: CameraGroup,
    output_path: Optional[str] = None,
    frustum_scale: float = 200,
    show_axes: bool = True,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Visualize camera positions and orientations in 3D space.

    Creates a 3D plot showing:
    - Camera positions as points
    - Camera frustums (pyramids) showing field of view
    - Camera orientation vectors
    - World coordinate system axes
    - Camera labels

    Parameters
    ----------
    camera_group : CameraGroup
        Calibrated camera group
    output_path : str, optional
        Path to save the figure (e.g., 'project_folder/output/camera_extrinsics.png')
    frustum_scale : float, optional
        Size of camera frustums in world units (default: 200)
    show_axes : bool, optional
        Whether to show world coordinate axes (default: True)
    show_labels : bool, optional
        Whether to show camera name labels (default: True)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 10))

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Collect camera positions for setting axis limits
    positions = []

    # Colors for cameras
    colors = plt.cm.tab10(np.linspace(0, 1, len(camera_group.cameras)))

    for idx, cam in enumerate(camera_group.cameras):
        cam_name = cam.get_name()
        color = colors[idx]

        # Get camera extrinsics
        if cam.rvec is None or cam.tvec is None:
            print(f"Warning: Camera {cam_name} has no extrinsics, skipping")
            continue

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(cam.rvec)

        # Camera position in world coordinates
        # The tvec in OpenCV is the position of world origin in camera coordinates
        # So camera position in world coordinates is -R.T @ tvec
        cam_position = -R.T @ cam.tvec.flatten()
        positions.append(cam_position)

        # Plot camera position
        ax.scatter(cam_position[0], cam_position[1], cam_position[2],
                  c=[color], s=100, marker='o', edgecolors='black', linewidths=2,
                  label=cam_name, zorder=10)

        # Draw camera frustum
        _draw_camera_frustum(ax, cam_position, R.T, scale=frustum_scale,
                           color=color, alpha=0.3)

        # Draw camera orientation axes
        axis_length = frustum_scale * 0.5

        # X-axis (red)
        x_axis = R.T[:, 0] * axis_length
        arrow_x = Arrow3D(
            [cam_position[0], cam_position[0] + x_axis[0]],
            [cam_position[1], cam_position[1] + x_axis[1]],
            [cam_position[2], cam_position[2] + x_axis[2]],
            mutation_scale=20, lw=2, arrowstyle='->', color='red', alpha=0.6
        )
        ax.add_artist(arrow_x)

        # Y-axis (green)
        y_axis = R.T[:, 1] * axis_length
        arrow_y = Arrow3D(
            [cam_position[0], cam_position[0] + y_axis[0]],
            [cam_position[1], cam_position[1] + y_axis[1]],
            [cam_position[2], cam_position[2] + y_axis[2]],
            mutation_scale=20, lw=2, arrowstyle='->', color='green', alpha=0.6
        )
        ax.add_artist(arrow_y)

        # Z-axis (blue) - pointing direction
        z_axis = R.T[:, 2] * axis_length
        arrow_z = Arrow3D(
            [cam_position[0], cam_position[0] + z_axis[0]],
            [cam_position[1], cam_position[1] + z_axis[1]],
            [cam_position[2], cam_position[2] + z_axis[2]],
            mutation_scale=20, lw=2, arrowstyle='->', color='blue', alpha=0.6
        )
        ax.add_artist(arrow_z)

        # Add camera label
        if show_labels:
            ax.text(cam_position[0], cam_position[1], cam_position[2] + frustum_scale * 0.3,
                   cam_name, fontsize=10, fontweight='bold', color=color)

    # Show world coordinate system origin
    if show_axes:
        origin = np.array([0, 0, 0])
        axis_length = frustum_scale * 0.8

        # World X-axis (red)
        arrow_wx = Arrow3D([0, axis_length], [0, 0], [0, 0],
                          mutation_scale=20, lw=3, arrowstyle='->', color='red')
        ax.add_artist(arrow_wx)
        ax.text(axis_length * 1.1, 0, 0, 'X', fontsize=12, fontweight='bold', color='red')

        # World Y-axis (green)
        arrow_wy = Arrow3D([0, 0], [0, axis_length], [0, 0],
                          mutation_scale=20, lw=3, arrowstyle='->', color='green')
        ax.add_artist(arrow_wy)
        ax.text(0, axis_length * 1.1, 0, 'Y', fontsize=12, fontweight='bold', color='green')

        # World Z-axis (blue)
        arrow_wz = Arrow3D([0, 0], [0, 0], [0, axis_length],
                          mutation_scale=20, lw=3, arrowstyle='->', color='blue')
        ax.add_artist(arrow_wz)
        ax.text(0, 0, axis_length * 1.1, 'Z', fontsize=12, fontweight='bold', color='blue')

        # Mark world origin
        ax.scatter([0], [0], [0], c='black', s=100, marker='x', linewidths=3, zorder=10)

    # Set axis limits based on camera positions
    if positions:
        positions = np.array(positions)
        center = positions.mean(axis=0)
        max_range = np.abs(positions - center).max() * 1.5

        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # Labels and title
    ax.set_xlabel('X (world)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (world)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (world)', fontsize=12, fontweight='bold')
    ax.set_title('Camera Extrinsics Visualization', fontsize=14, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved camera extrinsics plot to: {output_path}")

    return fig


def compute_reprojection_errors_per_frame(
    camera_group: CameraGroup,
    video_lists: List[List[str]],
    board,
    frame_step: int = 1
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Compute per-camera, per-frame reprojection errors from calibration videos.

    Parameters
    ----------
    camera_group : CameraGroup
        Calibrated camera group
    video_lists : List[List[str]]
        List of video lists (one per camera)
    board : CharucoBoard
        Board object for detection
    frame_step : int, optional
        Process every nth frame (default: 1 for all frames, 10 for every 10th frame)

    Returns
    -------
    Tuple[np.ndarray, List[str], List[int]]
        - errors: 2D array of shape (n_cameras, n_frames) with reprojection errors
        - camera_names: List of camera names
        - frame_indices: List of frame indices that were processed
    """
    n_cameras = len(camera_group.cameras)
    camera_names = [cam.get_name() for cam in camera_group.cameras]

    # Detect boards in all videos
    all_detections = []  # List of (frame_idx, camera_idx, corners_2d, ids)

    print(f"  Detecting boards in {n_cameras} videos...")
    for cam_idx, video_list in enumerate(video_lists):
        video_path = video_list[0]
        print(f"    Camera {cam_idx+1}/{n_cameras}: {Path(video_path).name}")
        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every frame_step-th frame
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            # Detect board
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = board.detect_image(gray)

            if detections is not None:
                # board.detect_image returns array with shape (n_corners, 4)
                # where columns are: [x, y, corner_id, corner_id] (last two are the same)
                if isinstance(detections, tuple):
                    # Handle tuple output (corners, ids)
                    corners_2d, ids = detections
                    if corners_2d is not None and len(corners_2d) > 0:
                        corners_2d = corners_2d.reshape(-1, 2)
                        ids = ids.flatten().astype(int)
                        all_detections.append((frame_idx, cam_idx, corners_2d, ids))
                elif len(detections) > 0:
                    # Handle array output
                    corners_2d = detections[:, :2]
                    ids = detections[:, 2].astype(int)
                    all_detections.append((frame_idx, cam_idx, corners_2d, ids))

            frame_idx += 1

        cap.release()

    # Group detections by frame
    frame_detections = {}
    for frame_idx, cam_idx, corners, ids in all_detections:
        if frame_idx not in frame_detections:
            frame_detections[frame_idx] = {}
        frame_detections[frame_idx][cam_idx] = (corners, ids)

    # Compute reprojection errors for frames with detections in multiple cameras
    frame_indices = []
    error_matrix = []

    for frame_idx in sorted(frame_detections.keys()):
        detections = frame_detections[frame_idx]

        # Need at least 2 cameras to triangulate
        if len(detections) < 2:
            continue

        # Build 2D points array for this frame
        # Find common corner IDs across cameras
        common_ids = set(detections[list(detections.keys())[0]][1])
        for cam_idx in detections.keys():
            _, ids = detections[cam_idx]
            common_ids = common_ids.intersection(set(ids))

        if len(common_ids) < 6:  # Need at least 6 points for robust RANSAC + ITERATIVE estimation
            continue

        common_ids = sorted(list(common_ids))

        # Build points_2d array: shape (n_cameras, n_points, 2)
        points_2d_frame = np.full((n_cameras, len(common_ids), 2), np.nan)

        for cam_idx, (corners, ids) in detections.items():
            for i, corner_id in enumerate(common_ids):
                if corner_id in ids:
                    idx = np.where(ids == corner_id)[0][0]
                    points_2d_frame[cam_idx, i, :] = corners[idx]

        # Get 3D board points for common IDs
        board_points_3d = np.array([board.get_object_points()[cid] for cid in common_ids])

        # Compute reprojection errors per camera
        frame_errors = np.full(n_cameras, np.nan)

        for cam_idx in range(n_cameras):
            if cam_idx not in detections:
                continue

            cam = camera_group.cameras[cam_idx]
            if cam.rvec is None or cam.tvec is None or cam.matrix is None:
                continue

            # Get detected points for this camera
            points_detected = points_2d_frame[cam_idx]

            # Remove NaN points
            valid_mask = ~np.isnan(points_detected[:, 0])
            if valid_mask.sum() == 0:
                continue

            points_detected_valid = points_detected[valid_mask]
            board_points_valid = board_points_3d[valid_mask]

            # Need at least 6 points for RANSAC + ITERATIVE
            if len(points_detected_valid) < 6:
                continue

            # Estimate board pose using RANSAC + ITERATIVE for robustness
            # RANSAC filters outlier corner detections, ITERATIVE refines the solution
            try:
                success, rvec_board, tvec_board, inliers = cv2.solvePnPRansac(
                    board_points_valid,
                    points_detected_valid,
                    cam.matrix,
                    cam.dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    reprojectionError=8.0,  # RANSAC inlier threshold (pixels)
                    confidence=0.99
                )

                if not success or inliers is None or len(inliers) < 6:
                    continue

                # Use only inlier points for error calculation
                inlier_indices = inliers.flatten()
                points_detected_valid = points_detected_valid[inlier_indices]
                board_points_valid = board_points_valid[inlier_indices]

            except cv2.error:
                # If solvePnPRansac fails, skip this frame
                continue

            # Project 3D board points using the estimated board pose
            points_projected, _ = cv2.projectPoints(
                board_points_valid,
                rvec_board,  # Board's pose in this frame
                tvec_board,
                cam.matrix,
                cam.dist
            )
            points_projected = points_projected.reshape(-1, 2)

            # Compute error
            errors = np.linalg.norm(points_detected_valid - points_projected, axis=1)
            frame_errors[cam_idx] = errors.mean()

        error_matrix.append(frame_errors)
        frame_indices.append(frame_idx)

    error_matrix = np.array(error_matrix).T  # Shape: (n_cameras, n_frames)

    return error_matrix, camera_names, frame_indices


def plot_reprojection_error_heatmap(
    camera_group: CameraGroup,
    video_lists: List[List[str]],
    board,
    output_path: Optional[str] = None,
    frame_step: int = 1,
    precomputed_errors: Optional[Tuple[np.ndarray, List[str], List[int]]] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Create a heatmap visualization of reprojection errors across cameras and frames.

    Generates a comprehensive visualization including:
    - Heatmap of errors per camera per frame
    - Histogram of error distribution
    - Statistical summary

    Parameters
    ----------
    camera_group : CameraGroup
        Calibrated camera group
    video_lists : List[List[str]]
        List of video lists (one per camera)
    board : CharucoBoard
        Board object for detection
    output_path : str, optional
        Path to save the figure (e.g., 'project_folder/output/reprojection_errors.png')
    frame_step : int, optional
        Process every nth frame (default: 1 for all frames, 10 for every 10th frame)
    precomputed_errors : tuple, optional
        Precomputed (error_matrix, camera_names, frame_indices) to avoid redundant computation
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (14, 8))

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Use precomputed errors if available, otherwise compute
    if precomputed_errors is not None:
        error_matrix, camera_names, frame_indices = precomputed_errors
    else:
        print("Computing reprojection errors per frame...")
        error_matrix, camera_names, frame_indices = compute_reprojection_errors_per_frame(
            camera_group, video_lists, board, frame_step
        )

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[4, 1],
                         hspace=0.3, wspace=0.3)

    # Main heatmap
    ax_heatmap = fig.add_subplot(gs[0, 0])

    # Replace NaN with a large value for visualization
    error_matrix_vis = error_matrix.copy()
    nan_mask = np.isnan(error_matrix_vis)
    error_matrix_vis[nan_mask] = -1  # Use -1 to indicate no data

    # Create heatmap
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle

    # Create custom colormap with gray for missing data
    cmap = plt.cm.YlOrRd
    cmap.set_under(color='lightgray')  # Color for missing data

    # Plot heatmap
    vmax = np.nanpercentile(error_matrix, 95)  # Use 95th percentile to avoid outliers
    im = ax_heatmap.imshow(error_matrix_vis, aspect='auto', cmap=cmap,
                          vmin=0, vmax=vmax, interpolation='nearest')

    # Set ticks and labels
    ax_heatmap.set_yticks(range(len(camera_names)))
    ax_heatmap.set_yticklabels(camera_names)

    # Reduce number of x-axis labels for readability
    if len(frame_indices) > 20:
        step = len(frame_indices) // 10
        tick_positions = range(0, len(frame_indices), step)
        tick_labels = [frame_indices[i] for i in tick_positions]
        ax_heatmap.set_xticks(tick_positions)
        ax_heatmap.set_xticklabels(tick_labels)
    else:
        ax_heatmap.set_xticks(range(len(frame_indices)))
        ax_heatmap.set_xticklabels(frame_indices)

    ax_heatmap.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax_heatmap.set_ylabel('Camera', fontsize=11, fontweight='bold')
    ax_heatmap.set_title('Reprojection Error Heatmap (pixels)', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label('Reprojection Error (pixels)', fontsize=10)

    # Error distribution histogram (right subplot)
    ax_hist = fig.add_subplot(gs[0, 1])

    # Flatten errors and remove NaN
    errors_flat = error_matrix[~np.isnan(error_matrix)]

    if len(errors_flat) > 0:
        ax_hist.hist(errors_flat, bins=30, orientation='horizontal',
                    color='coral', edgecolor='black', alpha=0.7)
        ax_hist.set_ylabel('Reprojection Error (pixels)', fontsize=10)
        ax_hist.set_xlabel('Count', fontsize=10)
        ax_hist.set_title('Error Distribution', fontsize=11, fontweight='bold')
        ax_hist.grid(True, alpha=0.3)

    # Per-camera statistics (bottom subplot)
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis('off')

    # Compute statistics
    stats_lines = []
    stats_lines.append("REPROJECTION ERROR STATISTICS")
    stats_lines.append("=" * 80)
    stats_lines.append(f"{'Camera':<20} {'Mean (px)':>12} {'Std (px)':>12} {'Min (px)':>12} {'Max (px)':>12} {'N Frames':>10}")
    stats_lines.append("-" * 80)

    for i, cam_name in enumerate(camera_names):
        cam_errors = error_matrix[i]
        valid_errors = cam_errors[~np.isnan(cam_errors)]

        if len(valid_errors) > 0:
            mean_err = valid_errors.mean()
            std_err = valid_errors.std()
            min_err = valid_errors.min()
            max_err = valid_errors.max()
            n_frames = len(valid_errors)
        else:
            mean_err = std_err = min_err = max_err = np.nan
            n_frames = 0

        stats_lines.append(
            f"{cam_name:<20} {mean_err:>12.3f} {std_err:>12.3f} "
            f"{min_err:>12.3f} {max_err:>12.3f} {n_frames:>10d}"
        )

    stats_lines.append("-" * 80)

    # Overall statistics
    if len(errors_flat) > 0:
        stats_lines.append(
            f"{'OVERALL':<20} {errors_flat.mean():>12.3f} {errors_flat.std():>12.3f} "
            f"{errors_flat.min():>12.3f} {errors_flat.max():>12.3f} {len(errors_flat):>10d}"
        )

        # Quality assessment
        mean_error = errors_flat.mean()
        if mean_error < 0.5:
            quality = "Excellent"
        elif mean_error < 1.0:
            quality = "Good"
        elif mean_error < 2.0:
            quality = "Acceptable"
        else:
            quality = "Poor - consider recalibration"

        stats_lines.append("")
        stats_lines.append(f"Quality Assessment: {quality}")

    stats_text = '\n'.join(stats_lines)
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Calibration Reprojection Error Analysis', fontsize=14, fontweight='bold', y=0.98)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved reprojection error heatmap to: {output_path}")

    return fig


# ============================================================================
# 9. Coordinate Frame Reorientation
# ============================================================================

# Helper Functions for Coordinate Transformations
# ------------------------------------------------

def extract_camera_world_pose(
    cam_rvec: np.ndarray,
    cam_tvec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera position and orientation in world coordinates from OpenCV extrinsics.

    OpenCV Convention:
        cam_rvec, cam_tvec define transformation: p_cam = R @ p_world + tvec
        where R = cv2.Rodrigues(cam_rvec)

    This is a WORLD-TO-CAMERA transformation:
        - R rotates from world frame to camera frame
        - tvec is the world origin position expressed in camera coordinates

    Parameters
    ----------
    cam_rvec : np.ndarray
        Camera rotation vector (world to camera) (3, 1) or (3,)
    cam_tvec : np.ndarray
        Camera translation vector (world origin in camera coords) (3, 1) or (3,)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        cam_position_world: Camera center position in world coordinates (3,)
        cam_orientation_world: Camera orientation matrix in world (3,3)
                              Columns are camera X, Y, Z axes in world frame

    Mathematical Derivation:
        Given: p_cam = R @ p_world + tvec
        At camera origin: p_cam = 0
        => 0 = R @ p_world_cam + tvec
        => p_world_cam = -R^-1 @ tvec = -R.T @ tvec

        Camera axes in world: R.T (inverse/transpose of world-to-camera rotation)
    """
    # Convert rotation vector to matrix
    world_to_cam_rotation, _ = cv2.Rodrigues(cam_rvec)

    # Camera position: -R.T @ tvec
    cam_position_world = -world_to_cam_rotation.T @ cam_tvec.flatten()

    # Camera orientation: R.T (columns are camera X, Y, Z axes in world)
    cam_orientation_world = world_to_cam_rotation.T

    return cam_position_world, cam_orientation_world


def average_rotation_matrices_svd(
    rotation_matrices: List[np.ndarray]
) -> np.ndarray:
    """
    Compute the mean rotation matrix from multiple rotations using SVD.

    This method:
    1. Computes element-wise mean of rotation matrices
    2. Projects the mean back onto SO(3) (rotation matrix manifold) using SVD
    3. Ensures right-handed coordinate system (det = +1)

    Why SVD?
        The space of rotation matrices (SO(3)) is a non-linear manifold.
        Simple element-wise averaging can produce non-orthogonal matrices.
        SVD finds the nearest valid rotation matrix to the arithmetic mean.

    Parameters
    ----------
    rotation_matrices : List[np.ndarray]
        List of 3x3 rotation matrices to average

    Returns
    -------
    np.ndarray
        mean_rotation: Averaged rotation matrix (3,3) with det = +1

    Algorithm:
        R_mean_approx = mean(rotations)  # Element-wise mean (may not be orthogonal)
        U, S, Vt = svd(R_mean_approx)
        R_mean = U @ Vt  # Projects onto SO(3) - closest rotation matrix
        if det(R_mean) < 0: flip sign to ensure det = +1 (right-handed)
    """
    if not rotation_matrices:
        raise ValueError("Cannot average empty list of rotation matrices")

    # Stack rotation matrices and compute element-wise mean
    R_stack = np.array(rotation_matrices)
    R_mean_approx = np.mean(R_stack, axis=0)

    # Project to nearest rotation matrix using SVD
    U, _, Vt = np.linalg.svd(R_mean_approx)
    R_mean = U @ Vt

    # Ensure right-handed coordinate system (det = +1)
    if np.linalg.det(R_mean) < 0:
        Vt[-1, :] *= -1
        R_mean = U @ Vt

    return R_mean


def apply_board_z_axis_flip(
    board_rotation_matrix: np.ndarray
) -> np.ndarray:
    """
    Rotate board frame 180° around X-axis to create upward-pointing world frame.

    Problem (OpenCV 4.6.0+ with CW ordering):
        ChArUco board Z-axis points INTO board surface (away from viewer).
        For overhead cameras looking down at a board on the ground:
        - Board Z points downward (into ground)
        - We want world Z to point upward (away from ground)
        - These are OPPOSITE directions

    Solution:
        Rotate 180° around X-axis: this flips both Y and Z axes.
        This is a PROPER ROTATION (det = +1), not a reflection (det = -1).
        Using a proper rotation is critical because cv2.Rodrigues() corrupts
        reflection matrices during conversion to/from axis-angle representation.

    Before rotation:
        - Board Z: points downward (into ground)
        - Board Y: some direction in the plane or perpendicular
        - Cameras: would be at negative Z (below ground) looking upward

    After 180° rotation around X:
        - World Z: points upward (away from ground)
        - World Y: flipped from board Y
        - World X: same as board X
        - Cameras: at positive Z heights, looking in -Z direction (downward)
        - Board: lies in XY plane at Z=0

    Parameters
    ----------
    board_rotation_matrix : np.ndarray
        Board orientation with Z pointing downward (OpenCV 4.6.0+, CW) (3,3)

    Returns
    -------
    np.ndarray
        world_rotation_matrix: Board orientation with Z pointing upward (3,3)
        This matrix has determinant +1 (proper rotation).

    Mathematical Note:
        180° rotation around X-axis:
        R_x(180°) = [[1,  0,  0],
                     [0, -1,  0],
                     [0,  0, -1]]
        det(R_x) = 1 * (-1) * (-1) = +1 ✓

        Compare to reflection (INCORRECT):
        diag([1, 1, -1]) has det = -1, which breaks cv2.Rodrigues()
    """
    # 180° rotation around X-axis (proper rotation, det = +1)
    rotation_180_around_x = np.array([[1,  0,  0],
                                       [0, -1,  0],
                                       [0,  0, -1]])

    # Apply rotation by post-multiplying
    # This rotates the board frame axes to create the desired world frame
    world_rotation_matrix = board_rotation_matrix @ rotation_180_around_x

    return world_rotation_matrix


def compute_frame_change_transform(
    new_origin_position: np.ndarray,
    new_axes_orientation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transformation parameters for changing coordinate frames.

    Coordinate Frame Change:
        Old frame: arbitrary calibration frame
        New frame: board-centric frame (origin at board center, axes aligned with board)

    Point Transformation:
        Points transform as:
        p_new = R_transform @ (p_old - new_origin)
        p_new = R_transform @ p_old + t_transform

        where: t_transform = -R_transform @ new_origin

    Camera Extrinsics Transformation:
        Camera extrinsics represent world-to-camera transforms.
        When the "world" frame changes, extrinsics must be updated:

        R_cam_new = R_cam_old @ R_transform.T
        tvec_new = R_transform @ tvec_old + t_transform

        Why transpose for R_cam?
            Camera extrinsics are "left-multiplied" by the frame change.
            The camera doesn't move in physical space; we're just expressing
            its pose in a different coordinate system.

    Parameters
    ----------
    new_origin_position : np.ndarray
        Position of new origin in old frame (3,)
    new_axes_orientation : np.ndarray
        Orientation of new axes in old frame (3,3)
        Columns are new X, Y, Z axes expressed in old frame coordinates

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        rotation_transform: R_transform = new_axes_orientation.T (3,3)
                           Transforms old frame coordinates to new frame
        translation_transform: t_transform = -R_transform @ new_origin (3,)
                              Translation offset for the frame change

    Why Transpose?
        new_axes_orientation expresses new axes in old frame (new→old)
        We want to transform old coords to new coords (old→new)
        Therefore: use inverse (transpose) of the orientation matrix
    """
    # Rotation transform: transpose of axes orientation (inverse for rotation matrices)
    rotation_transform = new_axes_orientation.T

    # Translation transform: accounts for origin shift
    translation_transform = -rotation_transform @ new_origin_position

    return rotation_transform, translation_transform


def detect_board_in_final_frames(
    video_lists: List[List[str]],
    board,
    camera_group: CameraGroup,
    duration_seconds: float = 3.0,
    min_corners: int = 10
) -> dict:
    """
    Detect calibration board in the final frames of calibration videos.

    Parameters
    ----------
    video_lists : List[List[str]]
        List of video lists (one per camera)
    board : CharucoBoard
        Board object for detection
    camera_group : CameraGroup
        Calibrated camera group
    duration_seconds : float, optional
        Duration in seconds to analyze from end of videos (default: 3.0)
    min_corners : int, optional
        Minimum number of corners required for valid detection (default: 10)

    Returns
    -------
    dict
        Dictionary mapping camera_idx to list of detections, where each detection is:
        {
            'frame_idx': int,
            'corners_2d': np.ndarray (N, 2),
            'ids': np.ndarray (N,),
            'rvec': np.ndarray (3, 1),
            'tvec': np.ndarray (3, 1)
        }
    """
    all_detections = {}

    for cam_idx, video_list in enumerate(video_lists):
        video_path = video_list[0]
        cam = camera_group.cameras[cam_idx]
        cam_name = cam.get_name()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            continue

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame range for final N seconds
        frames_to_analyze = int(fps * duration_seconds)
        start_frame = max(0, total_frames - frames_to_analyze)

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        detections_for_cam = []
        frame_idx = start_frame

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect board
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection_result = board.detect_image(gray)

            if detection_result is not None:
                # Parse detection result
                if isinstance(detection_result, tuple):
                    corners_2d, ids = detection_result
                    if corners_2d is not None and len(corners_2d) >= min_corners:
                        corners_2d = corners_2d.reshape(-1, 2)
                        ids = ids.flatten().astype(int)
                    else:
                        frame_idx += 1
                        continue
                else:
                    if len(detection_result) >= min_corners:
                        corners_2d = detection_result[:, :2]
                        ids = detection_result[:, 2].astype(int)
                    else:
                        frame_idx += 1
                        continue

                # Get 3D board points for detected corners
                board_points_3d = np.array([board.get_object_points()[cid] for cid in ids])

                # Estimate board pose in camera coordinates
                try:
                    success, rvec, tvec = cv2.solvePnP(
                        board_points_3d,
                        corners_2d,
                        cam.matrix,
                        cam.dist,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        detections_for_cam.append({
                            'frame_idx': frame_idx,
                            'corners_2d': corners_2d,
                            'ids': ids,
                            'rvec': rvec,
                            'tvec': tvec,
                            'n_corners': len(corners_2d)
                        })
                except cv2.error:
                    pass

            frame_idx += 1

        cap.release()

        if detections_for_cam:
            all_detections[cam_idx] = detections_for_cam
            print(f"    {cam_name}: {len(detections_for_cam)} detections")
        else:
            print(f"    {cam_name}: No detections")

    return all_detections


def transform_board_pose_to_world(
    rvec_board_in_cam: np.ndarray,
    tvec_board_in_cam: np.ndarray,
    cam_rvec: np.ndarray,
    cam_tvec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform board pose from camera coordinates to world coordinates.

    This function composes two transformations:
    1. Board to camera: (rvec_board_in_cam, tvec_board_in_cam)
    2. Camera to world: inverse of (cam_rvec, cam_tvec)

    Transformation Chain:
        board_to_world = cam_to_world ∘ board_to_cam

    Parameters
    ----------
    rvec_board_in_cam : np.ndarray
        Board rotation vector in camera coordinates (3, 1) or (3,)
    tvec_board_in_cam : np.ndarray
        Board translation vector in camera coordinates (3, 1) or (3,)
    cam_rvec : np.ndarray
        Camera rotation vector (world to camera) (3, 1) or (3,)
    cam_tvec : np.ndarray
        Camera translation vector (world origin in camera coords) (3, 1) or (3,)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        rvec_board_in_world: Board rotation vector in world coordinates (3, 1)
        tvec_board_in_world: Board translation vector in world coordinates (3, 1)
    """
    # STEP 1: Extract camera pose in world coordinates
    # -------------------------------------------------
    # Use helper function to get camera position and orientation
    cam_position_in_world, cam_to_world_rotation = extract_camera_world_pose(
        cam_rvec, cam_tvec
    )

    # STEP 2: Transform board rotation from camera to world frame
    # -----------------------------------------------------------
    # Convert board rotation vector to matrix
    board_to_cam_rotation, _ = cv2.Rodrigues(rvec_board_in_cam)

    # Rotation composition: board_to_world = cam_to_world ∘ board_to_cam
    # Transforms are composed right-to-left: first board_to_cam, then cam_to_world
    board_to_world_rotation = cam_to_world_rotation @ board_to_cam_rotation

    # STEP 3: Transform board position from camera to world frame
    # -----------------------------------------------------------
    # Board position in camera frame
    board_position_in_cam = tvec_board_in_cam.flatten()

    # Transform to world frame:
    # 1. Rotate the vector from camera frame to world frame
    # 2. Add the camera's position in world frame
    board_position_in_world = (
        cam_position_in_world +
        cam_to_world_rotation @ board_position_in_cam
    )

    # STEP 4: Convert back to OpenCV format (rotation vectors)
    # --------------------------------------------------------
    rvec_board_in_world, _ = cv2.Rodrigues(board_to_world_rotation)
    tvec_board_in_world = board_position_in_world.reshape(3, 1)

    return rvec_board_in_world, tvec_board_in_world


def compute_board_reference_frame(
    detections: dict,
    camera_group: CameraGroup,
    board
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the average board pose in world coordinates from multiple detections.

    This function averages board positions and orientations detected across multiple
    cameras and frames. It returns the RAW averaged board frame without any coordinate
    convention adjustments (e.g., Z-axis flipping).

    Mathematical Approach:
        - Position: Simple arithmetic mean of board positions
        - Orientation: SVD-based averaging to find nearest rotation matrix

    Parameters
    ----------
    detections : dict
        Dictionary from detect_board_in_final_frames()
        Maps camera_idx → list of detection dictionaries
    camera_group : CameraGroup
        Calibrated camera group with camera extrinsics
    board : CharucoBoard
        Board object (currently unused, kept for API compatibility)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        board_center_in_world: Board center position in world coordinates (3,)
        board_orientation_in_world: Board orientation matrix in world (3, 3)
                                   As detected (OpenCV 4.6.0+: Z points INTO board)
                                   Columns are board X, Y, Z axes in world frame

    Note:
        OpenCV 4.6.0+ with CW ordering: Board Z points INTO the board (away from cameras).
        For overhead cameras, this means Z points downward (into ground).
        The caller must apply a 180° rotation to flip Z upward for typical world frame convention.
    """
    # STEP 1: Collect all board pose detections in world coordinates
    # ---------------------------------------------------------------
    board_positions_in_world = []
    board_orientations_in_world = []

    for cam_idx, cam_detections in detections.items():
        cam = camera_group.cameras[cam_idx]

        for detection in cam_detections:
            # Transform each board detection from camera coordinates to world coordinates
            rvec_board_in_world, tvec_board_in_world = transform_board_pose_to_world(
                detection['rvec'],
                detection['tvec'],
                cam.rvec,
                cam.tvec
            )

            # Store position
            board_positions_in_world.append(tvec_board_in_world.flatten())

            # Convert rotation vector to matrix and store
            board_to_world_rotation, _ = cv2.Rodrigues(rvec_board_in_world)
            board_orientations_in_world.append(board_to_world_rotation)

    if not board_positions_in_world:
        raise ValueError(
            "No valid board detections found for coordinate frame computation"
        )

    # STEP 2: Average board position (simple arithmetic mean)
    # -------------------------------------------------------
    board_center_in_world = np.mean(board_positions_in_world, axis=0)

    # STEP 3: Average board orientation (SVD-based rotation averaging)
    # ----------------------------------------------------------------
    # Use helper function to properly average rotation matrices
    board_orientation_in_world = average_rotation_matrices_svd(
        board_orientations_in_world
    )

    return board_center_in_world, board_orientation_in_world


def apply_coordinate_transform(
    camera_group: CameraGroup,
    translation_old_origin_in_new_frame: np.ndarray,
    rotation_old_frame_to_new_frame: np.ndarray
) -> CameraGroup:
    """
    Apply a coordinate frame transformation to all cameras in the group.

    This function changes the world coordinate frame used to express camera poses.
    Camera positions in physical 3D space don't change; we're just expressing them
    in a different coordinate system.

    Transformation Formulas:
        For camera extrinsics (which represent world-to-camera transforms):
            R_cam_new = R_cam_old @ R_transform.T
            tvec_new = R_transform @ tvec_old + t_transform

        Why transpose for R_cam?
            Camera extrinsics transform world points to camera points.
            When the world frame changes, the camera's rotation matrix must be
            composed with the inverse of the frame change rotation.

    Parameters
    ----------
    camera_group : CameraGroup
        Camera group to transform (modified in place)
    translation_old_origin_in_new_frame : np.ndarray
        Where the old origin ends up in new frame coordinates (3,)
        Computed as: -R_transform @ old_origin_position
    rotation_old_frame_to_new_frame : np.ndarray
        Rotation matrix that transforms old frame to new frame (3, 3)
        Typically the transpose of the new frame's axes in old frame

    Returns
    -------
    CameraGroup
        Transformed camera group (same object as input, modified in place)

    Example:
        Changing from arbitrary calibration frame to board-centric frame:
            rotation = board_axes.T  # Old coords to new coords
            translation = -rotation @ board_center_in_old_frame
            apply_coordinate_transform(camera_group, translation, rotation)
    """
    for cam in camera_group.cameras:
        if cam.rvec is None or cam.tvec is None:
            continue

        # STEP 1: Convert camera rotation vector to matrix
        # ------------------------------------------------
        world_to_cam_in_old_frame, _ = cv2.Rodrigues(cam.rvec)

        # STEP 2: Transform camera rotation to new world frame
        # ----------------------------------------------------
        # Correct formula: R_cam_new = R_cam_old @ R_transform.T
        # The transpose accounts for the inverse direction of the transformation
        world_to_cam_in_new_frame = (
            world_to_cam_in_old_frame @ rotation_old_frame_to_new_frame.T
        )

        # STEP 3: Transform camera translation to new world frame
        # -------------------------------------------------------
        # Correct formula: tvec_new = -R_cam_old @ R_transform.T @ t_transform + tvec_old
        # This ensures camera position transforms correctly as a 3D point
        tvec_old_origin_in_cam = cam.tvec.flatten()
        tvec_new_origin_in_cam = (
            -world_to_cam_in_old_frame @ rotation_old_frame_to_new_frame.T @ translation_old_origin_in_new_frame +
            tvec_old_origin_in_cam
        )

        # STEP 4: Convert back to OpenCV format and update
        # ------------------------------------------------
        rvec_new, _ = cv2.Rodrigues(world_to_cam_in_new_frame)
        cam.set_rotation(rvec_new)
        cam.set_translation(tvec_new_origin_in_cam.reshape(3, 1))

    return camera_group


def reorient_to_board_frame(
    camera_group: CameraGroup,
    video_lists: List[List[str]],
    board,
    duration_seconds: float = 3.0,
    min_corners: int = 10,
    verbose: bool = True
) -> CameraGroup:
    """
    Reorient the coordinate frame so the origin is at the calibration board center
    with the XY plane coplanar with the board and Z-axis pointing upward.

    This function should be called after calibration is complete. It analyzes the
    final frames of the calibration videos where the board is placed on the ground,
    estimates the board's pose, and transforms the entire camera system to use the
    board as the coordinate frame origin.

    Parameters
    ----------
    camera_group : CameraGroup
        Calibrated camera group (will be modified in place)
    video_lists : List[List[str]]
        List of video lists used for calibration (one per camera)
    board : CharucoBoard
        Board object used for calibration
    duration_seconds : float, optional
        Duration in seconds to analyze from end of videos (default: 3.0)
    min_corners : int, optional
        Minimum corners required per detection (default: 10)
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    CameraGroup
        Transformed camera group (same object as input, modified in place)

    Examples
    --------
    >>> # After calibration
    >>> camera_group, results = run_calibration_pipeline('projects/calibration')
    >>> # Reorient to board frame
    >>> camera_group = reorient_to_board_frame(
    ...     camera_group,
    ...     video_lists,
    ...     board,
    ...     duration_seconds=3.0
    ... )
    """
    if verbose:
        print("\n  Reorienting coordinate frame to board placement...")
        print(f"    Analyzing final {duration_seconds} seconds of videos")

    # Step 1: Detect board in final frames
    detections = detect_board_in_final_frames(
        video_lists,
        board,
        camera_group,
        duration_seconds=duration_seconds,
        min_corners=min_corners
    )

    if not detections:
        raise ValueError(
            f"No board detections found in final {duration_seconds}s of videos. "
            "Ensure the board is visible to multiple cameras in the final frames."
        )

    total_detections = sum(len(dets) for dets in detections.values())
    if verbose:
        print(f"    Total detections: {total_detections} across {len(detections)} cameras")

    # Step 2: Compute average board pose in calibration frame
    if verbose:
        print("    Computing average board pose...")

    board_center_in_calib, board_orientation_detected = compute_board_reference_frame(
        detections,
        camera_group,
        board
    )

    if verbose:
        print(f"    Board center (calibration frame): [{board_center_in_calib[0]:.2f}, "
              f"{board_center_in_calib[1]:.2f}, {board_center_in_calib[2]:.2f}]")

    # Step 3: Apply 180° rotation to flip Z-axis upward
    # ---------------------------------------------------
    # OpenCV 4.6.0+ (CW): Board Z points INTO the board (downward, into ground).
    # We want world Z pointing upward (away from ground).
    # Solution: Rotate 180° around X-axis (proper rotation, det=+1).
    # This flips both Y and Z axes, making Z point upward.
    if verbose:
        print("    Flipping board Z-axis for desired world frame...")

    board_orientation_as_world = apply_board_z_axis_flip(board_orientation_detected)

    # Step 4: Compute coordinate frame transformation parameters
    # ----------------------------------------------------------
    # Transform from calibration frame to board-centric frame
    if verbose:
        print("    Computing coordinate transformation...")

    rotation_calib_to_board, translation_calib_to_board = compute_frame_change_transform(
        new_origin_position=board_center_in_calib,
        new_axes_orientation=board_orientation_as_world
    )

    # Step 5: Apply transformation to all camera extrinsics
    # -----------------------------------------------------
    if verbose:
        print("    Applying transformation to all cameras...")

    apply_coordinate_transform(
        camera_group,
        translation_old_origin_in_new_frame=translation_calib_to_board,
        rotation_old_frame_to_new_frame=rotation_calib_to_board
    )

    if verbose:
        print("    Reorientation complete!")
        print("      Origin: board center | XY plane: board plane | Z axis: upward")

    return camera_group


# ============================================================================
# 10. Pipeline Orchestration
# ============================================================================

DEFAULT_VIZ_CONFIG = {
    'extrinsics_output': None,
    'heatmap_output': None,
    'frustum_scale': 200,
    'frame_step': 10,  # Sample every 10th frame by default
    'show_axes': True,
    'show_labels': True,
    'extrinsics_figsize': (14, 12),
    'heatmap_figsize': (16, 10)
}


def _merge_visualization_config(user_config: Optional[dict], project_folder: str) -> dict:
    """
    Merge user visualization config with defaults and set auto-paths.

    Parameters
    ----------
    user_config : dict, optional
        User-provided visualization configuration
    project_folder : str
        Path to project folder for auto-generating output paths

    Returns
    -------
    dict
        Merged configuration with all required keys
    """
    config = DEFAULT_VIZ_CONFIG.copy()
    if user_config:
        config.update(user_config)

    # Set default paths if not provided
    viz_dir = Path(project_folder) / 'output' / 'visualizations'
    if config['extrinsics_output'] is None:
        config['extrinsics_output'] = str(viz_dir / 'camera_extrinsics.png')
    if config['heatmap_output'] is None:
        config['heatmap_output'] = str(viz_dir / 'reprojection_error_heatmap.png')

    return config


def run_calibration_pipeline(
    project_folder: str,
    config_path: Optional[str] = None,
    generate_board_image: bool = True,
    generate_visualizations: bool = True,
    reorient_to_board: bool = True,
    reorient_duration_seconds: float = 3.0,
    visualization_config: Optional[dict] = None,
    video_extension: Optional[str] = None,
    verbose: bool = True,
    metadata: Optional[dict] = None
) -> Tuple[CameraGroup, dict]:
    """
    Run complete multi-camera calibration pipeline with optional visualizations.

    This is the main entry point for calibration workflows. It orchestrates all
    steps from configuration loading through calibration to validation and visualization.

    Parameters
    ----------
    project_folder : str
        Path to project folder (e.g., 'projects/calibration_121125')
    config_path : str, optional
        Path to config.toml file. If None, uses {project_folder}/config.toml
    generate_board_image : bool, optional
        Whether to generate and save board.png (default: True)
    generate_visualizations : bool, optional
        Whether to generate visualization plots after calibration (default: True)
    visualization_config : dict, optional
        Configuration for visualizations. Supported keys:
        - 'extrinsics_output': str - Path for extrinsics plot
        - 'heatmap_output': str - Path for heatmap
        - 'frustum_scale': float - Camera frustum scale (default: 200)
        - 'max_frames': int - Max frames for error analysis (default: 500, None=all)
        - 'show_axes': bool - Show world axes (default: True)
        - 'show_labels': bool - Show camera labels (default: True)
        - 'extrinsics_figsize': tuple - Figure size for extrinsics (default: (14, 12))
        - 'heatmap_figsize': tuple - Figure size for heatmap (default: (16, 10))
    video_extension : str, optional
        Video file extension (default: from config or 'avi')
    verbose : bool, optional
        Print progress information (default: True)
    metadata : dict, optional
        Optional metadata to include in calibration file

    Returns
    -------
    Tuple[CameraGroup, dict]
        - camera_group: Calibrated CameraGroup object
        - results: Dictionary containing:
            - 'metrics': Validation metrics
            - 'calibration_path': Path to calibration.toml
            - 'summary_path': Path to calibration summary
            - 'board_image_path': Path to board.png (if generated)
            - 'extrinsics_plot_path': Path to extrinsics plot (if generated)
            - 'heatmap_plot_path': Path to heatmap plot (if generated)
            - 'config': Loaded configuration
            - 'board': Board object used

    Examples
    --------
    Basic usage with all defaults:
    >>> camera_group, results = run_calibration_pipeline('projects/calibration_121125')

    Disable visualizations for faster processing:
    >>> camera_group, results = run_calibration_pipeline(
    ...     'projects/calibration_121125',
    ...     generate_visualizations=False
    ... )

    Custom visualization settings:
    >>> viz_config = {
    ...     'frustum_scale': 300,
    ...     'max_frames': 1000,
    ...     'extrinsics_output': 'custom/path/cameras.png'
    ... }
    >>> camera_group, results = run_calibration_pipeline(
    ...     'projects/calibration_121125',
    ...     visualization_config=viz_config
    ... )
    Quiet mode without visualizations:
    >>> camera_group, results = run_calibration_pipeline(
    ...     'projects/calibration_121125',
    ...     generate_visualizations=False,
    ...     verbose=False
    ... )
    """
    # Resolve paths
    project_path = Path(project_folder)
    if config_path is None:
        config_path = project_path / 'config.toml'
    else:
        config_path = Path(config_path)

    # Initialize results dictionary
    results = {}

    if verbose:
        print(f"\nStarting calibration for project: {project_folder}")
        print("=" * 60)

    # Step 1/8: Load configuration
    if verbose:
        print("\n[1/8] Loading configuration...")
    config = load_config(str(config_path))
    results['config'] = config
    if verbose:
        print(f"   Loaded config from {config_path}")

    # Step 2/8: Create board
    if verbose:
        print("\n[2/8] Creating calibration board...")
    board = create_board_from_config(config)
    results['board'] = board
    if verbose:
        board_params = config.get('calibration', {})
        print(f"   Created {board_params.get('board_type', 'charuco')} board")
        print(f"    Size: {board_params.get('board_size', [])}")
        print(f"    Square length: {board_params.get('board_square_side_length')} mm")

    # Step 3/8: Save board image (optional)
    if generate_board_image:
        if verbose:
            print("\n[3/8] Generating board image...")
        board_image_path = project_path / 'output' / 'board.png'
        save_board_image(board, str(board_image_path))
        results['board_image_path'] = str(board_image_path)
        if verbose:
            print(f"   Saved board image to {board_image_path}")
    else:
        if verbose:
            print("\n[3/8] Skipping board image generation")
        results['board_image_path'] = None

    # Step 4/8: Discover videos
    if verbose:
        print("\n[4/8] Discovering calibration videos...")
    if video_extension is None:
        video_extension = config.get('video_extension', 'avi')
    video_paths = discover_calibration_videos(project_folder, video_extension)

    if not video_paths:
        raise ValueError(f"No videos found in {project_path / 'videos'}")

    if verbose:
        print(f"   Found {len(video_paths)} video(s)")
        for vp in video_paths:
            print(f"    - {Path(vp).name}")

    # Step 5/8: Extract camera names
    if verbose:
        print("\n[5/8] Extracting camera names...")
    cam_regex = config.get('triangulation', {}).get('cam_regex', r'(\w+)')
    cam_names = extract_camera_names(video_paths, cam_regex)
    if verbose:
        print(f"   Extracted {len(cam_names)} camera name(s):")
        for cn in cam_names:
            print(f"    - {cn}")

    # Organize videos
    video_lists = organize_videos_by_camera(video_paths)

    # Step 6/8: Calibrate
    if verbose:
        print("\n[6/8] Running calibration...")
        print("  This may take 10-20 minutes depending on video length and number of cameras...")

    fisheye = config.get('calibration', {}).get('fisheye', False)
    camera_group, bundle_error = calibrate_cameras(
        video_lists=video_lists,
        board=board,
        cam_names=cam_names,
        fisheye=fisheye,
        verbose=verbose
    )
    results['bundle_adjustment_error'] = bundle_error
    if verbose:
        print("   Calibration complete")
        print(f"   Bundle adjustment error: {bundle_error:.4f} pixels")

    # Step 7/8: Validate and save
    if verbose:
        print("\n[7/8] Validating and saving calibration...")
    metrics = validate_calibration(camera_group)
    results['metrics'] = metrics

    if verbose:
        print(f"   Calibrated {metrics['n_cameras']} camera(s)")
        print(f"  Intrinsics initialized: {metrics['all_intrinsics_initialized']}")
        print(f"  Extrinsics initialized: {metrics['all_extrinsics_initialized']}")

        if metrics['focal_lengths']:
            print("\n  Focal lengths (fx, fy):")
            for cam_name, (fx, fy) in metrics['focal_lengths'].items():
                print(f"    {cam_name}: ({fx:.2f}, {fy:.2f})")

    # Save calibration
    calib_output = project_path / 'calibration.toml'
    save_calibration(camera_group, project_folder, metadata=metadata)
    results['calibration_path'] = str(calib_output)
    if verbose:
        print(f"\n   Saved calibration to {calib_output}")

    # Merge visualization config to get frame_step parameter
    viz_config = _merge_visualization_config(visualization_config, project_folder)
    validation_frame_step = viz_config.get('frame_step', 10)

    # Compute per-frame validation error (will be reused for heatmap if visualizations enabled)
    if verbose:
        print(f"  Computing per-frame validation error (sampling every {validation_frame_step}th frame)...")
    try:
        error_matrix, camera_names_list, frame_indices = compute_reprojection_errors_per_frame(
            camera_group, video_lists, board, frame_step=validation_frame_step
        )
        # Compute overall mean error from all valid measurements
        valid_errors = error_matrix[~np.isnan(error_matrix)]
        per_frame_validation_error = valid_errors.mean() if len(valid_errors) > 0 else None
        if verbose and per_frame_validation_error is not None:
            print(f"   Per-frame validation error: {per_frame_validation_error:.4f} pixels")

        # Store for reuse in visualization to avoid redundant computation
        precomputed_errors = (error_matrix, camera_names_list, frame_indices)
    except Exception as e:
        if verbose:
            print(f"   Warning: Failed to compute per-frame validation error: {e}")
        per_frame_validation_error = None
        precomputed_errors = None

    # Save calibration summary
    summary_path = project_path / 'output' / 'calibration_summary.txt'
    save_calibration_summary(
        camera_group,
        config,
        str(summary_path),
        bundle_adjustment_error=bundle_error,
        per_frame_validation_error=per_frame_validation_error
    )
    results['summary_path'] = str(summary_path)
    if verbose:
        print(f"   Saved calibration summary to {summary_path}")

    # Optional: Reorient to board frame
    if reorient_to_board:
        if verbose:
            print("\n[7b/8] Reorienting coordinate frame to board placement...")

        # Make a deep copy of camera extrinsics before reorienting
        # (intrinsics remain the same, only extrinsics change)
        import copy
        camera_group_backup = copy.deepcopy(camera_group)

        try:
            # Reorient the camera group
            reorient_to_board_frame(
                camera_group,
                video_lists,
                board,
                duration_seconds=reorient_duration_seconds,
                verbose=verbose
            )

            # Save reoriented calibration to separate file
            calib_reoriented_output = project_path / 'calibration_reoriented.toml'
            camera_group.dump(str(calib_reoriented_output))
            results['calibration_reoriented_path'] = str(calib_reoriented_output)

            if verbose:
                print(f"   Saved reoriented calibration to {calib_reoriented_output}")

        except Exception as e:
            if verbose:
                print(f"   Warning: Reorientation failed: {e}")
                print("   Proceeding with original calibration for visualizations")
            # Restore original calibration if reorientation failed
            camera_group = camera_group_backup
            results['calibration_reoriented_path'] = None
    else:
        results['calibration_reoriented_path'] = None

    # Step 8/8: Generate visualizations (optional)
    if generate_visualizations:
        if verbose:
            print("\n[8/8] Generating visualizations...")

        # Create output directory
        viz_dir = Path(viz_config['extrinsics_output']).parent
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Generate camera extrinsics 3D plot
        if verbose:
            print("  Generating 3D camera extrinsics plot...")
        try:
            plot_camera_extrinsics(
                camera_group,
                output_path=viz_config['extrinsics_output'],
                frustum_scale=viz_config['frustum_scale'],
                show_axes=viz_config['show_axes'],
                show_labels=viz_config['show_labels'],
                figsize=viz_config['extrinsics_figsize']
            )
            results['extrinsics_plot_path'] = viz_config['extrinsics_output']
            if verbose:
                print(f"   Saved to {viz_config['extrinsics_output']}")
        except Exception as e:
            if verbose:
                print(f"   Warning: Failed to generate extrinsics plot: {e}")
            results['extrinsics_plot_path'] = None

        # Generate reprojection error heatmap (reuse precomputed errors)
        if verbose:
            print("  Generating reprojection error heatmap...")
        try:
            plot_reprojection_error_heatmap(
                camera_group,
                video_lists,
                board,
                output_path=viz_config['heatmap_output'],
                frame_step=validation_frame_step,
                precomputed_errors=precomputed_errors,  # Reuse computation from summary
                figsize=viz_config['heatmap_figsize']
            )
            results['heatmap_plot_path'] = viz_config['heatmap_output']
            if verbose:
                print(f"   Saved to {viz_config['heatmap_output']}")
        except Exception as e:
            if verbose:
                print(f"   Warning: Failed to generate error heatmap: {e}")
            results['heatmap_plot_path'] = None
    else:
        if verbose:
            print("\n[8/8] Skipping visualization generation")
        results['extrinsics_plot_path'] = None
        results['heatmap_plot_path'] = None

    if verbose:
        print("\n" + "=" * 60)
        print("Calibration pipeline complete!")

    return camera_group, results


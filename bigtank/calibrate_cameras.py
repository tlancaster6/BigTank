"""
Multi-camera calibration module using aniposelib.

This module provides modular functions for performing multi-camera calibration
using ChArUco boards. Functions are designed to be imported and composed in
other scripts.
"""

import re
from pathlib import Path
from typing import List, Optional

import cv2

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib_fallback
    tomllib = None

from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup


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
) -> CameraGroup:
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
    CameraGroup
        Calibrated camera group
    """
    # Create camera group
    cgroup = create_camera_group(cam_names, fisheye=fisheye)

    # Perform calibration
    cgroup.calibrate_videos(
        video_lists,
        board,
        init_intrinsics=init_intrinsics,
        init_extrinsics=init_extrinsics,
        verbose=verbose
    )

    return cgroup


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
    reprojection_error: Optional[float] = None
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
    reprojection_error : float, optional
        Mean reprojection error in pixels if available
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

    # Reprojection error
    if reprojection_error is not None:
        lines.append("-" * 70)
        lines.append("CALIBRATION QUALITY")
        lines.append("-" * 70)
        lines.append(f"Mean reprojection error:  {reprojection_error:.4f} pixels")
        lines.append("")

        # Quality assessment
        if reprojection_error < 0.5:
            quality = "Excellent"
        elif reprojection_error < 1.0:
            quality = "Good"
        elif reprojection_error < 2.0:
            quality = "Acceptable"
        else:
            quality = "Poor - consider recalibration"
        lines.append(f"Quality assessment:       {quality}")
        lines.append("")

    lines.append("=" * 70)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


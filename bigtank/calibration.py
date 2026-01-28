"""
Core calibration functions for multi-camera calibration using aniposelib.

This module provides functions for video discovery, camera calibration,
persistence, and validation.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from aniposelib.cameras import CameraGroup

from .config import extract_calibration_params


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
# Video Discovery and Organization
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
# Camera Group Management
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
# Calibration Execution
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
# Calibration Persistence
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
# Validation and Metrics
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

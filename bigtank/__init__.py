"""
BigTank - Multi-camera calibration tools using aniposelib.

This package provides modular functions for performing multi-camera calibration
using ChArUco boards. Functions are designed to be imported and composed in
other scripts.

Example
-------
>>> from bigtank import run_calibration_pipeline
>>> camera_group, results = run_calibration_pipeline('projects/my_calibration')

Or import specific functions:

>>> from bigtank.calibration import calibrate_cameras, load_calibration
>>> from bigtank.visualization import plot_camera_extrinsics
"""

# Configuration management
from .config import (
    load_config,
    extract_calibration_params,
    get_config_value,
    create_board_from_config,
    save_board_image,
)

# Core calibration functions
from .calibration import (
    discover_calibration_videos,
    extract_camera_names,
    organize_videos_by_camera,
    create_camera_group,
    calibrate_cameras,
    save_calibration,
    load_calibration,
    validate_calibration,
    compute_reprojection_error,
    save_calibration_summary,
)

# Visualization functions
from .visualization import (
    Arrow3D,
    plot_camera_extrinsics,
    compute_reprojection_errors_per_frame,
    plot_reprojection_error_heatmap,
)

# Coordinate transformation functions
from .coordinates import (
    extract_camera_world_pose,
    average_rotation_matrices_svd,
    apply_board_z_axis_flip,
    compute_frame_change_transform,
    detect_board_in_final_frames,
    transform_board_pose_to_world,
    compute_board_reference_frame,
    apply_coordinate_transform,
    reorient_to_board_frame,
)

# Pipeline orchestration
from .pipeline import (
    DEFAULT_VIZ_CONFIG,
    run_calibration_pipeline,
)

__all__ = [
    # Config
    'load_config',
    'extract_calibration_params',
    'get_config_value',
    'create_board_from_config',
    'save_board_image',
    # Calibration
    'discover_calibration_videos',
    'extract_camera_names',
    'organize_videos_by_camera',
    'create_camera_group',
    'calibrate_cameras',
    'save_calibration',
    'load_calibration',
    'validate_calibration',
    'compute_reprojection_error',
    'save_calibration_summary',
    # Visualization
    'Arrow3D',
    'plot_camera_extrinsics',
    'compute_reprojection_errors_per_frame',
    'plot_reprojection_error_heatmap',
    # Coordinates
    'extract_camera_world_pose',
    'average_rotation_matrices_svd',
    'apply_board_z_axis_flip',
    'compute_frame_change_transform',
    'detect_board_in_final_frames',
    'transform_board_pose_to_world',
    'compute_board_reference_frame',
    'apply_coordinate_transform',
    'reorient_to_board_frame',
    # Pipeline
    'DEFAULT_VIZ_CONFIG',
    'run_calibration_pipeline',
]

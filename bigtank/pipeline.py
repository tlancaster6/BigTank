"""
Pipeline orchestration for multi-camera calibration.

This module provides the main entry point for running the complete
calibration workflow.
"""

import copy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from aniposelib.cameras import CameraGroup

from .config import (
    load_config,
    create_board_from_config,
    save_board_image,
)
from .calibration import (
    discover_calibration_videos,
    extract_camera_names,
    organize_videos_by_camera,
    calibrate_cameras,
    save_calibration,
    validate_calibration,
    save_calibration_summary,
)
from .visualization import (
    plot_camera_extrinsics,
    plot_reprojection_error_heatmap,
    compute_reprojection_errors_per_frame,
)
from .coordinates import reorient_to_board_frame


# ============================================================================
# Pipeline Configuration
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


# ============================================================================
# Main Pipeline
# ============================================================================

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

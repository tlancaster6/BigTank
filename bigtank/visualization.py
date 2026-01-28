"""
Visualization functions for multi-camera calibration.

This module provides functions for visualizing camera extrinsics,
reprojection errors, and other calibration metrics.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

from aniposelib.cameras import CameraGroup


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

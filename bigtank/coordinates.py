"""
Coordinate frame reorientation for multi-camera calibration.

This module provides functions for transforming camera calibration results
to a board-centric coordinate frame.

COORDINATE FRAMES REFERENCE
---------------------------

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
     Z points downward (upwards from ground, towards cameras above)

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
4. Rotate board frame 180 around X-axis (OpenCV 4.6.0+: board Z downward -> world Z upward)
5. Compute coordinate frame transformation parameters
6. Apply transformation to all camera extrinsics

After reorientation:
    - Origin at board center on ground
    - XY plane is the ground plane
    - Z-axis points upward
    - Cameras at positive Z heights, looking in -Z direction (downward)
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from aniposelib.cameras import CameraGroup


# ============================================================================
# Helper Functions for Coordinate Transformations
# ============================================================================

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
    Rotate board frame 180 around X-axis to create upward-pointing world frame.

    Problem (OpenCV 4.6.0+ with CW ordering):
        ChArUco board Z-axis points INTO board surface (away from viewer).
        For overhead cameras looking down at a board on the ground:
        - Board Z points downward (into ground)
        - We want world Z to point upward (away from ground)
        - These are OPPOSITE directions

    Solution:
        Rotate 180 around X-axis: this flips both Y and Z axes.
        This is a PROPER ROTATION (det = +1), not a reflection (det = -1).
        Using a proper rotation is critical because cv2.Rodrigues() corrupts
        reflection matrices during conversion to/from axis-angle representation.

    Before rotation:
        - Board Z: points downward (into ground)
        - Board Y: some direction in the plane or perpendicular
        - Cameras: would be at negative Z (below ground) looking upward

    After 180 rotation around X:
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
        180 rotation around X-axis:
        R_x(180) = [[1,  0,  0],
                     [0, -1,  0],
                     [0,  0, -1]]
        det(R_x) = 1 * (-1) * (-1) = +1

        Compare to reflection (INCORRECT):
        diag([1, 1, -1]) has det = -1, which breaks cv2.Rodrigues()
    """
    # 180 rotation around X-axis (proper rotation, det = +1)
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
        new_axes_orientation expresses new axes in old frame (new->old)
        We want to transform old coords to new coords (old->new)
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
        board_to_world = cam_to_world o board_to_cam

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

    # Rotation composition: board_to_world = cam_to_world o board_to_cam
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
        Maps camera_idx -> list of detection dictionaries
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
        The caller must apply a 180 rotation to flip Z upward for typical world frame convention.
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

    # Step 3: Apply 180 rotation to flip Z-axis upward
    # ---------------------------------------------------
    # OpenCV 4.6.0+ (CW): Board Z points INTO the board (downward, into ground).
    # We want world Z pointing upward (away from ground).
    # Solution: Rotate 180 around X-axis (proper rotation, det=+1).
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

#!/usr/bin/env python3
"""
Main script for running multi-camera calibration pipeline.

Usage:
    python main.py projects/calibration_121125
    python main.py projects/calibration_121125 --no-board-image
    python main.py projects/calibration_121125 --quiet
"""

import argparse
import sys
from pathlib import Path

from bigtank.calibrate_cameras import (
    load_config,
    create_board_from_config,
    save_board_image,
    discover_calibration_videos,
    extract_camera_names,
    organize_videos_by_camera,
    calibrate_cameras,
    save_calibration,
    validate_calibration,
    save_calibration_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-camera calibration pipeline using aniposelib',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py projects/calibration_121125
  python main.py projects/calibration_121125 --no-board-image --quiet
        """
    )

    parser.add_argument(
        'project_folder',
        type=str,
        help='Path to project folder (e.g., projects/calibration_121125)'
    )

    parser.add_argument(
        '--no-board-image',
        action='store_true',
        help='Skip generating board image PNG'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output during calibration'
    )

    parser.add_argument(
        '--video-extension',
        type=str,
        default=None,
        help='Video file extension (default: from config or "avi")'
    )

    args = parser.parse_args()

    # Validate project folder exists
    project_path = Path(args.project_folder)
    if not project_path.exists():
        print(f"Error: Project folder '{args.project_folder}' does not exist", file=sys.stderr)
        sys.exit(1)

    config_path = project_path / 'config.toml'
    if not config_path.exists():
        print(f"Error: Config file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Starting calibration for project: {args.project_folder}")
    print("=" * 60)

    # Load configuration
    print("\n[1/7] Loading configuration...")
    config = load_config(str(config_path))
    print(f"   Loaded config from {config_path}")

    # Create board
    print("\n[2/7] Creating calibration board...")
    board = create_board_from_config(config)
    board_params = config.get('calibration', {})
    print(f"   Created {board_params.get('board_type', 'charuco')} board")
    print(f"    Size: {board_params.get('board_size', [])}")
    print(f"    Square length: {board_params.get('board_square_side_length')} mm")

    # Save board image
    if not args.no_board_image:
        print("\n[3/7] Generating board image...")
        board_image_path = project_path / 'output' / 'board.png'
        save_board_image(board, str(board_image_path))
        print(f"   Saved board image to {board_image_path}")
    else:
        print("\n[3/7] Skipping board image generation (--no-board-image)")

    # Discover videos
    print("\n[4/7] Discovering calibration videos...")
    video_extension = args.video_extension or config.get('video_extension', 'avi')
    video_paths = discover_calibration_videos(args.project_folder, video_extension)

    if not video_paths:
        print(f"  Error: No videos found in {project_path / 'videos'}", file=sys.stderr)
        sys.exit(1)

    print(f"   Found {len(video_paths)} video(s)")
    for vp in video_paths:
        print(f"    - {Path(vp).name}")

    # Extract camera names
    print("\n[5/7] Extracting camera names...")
    cam_regex = config.get('triangulation', {}).get('cam_regex', r'(\w+)')
    cam_names = extract_camera_names(video_paths, cam_regex)
    print(f"   Extracted {len(cam_names)} camera name(s):")
    for cn in cam_names:
        print(f"    - {cn}")

    # Organize videos
    video_lists = organize_videos_by_camera(video_paths)

    # Calibrate
    print("\n[6/7] Running calibration...")
    print("  This may take 10-20 minutes depending on video length and number of cameras...")

    verbose = not args.quiet
    fisheye = config.get('calibration', {}).get('fisheye', False)

    cgroup = calibrate_cameras(
        video_lists=video_lists,
        board=board,
        cam_names=cam_names,
        fisheye=fisheye,
        verbose=verbose
    )
    print("   Calibration complete")

    # Validate
    print("\n[7/7] Validating calibration...")
    metrics = validate_calibration(cgroup)
    print(f"   Calibrated {metrics['n_cameras']} camera(s)")
    print(f"  Intrinsics initialized: {metrics['all_intrinsics_initialized']}")
    print(f"  Extrinsics initialized: {metrics['all_extrinsics_initialized']}")

    if metrics['focal_lengths']:
        print("\n  Focal lengths (fx, fy):")
        for cam_name, (fx, fy) in metrics['focal_lengths'].items():
            print(f"    {cam_name}: ({fx:.2f}, {fy:.2f})")

    # Save calibration
    calib_output = project_path / 'calibration.toml'
    save_calibration(cgroup, args.project_folder)
    print(f"\n   Saved calibration to {calib_output}")

    # Save calibration summary
    summary_path = project_path / 'output' / 'calibration_summary.txt'
    save_calibration_summary(cgroup, config, str(summary_path))
    print(f"   Saved calibration summary to {summary_path}")
    print("\n" + "=" * 60)
    print("Calibration pipeline complete!")
    print(f"Results saved to: {args.project_folder}")
    print(f"  - calibration.toml")
    print(f"  - output/calibration_summary.txt")
    if not args.no_board_image:
        print(f"  - output/board.png")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Main script for running multi-camera calibration pipeline.

Usage:
    python main.py projects/calibration_121125
    python main.py projects/calibration_121125 --no-board-image
    python main.py projects/calibration_121125 --quiet
    python main.py projects/calibration_121125 --no-visualizations
"""

import argparse
import sys
from pathlib import Path

from bigtank.calibrate_cameras import run_calibration_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-camera calibration pipeline using aniposelib',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py projects/calibration_121125
  python main.py projects/calibration_121125 --no-board-image --quiet
  python main.py projects/calibration_121125 --no-visualizations
  python main.py projects/calibration_121125 --max-frames 1000
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
        '--no-visualizations',
        action='store_true',
        help='Skip generating visualization plots (faster)'
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

    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='Max frames for reprojection error analysis (default: 0, 0=all)'
    )

    parser.add_argument(
        '--frustum-scale',
        type=float,
        default=200,
        help='Camera frustum scale for 3D visualization. Only impacts visualization, not calibration (default: 200)'
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

    # Build visualization config from CLI args
    viz_config = {
        'frustum_scale': args.frustum_scale,
        'max_frames': args.max_frames if args.max_frames > 0 else None,
    }

    # Run pipeline
    print(f"Starting calibration for project: {args.project_folder}")
    print("=" * 60)

    try:
        camera_group, results = run_calibration_pipeline(
            project_folder=args.project_folder,
            generate_board_image=not args.no_board_image,
            generate_visualizations=not args.no_visualizations,
            visualization_config=viz_config,
            video_extension=args.video_extension,
            verbose=not args.quiet
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Calibration pipeline complete!")
        print(f"Results saved to: {args.project_folder}")
        print(f"  - {results['calibration_path']}")
        print(f"  - {results['summary_path']}")
        if results.get('board_image_path'):
            print(f"  - {results['board_image_path']}")
        if results.get('extrinsics_plot_path'):
            print(f"  - {results['extrinsics_plot_path']}")
        if results.get('heatmap_plot_path'):
            print(f"  - {results['heatmap_plot_path']}")

    except Exception as e:
        print(f"\nError during calibration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

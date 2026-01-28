"""
Configuration management and board creation for multi-camera calibration.

This module provides functions for loading configuration files and creating
CharucoBoard objects for calibration.
"""

from pathlib import Path

import cv2

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib_fallback
    tomllib = None

from aniposelib.boards import CharucoBoard


# ============================================================================
# Configuration Management
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
# Board Creation
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

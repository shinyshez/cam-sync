import pytest
from unittest.mock import patch, MagicMock

from server.entrypoint import parse_mode


def test_two_args_is_mode_c():
    """Two positional args → Mode C (cli)."""
    mode, args = parse_mode(["entrypoint", "/data/vid1.mp4", "/data/vid2.mp4"])
    assert mode == "cli"
    assert args["vid1"] == "/data/vid1.mp4"
    assert args["vid2"] == "/data/vid2.mp4"


def test_no_args_is_mode_d():
    """No positional args → Mode D (webapp)."""
    mode, args = parse_mode(["entrypoint"])
    assert mode == "webapp"
    assert "vid1" not in args


def test_test_flag():
    """--test flag is detected."""
    mode, args = parse_mode(["entrypoint", "--test"])
    assert mode == "test"


def test_optional_params_mode_c():
    """Mode C can accept --sample-fps and --band-fraction."""
    mode, args = parse_mode([
        "entrypoint", "/data/vid1.mp4", "/data/vid2.mp4",
        "--sample-fps", "5.0", "--band-fraction", "0.3"
    ])
    assert mode == "cli"
    assert args["sample_fps"] == 5.0
    assert args["band_fraction"] == 0.3

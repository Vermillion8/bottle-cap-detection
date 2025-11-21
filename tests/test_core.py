import pytest
from typer.testing import CliRunner

from bsort.data import HSV_RANGES, classify_hue
from bsort.main import app

runner = CliRunner()


# --- 1. Test the Math (Unit Tests) ---
def test_classify_light_blue():
    """
    Verifies that a hue value inside the 'light_blue' range
    returns Class 0.
    """
    # Pick a number strictly inside your Light Blue range (e.g., 95)
    # Assuming range is approx (85, 105)
    low, high = HSV_RANGES["light_blue"]
    mid_point = (low + high) / 2

    assert classify_hue(mid_point) == 0, f"Hue {mid_point} should be Light Blue (0)"


def test_classify_dark_blue():
    """
    Verifies that a hue value inside the 'dark_blue' range
    returns Class 1.
    """
    low, high = HSV_RANGES["dark_blue"]
    mid_point = (low + high) / 2

    assert classify_hue(mid_point) == 1, f"Hue {mid_point} should be Dark Blue (1)"


def test_classify_other():
    """
    Verifies that a hue value OUTSIDE your blue ranges
    returns Class 2.
    """
    # Red is typically around 0 or 170 in OpenCV HSV
    assert classify_hue(0) == 2, "Hue 0 (Red) should be Class 2 (Other)"
    assert classify_hue(170) == 2, "Hue 170 (Red) should be Class 2 (Other)"


# --- 2. Test the Application (Integration Test) ---
def test_cli_help():
    """
    Verifies that the CLI app launches and lists the available commands.
    """
    result = runner.invoke(app, ["--help"])

    # 1. Check if the app exited successfully (Code 0 means OK)
    assert result.exit_code == 0

    # 2. Check if our key commands are listed in the help output
    # This proves main.py loaded and registered the functions correctly
    assert "train" in result.stdout
    assert "infer" in result.stdout

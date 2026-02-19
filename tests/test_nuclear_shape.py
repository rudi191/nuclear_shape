"""
Test the core nuclear_shape functionality using synthetic data.

Tests are self-contained (no repo or test files required), so they work
after a plain `pip install nuclear_shape`.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from nuclear_shape import nuclear_shape


def _make_synthetic_cmm(num_points=100, seed=42):
    """Create a temporary .cmm file with synthetic ellipsoid-like point cloud."""
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    x = 5 * np.sin(phi) * np.cos(theta)
    y = 3 * np.sin(phi) * np.sin(theta)
    z = 2 * np.cos(phi)

    root = ET.Element("marker_set")
    for i in range(num_points):
        marker = ET.SubElement(root, "marker")
        marker.set("id", str(i))
        marker.set("x", str(x[i]))
        marker.set("y", str(y[i]))
        marker.set("z", str(z[i]))

    fd, path = tempfile.mkstemp(suffix=".cmm")
    ET.ElementTree(root).write(path)
    return path


@pytest.fixture
def cmm_path():
    """Synthetic .cmm file; no repo or test data needed."""
    path = _make_synthetic_cmm()
    yield path
    Path(path).unlink(missing_ok=True)


def test_nuclear_shape_initialization(cmm_path):
    """Test that nuclear_shape can be initialized with a .cmm file."""
    shape = nuclear_shape(cmm_path)

    assert hasattr(shape, "matrix")
    assert hasattr(shape, "center")
    assert hasattr(shape, "results")
    assert shape.matrix.shape[1] == 3
    assert len(shape.matrix.shape) == 2


def test_ellipsoid_fit(cmm_path):
    """Test ellipsoid fitting."""
    shape = nuclear_shape(cmm_path)
    shape.ellipsoid_fit()

    assert "ellipsoid" in shape.results
    assert "center" in shape.results["ellipsoid"]
    assert "axes" in shape.results["ellipsoid"]
    assert "sphericity" in shape.results["ellipsoid"]

    sphericity = shape.results["ellipsoid"]["sphericity"]
    assert 0 <= sphericity["sphericity_volume"] <= 1
    assert 0 <= sphericity["sphericity_axes"] <= 1


def test_ellipsoid_inner(cmm_path):
    """Test inner ellipsoid computation."""
    shape = nuclear_shape(cmm_path)
    shape.ellipsoid_inner()

    assert "ellipsoid_inner" in shape.results
    assert "center" in shape.results["ellipsoid_inner"]
    assert "radii" in shape.results["ellipsoid_inner"]


def test_ellipsoid_outer(cmm_path):
    """Test outer ellipsoid computation."""
    shape = nuclear_shape(cmm_path)
    shape.ellipsoid_outer()

    assert "ellipsoid_outer" in shape.results
    assert "center" in shape.results["ellipsoid_outer"]
    assert "radii" in shape.results["ellipsoid_outer"]


def test_principal_components(cmm_path):
    """Test PCA computation."""
    shape = nuclear_shape(cmm_path)
    shape.principal_components()

    assert "PCA" in shape.results
    assert "components" in shape.results["PCA"]
    assert "variance" in shape.results["PCA"]
    assert len(shape.results["PCA"]["variance"]) == 3


def test_full_pipeline(cmm_path):
    """Test running the full analysis pipeline."""
    shape = nuclear_shape(cmm_path)
    shape.ellipsoid_fit()
    shape.ellipsoid_inner()
    shape.ellipsoid_outer()
    shape.principal_components()

    expected_keys = [
        "ellipsoid",
        "ellipsoid_inner",
        "ellipsoid_outer",
        "PCA",
    ]
    for key in expected_keys:
        assert key in shape.results, f"Missing result key: {key}"

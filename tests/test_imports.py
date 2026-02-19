"""
Test that all imports work correctly after installation.
"""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import nuclear_shape
    assert hasattr(nuclear_shape, 'nuclear_shape')
    assert hasattr(nuclear_shape, 'plotting_results')


def test_main_class_import():
    """Test that the main class can be imported and instantiated."""
    from nuclear_shape import nuclear_shape
    assert callable(nuclear_shape)


def test_plotting_import():
    """Test that plotting module can be imported."""
    from nuclear_shape import plotting_results
    assert hasattr(plotting_results, 'sphericity_plot')
    assert hasattr(plotting_results, 'pca_plot')
    assert hasattr(plotting_results, 'render_model')


def test_dependencies():
    """Test that all required dependencies are available."""
    import numpy
    import scipy
    import sklearn
    import tqdm
    import pandas
    import seaborn
    import matplotlib
    import jinja2
    import trimesh
    import cvxpy
    
    # If we get here, all imports succeeded
    assert True

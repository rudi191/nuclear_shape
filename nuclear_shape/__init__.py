"""
Nuclear Shape Analysis Package

A Python package for analyzing nuclear shape from 3D-genome models 
generated from Hi-C data.
"""

from .geometry import NuclearShape
from . import visualize

__version__ = "0.1.0"
__all__ = ["NuclearShape", "visualize"]

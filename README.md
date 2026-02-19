nuclear_shape

Tools for analysing nuclear shape from 3D genome models generated from Hi-C data.

The package provides:

Ellipsoid fitting

PCA-based shape characterization

Sphericity metrics

2D and 3D visualization tools

Installation
🔧 Local Installation (Development Mode)
pip install -e .

📦 Standard Installation (after PyPI release)
pip install nuclear_shape

Quick Start
from nuclear_shape import NuclearShape

# Load a Chrom3D .cmm file
shape = NuclearShape("path/to/file.cmm")

# Run analyses
shape.ellipsoid_fit()
shape.ellipsoid_inner()
shape.ellipsoid_outer()
shape.compute_pca()

# Print metrics
shape.print_metrics()

# Plot results
shape.plot("sphericity", show=True)
shape.plot("pca", show=True)

# Render 3D models
shape.render("ellipsoid", show=True)
shape.render("point_cloud", show=True)

Features
🧮 Ellipsoid Fitting

Algebraic ellipsoid fit

Maximum-volume inscribed ellipsoid

Minimum-volume enclosing ellipsoid

📊 Principal Component Analysis

PCA-based ellipsoid

Orientation metrics

Anisotropy measures

📐 Shape Metrics

Wadell sphericity

Volume and surface area estimates

Axis ratios

Elongation metrics

🎨 Visualization
2D Plots

Sphericity

PCA projections

3D Rendering

Point cloud

Fitted ellipsoid

Inner/outer ellipsoids

PCA ellipsoid

Input Format

The package expects Chrom3D .cmm files,
an XML-based format containing 3D coordinates of nuclear markers.

Output

The analysis produces:

Sphericity metrics

Ellipsoid parameters

PCA axes and explained variance

2D plots

3D renderings

Optional .obj exports

Example Data

A small example .cmm file is included in:

test/example_real_data_0.cmm


Run the full pipeline:

python test/test.py

Dependencies

Core dependencies (installed automatically):

numpy

scipy

pandas

matplotlib

seaborn

trimesh

License

MIT License
Copyright (c) 2026

Citation

If you use this package in your research, please cite it appropriately.

nuclear_shape
Tools for analysing nuclear shape from 3D‑genome models generated from Hi‑C data.

Installation
Local installation (recommended during development)
bash
pip install -e .
Standard installation (after PyPI release)
bash
pip install nuclear_shape
Quick Start
python
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
Ellipsoid Fitting
Standard algebraic ellipsoid fit

Maximum‑volume inscribed ellipsoid

Minimum‑volume enclosing ellipsoid

Principal Component Analysis
PCA‑based nuclear orientation

PCA ellipsoid visualization

Shape Metrics
Wadell sphericity

Volume and surface area estimates

Axis ratios and elongation

Visualization
2D plots (sphericity, PCA projection)

3D rendering of:

point cloud

fitted ellipsoid

inner/outer ellipsoids

PCA ellipsoid

Input Format
The package expects Chrom3D .cmm files containing 3D coordinates of nuclear markers in XML format.

Output
The analysis produces:

Sphericity metrics

Ellipsoid parameters

PCA axes and explained variance

2D plots

3D renderings

Optional OBJ exports

Example Data
A small example .cmm file is included in:

Code
test/example_real_data_0.cmm
You can run the full pipeline using:

bash
python test/test.py
Dependencies
Core dependencies (automatically installed):

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
If you use this package in your research, please cite appropriately.

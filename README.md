# nuclear_shape

Python package for analysing nuclear shape from 3D-genome models generated from Hi-C data.

## Installation

### From source (local installation)

```bash
git clone <repository-url>
cd nuclear_shape
pip install .
```

### Development installation (editable mode)

```bash
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install nuclear_shape
```

### Verify Installation

After installation, verify everything works (tests use synthetic data; no repo or data files required):

```bash
# Using the verification script
python -m nuclear_shape.verify_installation

# Or if installed:
verify-installation

# Or run pytest tests
pytest tests/
```

## Features

- **Ellipsoid fitting**: Multiple ellipsoid fitting methods including:
  - Standard algebraic ellipsoid fit
  - Maximum-volume inscribed ellipsoid
  - Minimum-volume enclosing ellipsoid
- **Principal Component Analysis**: PCA-based shape characterization
- **Sphericity metrics**: Multiple sphericity calculations including Wadell sphericity
- **Visualization**: Comprehensive plotting and 3D rendering capabilities
- **Batch processing**: Process multiple files and generate statistics

## Usage

### Basic Usage

```python
from nuclear_shape import nuclear_shape

# Load and analyze a .cmm file
shape = nuclear_shape("path/to/file.cmm")

# Compute various shape metrics
shape.ellipsoid_fit()
shape.ellipsoid_inner()
shape.ellipsoid_outer()
shape.principal_components()

# Access results
print(shape.results["ellipsoid"]["sphericity"])
```

### Library usage (recommended)

This project is intended to be used as a Python library (import and call the class + plotting helpers).
If you want a CLI later, it’s easy to add back as a thin wrapper around the library API.

## Input Format

The package expects `.cmm` files (Chrom3D marker format) containing 3D coordinates of nuclear markers in XML format.

## Output

The analysis produces:
- Sphericity metrics for different shape models
- Volume and surface area measurements
- Aspect ratios and shape classification (oblate/prolate)
- PCA-based anisotropy and elongation metrics
- Visualizations (plots and 3D renderings)
- Optional LaTeX tables for publication

## Dependencies

- numpy
- scipy
- scikit-learn
- tqdm
- pandas
- seaborn
- matplotlib
- Jinja2
- trimesh
- cvxpy

## License

MIT License

Copyright (c) 2026 rudi191

## Citation

If you use this package in your research, please cite appropriately.

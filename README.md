# nuclear_shape

A small Python library for analysing nuclear shape from 3D‑genome models (Chrom3D `.cmm` files).  
Includes ellipsoid fitting, PCA, basic shape metrics, and simple plotting/rendering.

## Installation

```bash
pip install -e .
```bash

Basic Usage

python
from nuclear_shape import NuclearShape

shape = NuclearShape("path/to/file.cmm")

shape.ellipsoid_fit()
shape.ellipsoid_inner()
shape.ellipsoid_outer()
shape.compute_pca()

shape.print_metrics()
shape.plot("sphericity", show=True)
shape.render("ellipsoid", show=True)



Example Data
An example .cmm file and a test script are included:

Code

test/example_real_data_0.cmm
python test/test.py


License
MIT License

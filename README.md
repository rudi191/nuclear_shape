# nuclear_shape

A small Python library for analysing nuclear shape from 3D‑genome models (Chrom3D `.cmm` files).  
Includes ellipsoid fitting, PCA, basic shape metrics, and simple plotting/rendering.

## Installation

```bash
pip install nuclear_shape
```


## Basic Usage

```python
from nuclear_shape import NuclearShape

shape = NuclearShape("path/to/file.cmm")

shape.ellipsoid_fit()
shape.ellipsoid_inner()
shape.ellipsoid_outer()
shape.compute_pca()

shape.print_metrics()

shape.plot("sphericity", show=True)
shape.render("ellipsoid", show=True)

```
## Plot and Render 
`shape.plot(kind, save=False, show=True, path='path/to/save/file')`
#### Parameters
**kind** : *str*  
The type of metric layout to plot (`'sphericity', pca, point_cloud`).

**save** : *bool, optional*  
If True, saves the plot to disk. Default is `False`.

**show** : *bool, optional*  
If True, opens the local interactive window to view the graph. Default is `True`.

**path** : *str, optional*  
The destination directory of the saved file. Uses fixed file names depending on the kind: `'sphericity_plot.png, pca_projection.png, point_cloud.png'`.

## Example Data
An example .cmm file and a test script are included:

```code
test/example_real_data_0.cmm
python test/test.py
```

## License
MIT License


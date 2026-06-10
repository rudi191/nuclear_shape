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
## Plot and Render Methods
### `shape.plot(kind, save=False, show=True, path='path/to/save/file')` 
Simple 2D bar plot of sphericity between the ellipsoid methods, pca projection in 2D and a 3D plot of the points of the 3D genome.
#### Parameters
**kind** : *str*  
The type of metric or geometry to plot (`'sphericity', 'pca', 'point_cloud'`).

**save** : *bool, optional*  
If True, saves the plot to disk. Default is `False`.

**show** : *bool, optional*  
If True, opens the local interactive window to view the graph. Default is `True`.

**path** : *str, optional*  
The destination directory of the saved file. Uses fixed file names depending on the kind: `'sphericity_plot.png, pca_projection.png, point_cloud.png'`.

### `shape.render(model="ellipsoid", name="file_name", show=True, save=True, path="path/to/save/files", export_obj=True)` 

#### Parameters
**model** : *str*  
Type of model (`'ellipsoid', 'ellispoid_inner', 'ellipsoid_outer', 'pca', 'point_cloud', `).

**name** : *str, optional*  
Specifies file name for the saved files. Default is `None`, naming the files based on the type of model chosen.

**show** : *bool, optional*  
If True, opens the local interactive window to view the figure. Default is `True`.

**save** : *bool, optional*  
If True, saves the plot to disk. Default is `False`.

**path** : *str, optional*  
The destination directory of the saved files. 

**export_obj** : *bool, optional*  
If True, exports the shapes as object-files that can be used in visualization softwares (such as ChimeraX). Default is `True`.

## Metrics


## Example Data
An example .cmm file and a test script are included:

```code
test/example_real_data_0.cmm
python test/test.py
```

## License
MIT License


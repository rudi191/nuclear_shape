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
### Ellipsoid Metrics
Based on the semi-axes $a \geq b \geq c$ of the fitted ellispoid.

| Metric | Description |
|---|---|
| `sphericity_volume` | $(abc)^{1/3} / a$ — geometric mean of the semi-axes normalised by the longest. Ranges 0–1; 1 = perfect sphere. |
| `sphericity_axes` | $c / a$ — ratio of shortest to longest semi-axis. Ranges 0–1; 1 = perfect sphere. |
| `aspect_ratio_ab` | $b / a$ — ratio intermediate to longest axis. |
| `aspect_ratio_bc` | $c / b$ — ratio shortest to intermediate axis. |
| `aspect_ratio_ac` | $c / a$ — ratio shortest to longest axis`. |
| `flattening` | $(a - c) / a$ — ranges 0–1; 0 = sphere, 1 = max flattening. |
| `eccentricity` | $\sqrt{1 - (c/a)^2}$ for oblate shapes; $\sqrt{1 - (b/a)^2}$ for prolate. Classified by whether $b$ is closer to $a$ (oblate) or $c$ (prolate). |
| `shape_type` | `oblate` if $\|a-b\| < \|b-c\|$, `prolate` otherwise. |

### PCA Metrics
 
Computed from a standard PCA of the point cloud. See the [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).  Here $\lambda_i$ mark the explained variance ratios (summing to 1), ordered $\lambda_1 \geq \lambda_2 \geq \lambda_3$. Two metrics are calculated from these ratios.  
 
| Metric | Description |
|---|---|
| `anisotropy` | $\lambda_1 - \lambda_3$ — ranges 0–1; 0 = uniform |
| `elongation` | $\lambda_1 / \lambda_3$ — ratio of most variance and least variance in the point cloud. 1 = uniform, larger values indicate a more elongated or flattened shape. |

## Example Data
An example .cmm file and a test script are included:

```code
test/example_real_data_0.cmm
python test/test.py
```

## License
MIT License


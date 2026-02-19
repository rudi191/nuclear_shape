
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import trimesh


# =========================================================
# Internal helper: build ellipsoid mesh
# =========================================================

def _ellipsoid_mesh(center, radii, rotation, nu=60, nv=40):
    """
    Generate a trimesh.Trimesh ellipsoid mesh.
    """
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)

    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    xyz = np.stack([x, y, z], axis=-1)
    xyz_rot = xyz @ rotation.T + center

    verts = xyz_rot.reshape(-1, 3)
    faces = []

    for i in range(nu - 1):
        for j in range(nv - 1):
            v0 = i * nv + j
            v1 = (i + 1) * nv + j
            v2 = (i + 1) * nv + (j + 1)
            v3 = i * nv + (j + 1)
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# =========================================================
# 2D PLOTS
# =========================================================

def plot_sphericity(shape_obj, save=False, show=True, path=None):
    """
    Plot sphericity values for ellipsoid, inner ellipsoid, and outer ellipsoid.
    """
    ell = shape_obj.results["ellipsoid"]["sphericity"]["sphericity_volume"]
    inner_ell = shape_obj.results["ellipsoid_inner"]["sphericity"]["sphericity_volume"]
    outer_ell = shape_obj.results["ellipsoid_outer"]["sphericity"]["sphericity_volume"]

    df = pd.DataFrame({
        "Method": ["Ellipsoid", "Inner ellipsoid", "Outer ellipsoid"],
        "Sphericity": [ell, inner_ell, outer_ell]
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="Method", y="Sphericity", palette="viridis")
    ax.set_ylim(0, 1)
    ax.set_title("Sphericity Comparison")

    if save and path is not None:
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(path) / "sphericity_plot.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_pca(shape_obj, save=False, show=True, path=None):
    """
    Plot PCA projection (PC1 vs PC2) of the nuclear point cloud.
    """
    pca_points = shape_obj.results["PCA"]["transformed_points"]
    df = pd.DataFrame(pca_points, columns=["PC1", "PC2", "PC3"])

    sns.set(style="white", context="talk")
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="PC1", y="PC2", s=12, edgecolor=None)

    plt.title("PCA Projection (PC1 vs PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if save and path is not None:
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(path) / "pca_projection.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_point_cloud(shape_obj, save=False, show=True, path=None):
    """
    Plot the raw 3D point cloud of the nucleus.
    """
    points = shape_obj.matrix

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=4, alpha=0.7, c=points[:, 2], cmap="viridis")

    ax.set_title("Point Cloud")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if save and path is not None:
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(path) / "point_cloud.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


# =========================================================
# 3D RENDERING
# =========================================================

def render_model(shape_obj, model="ellipsoid", save=False, show=True, path=None, export_obj=True):
    """
    Render a 3D model (ellipsoid, inner, outer, PCA, or point cloud).
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    mesh = None

    # -------------------------
    # Ellipsoid models
    # -------------------------
    if model in ["ellipsoid", "ellipsoid_inner", "ellipsoid_outer"]:
        ell = shape_obj.results[model]
        mesh = _ellipsoid_mesh(ell["center"], ell["radii"], ell["rotation"])

        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                        triangles=mesh.faces, color="cornflowerblue", alpha=0.6)

    # -------------------------
    # PCA ellipsoid
    # -------------------------
    elif model == "pca":
        pca = shape_obj.results["PCA"]
        center = np.mean(shape_obj.matrix, axis=0)
        radii = 3 * np.sqrt(pca["variance"])
        rotation = pca["components"].T

        mesh = _ellipsoid_mesh(center, radii, rotation)

        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                        triangles=mesh.faces, color="cornflowerblue", alpha=0.6)

    # -------------------------
    # Point cloud
    # -------------------------
    elif model == "point_cloud":
        pts = shape_obj.matrix
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, alpha=0.7, color="steelblue")

    else:
        raise ValueError(f"Unknown model: {model}")

    # Formatting
    ax.set_title(f"3D Model: {model}")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Save figure
    if save and path is not None:
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(path) / f"{model}.png", dpi=300, bbox_inches="tight")

    # OBJ export
    if export_obj and mesh is not None and path is not None:
        Path(path).mkdir(parents=True, exist_ok=True)
        mesh.export(Path(path) / f"{model}.obj")

    if show:
        plt.show()

    plt.close()

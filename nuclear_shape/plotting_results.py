# For plotting and presenting results:

# Libraries
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import trimesh

#Tables

# exporting results to latex table:

def export_full_table(df, caption, label, filename="table.tex"):
    table = df.to_latex(
        index=False,
        float_format="%.3f",
        escape=False
    )

    table = table.replace("\\begin{tabular}", "\\begin{tabular}")
    table = table.replace("\\toprule", "\\hline")
    table = table.replace("\\midrule", "\\hline")
    table = table.replace("\\bottomrule", "\\hline")

    # Wrap in full LaTeX table environment
    wrapped = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        f"{table}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )

    with open(filename, "w") as f:
        f.write(wrapped)

    return wrapped


def export_sphericity_table(shape_obj, filename="sphericity.tex"):
    ell = shape_obj.results["ellipsoid"]["sphericity"]["sphericity_volume"]
    inner_ell = shape_obj.results["ellipsoid_inner"]["sphericity"]["sphericity_volume"]
    outer_ell = shape_obj.results["ellipsoid_outer"]["sphericity"]["sphericity_volume"]

    df = pd.DataFrame({
        "Method": ["Ellipsoid", "inner_ellipsoid", "outer_ellipsoid"],
        "Sphericity": [ell, inner_ell, outer_ell]
    })

    return export_full_table(
        df,
        caption="Sphericity values for different geometric models.",
        label="tab:sphericity",
        filename=filename
    )


#Figures


def sphericity_plot(shape_obj, save=False, show=True, path="../figures/"):

#bar plot with sphericity metrics for all methods
    # Extract values
    ell = shape_obj.results["ellipsoid"]["sphericity"]["sphericity_volume"]
    inner_ell = shape_obj.results["ellipsoid_inner"]["sphericity"]["sphericity_volume"]
    outer_ell = shape_obj.results["ellipsoid_outer"]["sphericity"]["sphericity_volume"]

    df = pd.DataFrame({
        "Method": ["Ellipsoid", "inner_ellipsoid", "outer_ellipsoid"],
        "Sphericity": [ell, inner_ell, outer_ell]
    })

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="Method", y="Sphericity", palette="viridis")
    ax.set_ylim(0, 1)
    ax.set_title("Sphericity Comparison")

    # Save or show
    if save:
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(path) / "sphericity_plot.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()






def pca_plot(shape_obj, save=False, show=True, path="../figures/"):

    # Extract PCA-transformed coordinates
    pca_points = shape_obj.results["PCA"]["transformed_points"]
    df = pd.DataFrame(pca_points, columns=["PC1", "PC2", "PC3"])

    # Plot
    sns.set(style="white", context="talk")
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="PC1", y="PC2", s=12, edgecolor=None)

    plt.title("PCA Projection (PC1 vs PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Save or show
    if save:
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(path) / "pca_projection.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()




def render_model_matplotlib(
    shape_obj,
    model,
    save=False,
    show=True,
    path="../mesh/",
    export_obj=True
):

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    mesh = None  # will hold a trimesh.Trimesh if exportable

    # ---------------------------------------------------------
    # 1. Ellipsoid
    # ---------------------------------------------------------
    if model == "ellipsoid":
        ell = shape_obj.results["ellipsoid"]

        center = ell["center"]
        radii = ell["radii"]
        rotation = ell["rotation"]

        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1)
        xyz_rot = xyz @ rotation.T + center

        ax.plot_surface(
            xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2],
            rstride=1, cstride=1, color="cornflowerblue",
            alpha=0.6, linewidth=0
        )

        # Build trimesh for export
        verts = xyz_rot.reshape(-1, 3)
        nu, nv = x.shape
        faces = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                v0 = i * nv + j
                v1 = (i + 1) * nv + j
                v2 = (i + 1) * nv + (j + 1)
                v3 = i * nv + (j + 1)
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # ---------------------------------------------------------
    # inner_ellipsoid 
    # ---------------------------------------------------------
    elif model == "ellipsoid_inner":
        ell = shape_obj.results["ellipsoid_inner"]

        center = ell["center"]
        radii = ell["radii"]
        rotation = ell["rotation"]

        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1)
        xyz_rot = xyz @ rotation.T + center

        ax.plot_surface(
            xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2],
            rstride=1, cstride=1, color="cornflowerblue",
            alpha=0.6, linewidth=0
        )

        # Build trimesh for export
        verts = xyz_rot.reshape(-1, 3)
        nu, nv = x.shape
        faces = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                v0 = i * nv + j
                v1 = (i + 1) * nv + j
                v2 = (i + 1) * nv + (j + 1)
                v3 = i * nv + (j + 1)
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)


    # ---------------------------------------------------------
    # outer_ellipsoid 
    # ---------------------------------------------------------
    elif model == "ellipsoid_outer":
        ell = shape_obj.results["ellipsoid_outer"]

        center = ell["center"]
        radii = ell["radii"]
        rotation = ell["rotation"]

        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1)
        xyz_rot = xyz @ rotation.T + center

        ax.plot_surface(
            xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2],
            rstride=1, cstride=1, color="cornflowerblue",
            alpha=0.6, linewidth=0
        )

        # Build trimesh for export
        verts = xyz_rot.reshape(-1, 3)
        nu, nv = x.shape
        faces = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                v0 = i * nv + j
                v1 = (i + 1) * nv + j
                v2 = (i + 1) * nv + (j + 1)
                v3 = i * nv + (j + 1)
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # ---------------------------------------------------------
    # PCA Ellipsoid
    # ---------------------------------------------------------
    elif model == "pca":
        pca_res = shape_obj.results["PCA"]
        
        # Center is the mean of the original points
        center = np.mean(shape_obj.matrix, axis=0)
        
        # Radii: use a multiplier of the standard deviations
        # 2.0 captures ~95% of points, 2.5-3.0 for ~99%
        std_multiplier = 3.0  # adjust this based on your needs
        radii = std_multiplier * np.sqrt(pca_res["variance"])
        
        # Rotation matrix is the principal components (eigenvectors)
        rotation = pca_res["components"].T  # transpose to get column vectors
        
        # Generate ellipsoid mesh
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        
        # Stack and rotate
        xyz = np.stack([x, y, z], axis=-1)
        xyz_rot = xyz @ rotation.T + center
        
        # Plot surface
        ax.plot_surface(
            xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2],
            rstride=1, cstride=1, color="cornflowerblue",
            alpha=0.6, linewidth=0
        )
        
        # Build trimesh for export
        verts = xyz_rot.reshape(-1, 3)
        nu, nv = x.shape
        faces = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                v0 = i * nv + j
                v1 = (i + 1) * nv + j
                v2 = (i + 1) * nv + (j + 1)
                v3 = i * nv + (j + 1)
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # ---------------------------------------------------------
    # Point Cloud (NO OBJ EXPORT)
    # ---------------------------------------------------------
    elif model == "point_cloud":
        pts = shape_obj.matrix
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=5, alpha=0.7, color="steelblue"
        )
        mesh = None  # explicitly no export

    else:
        raise ValueError("Unknown model type.")

    # Formatting
    ax.set_title(f"3D Model: {model}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    # Save figure
    if save:
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(path) / f"{model}_3d.png", dpi=300, bbox_inches="tight")

    # OBJ Export (except point cloud)
    if export_obj and mesh is not None:
        Path(path).mkdir(parents=True, exist_ok=True)
        obj_path = Path(path) / f"{model}.obj"
        mesh.export(obj_path)
        print(f"OBJ exported to: {obj_path}")

    if show:
        plt.show()

    plt.close()


def render_model(shape_obj, model="ellipsoid", save=False, show=True, path="../figures/", export_obj=True):
    """Render 3D model using matplotlib."""
    if model == "all":
        models = ["ellipsoid", "ellipsoid_inner", "ellipsoid_outer", "pca", "point_cloud"]
        for m in models:
            render_model_matplotlib(shape_obj, m, save, show, path, export_obj)
        return
    return render_model_matplotlib(shape_obj, model, save, show, path, export_obj)

def point_cloud(shape_obj, save=False, show=True, path="../figures/"):

    points = shape_obj.matrix
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:,0], points[:,1], points[:,2], s=4, alpha=0.7, c=points[:,2], cmap="viridis")

    ax.set_title("Point Cloud Reference")
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if save:
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(path) / "point_cloud.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()

def plot_batch_statistics(df, save=False, show=True, path="../figures/"):
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        # ---------------------------------------------------------
        # 1. Sphericity distributions
        # ---------------------------------------------------------
        ax[0, 0].hist(df["sphericity_ellipsoid_volume"], bins=20, alpha=0.6, label="Ellipsoid (volume)")
        ax[0, 0].hist(df["sphericity_inner_ellipsoid_volume"], bins=20, alpha=0.6, label="Inner ellipsoid")
        ax[0, 0].hist(df["sphericity_outer_ellipsoid_volume"], bins=20, alpha=0.6, label="Outer ellipsoid")
        ax[0, 0].set_title("Sphericity Distributions")
        ax[0, 0].legend()

        # ---------------------------------------------------------
        # 2. Inner vs outer sphericity
        # ---------------------------------------------------------
        ax[0, 1].scatter(df["sphericity_inner_ellipsoid_volume"], df["sphericity_outer_ellipsoid_volume"], alpha=0.7)
        ax[0, 1].set_title("Inner vs Outer Ellipsoid Sphericity")
        ax[0, 1].set_xlabel("Inner")
        ax[0, 1].set_ylabel("Outer")

        # ---------------------------------------------------------
        # 3. Ellipsoid volume distribution
        # ---------------------------------------------------------
        ax[1, 0].hist(df["ellipsoid_volume"], bins=20, alpha=0.7, color="steelblue", edgecolor="black")
        ax[1, 0].set_title("Ellipsoid Volume")

        # ---------------------------------------------------------
        # 4. PCA variance
        # ---------------------------------------------------------
        ax[1, 1].boxplot(
                [df["pca_var1"], df["pca_var2"], df["pca_var3"]],
                    labels=["PC1", "PC2", "PC3"]
                )
        ax[1, 1].set_title("PCA Variance Explained")

        plt.tight_layout()

        if save:
                Path(path).mkdir(parents=True, exist_ok=True)
                plt.savefig(Path(path) / "batch_statistics.png", dpi=300)

        if show:
            plt.show()

        plt.close()









from tqdm import tqdm
from nuclear_shape import NuclearShape

# ---------------------------------------------------------
# 1. Load file and initialize object
# ---------------------------------------------------------

print("\n=== Loading .cmm file ===")
shape = NuclearShape("test/example_real_data_0.cmm")
print("Loaded matrix shape:", shape.matrix.shape)


# ---------------------------------------------------------
# 2. Run all analyses (with tqdm)
# ---------------------------------------------------------

print("\n=== Running Test ===")

analysis_steps = [
    ("Ellipsoid fit", shape.ellipsoid_fit),
    ("Inner ellipsoid", shape.ellipsoid_inner),
    ("Outer ellipsoid", shape.ellipsoid_outer),
    ("PCA", shape.compute_pca),
]

for label, func in tqdm(analysis_steps, desc="Analysis", unit="step"):
    func()


# ---------------------------------------------------------
# 3. Print metrics
# ---------------------------------------------------------

print("\n=== Printing metrics ===")
shape.print_metrics()


# ---------------------------------------------------------
# 4. Test 2D plots
# ---------------------------------------------------------

print("\n=== Plotting 2D visualizations ===")

plot_steps = [
    ("Sphericity", "sphericity"),
    ("PCA projection", "pca"),
    ("Point cloud", "point_cloud"),
]

for label, key in tqdm(plot_steps, desc="2D Plots", unit="plot"):
    shape.plot(key, show=True)


# ---------------------------------------------------------
# 5. Test 3D renders
# ---------------------------------------------------------

print("\n=== Rendering 3D models ===")

render_steps = [
    "ellipsoid",
    "ellipsoid_inner",
    "ellipsoid_outer",
    "pca",
    "point_cloud",
]

for model in tqdm(render_steps, desc="3D Renders", unit="model"):
    shape.render(model, show=True)


# ---------------------------------------------------------
# 6. Test rendering all models + OBJ export
# ---------------------------------------------------------

print("\n=== Rendering ALL models and exporting OBJ ===")

for model in tqdm(render_steps, desc="Exporting", unit="model"):
    shape.render(model, show=False, save=True, path="test_output", export_obj=True)

print("\n=== Test completed successfully ===")

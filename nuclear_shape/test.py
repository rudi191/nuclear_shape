




from nuclear_shape import NuclearShape

# ---------------------------------------------------------
# 1. Load file and initialize object
# ---------------------------------------------------------

print("\n=== Loading .cmm file ===")
shape = NuclearShape("test/example_real_data_0.cmm")
print("Loaded matrix shape:", shape.matrix.shape)


# ---------------------------------------------------------
# 2. Run all analyses
# ---------------------------------------------------------

print("\n=== Running ellipsoid fit ===")
shape.ellipsoid_fit()

print("\n=== Running inner ellipsoid ===")
shape.ellipsoid_inner()

print("\n=== Running outer ellipsoid ===")
shape.ellipsoid_outer()

print("\n=== Running PCA ===")
shape.compute_pca()


# ---------------------------------------------------------
# 3. Print metrics
# ---------------------------------------------------------

print("\n=== Printing metrics ===")
shape.print_metrics()


# ---------------------------------------------------------
# 4. Test 2D plots
# ---------------------------------------------------------

print("\n=== Plotting sphericity ===")
shape.plot("sphericity", show=True)

print("\n=== Plotting PCA projection ===")
shape.plot("pca", show=True)

print("\n=== Plotting point cloud ===")
shape.plot("point_cloud", show=True)


# ---------------------------------------------------------
# 5. Test 3D renders
# ---------------------------------------------------------

print("\n=== Rendering ellipsoid ===")
shape.render("ellipsoid", show=True)

print("\n=== Rendering inner ellipsoid ===")
shape.render("ellipsoid_inner", show=True)

print("\n=== Rendering outer ellipsoid ===")
shape.render("ellipsoid_outer", show=True)

print("\n=== Rendering PCA ellipsoid ===")
shape.render("pca", show=True)

print("\n=== Rendering point cloud ===")
shape.render("point_cloud", show=True)


# ---------------------------------------------------------
# 6. Test rendering all models + OBJ export
# ---------------------------------------------------------

print("\n=== Rendering ALL models and exporting OBJ ===")
for model in ["ellipsoid", "ellipsoid_inner", "ellipsoid_outer", "pca", "point_cloud"]:
    shape.render(model, show=False, save=True, path="test_output", export_obj=True)

print("\n=== Test completed successfully ===")






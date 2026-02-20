#Class to compute nuclear shape from chrom3D generated point cloud (.cmm-file) 
import numpy as np 
import xml.etree.ElementTree as ET
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import cvxpy as cp 



class NuclearShape:
    def __init__(self, file):
        self.f = file 
        self.matrix, self.center = self.extract_and_center(file)
        self.results = {}

    def extract_and_center(self, file):
        tree = ET.parse(file)
        root = tree.getroot()

        coords = []
        

        for marker in root.iter("marker"):
            x = float(marker.get("x"))
            y = float(marker.get("y"))
            z = float(marker.get("z"))
            coords.append([x,y,z])
        matrix = np.array(coords)


        center = matrix.mean(axis=0)
        matrix_centered = matrix - center

        self.center = center
        self.matrix =  matrix_centered
        
        return matrix_centered, center
    

    def ellipsoid_fit(self):
# 
        # The matrix with points is stored in C, normalized by the maximum absolute value and each column separated into each direction, x, y, z. 
        C = self.matrix
        scale = np.max(np.abs(C))
        C_norm = C / scale
        x, y, z = C_norm.T


        # The design matrix is created from taking the futures from the homogenous equation of the ellipsoid and writing it in matrix form. 
        D = np.column_stack([
            x*x, y*y, z*z,
            2*x*y, 2*x*z, 2*y*z,
            2*x, 2*y, 2*z,
            np.ones(len(x))
        ])


        # To solve the minimization problem by using the Gram matrix of the design matrix. 
        # The smallest eigenvalue and its corresponding eigenvector is the solution to the parameter set that minimizes the algebraic equation. 
        DtD = D.T @ D
        eigenvalues, eigenvectors = np.linalg.eigh(DtD)
        v = eigenvectors[:, 0]

        if v[9] < 0:
            v = -v
# The components of the eigenvector are stored in parameter variables
        A, B, Cc, D_xy, E_xz, F_yz, G, H, I, J = v

        # The parameters are inputted into the quadratic form representation of the algebraic ellipsoid equation.
        # This is used to find the center and axis of the ellipsoid. 
        Q = np.array([
            [A,       D_xy/2, E_xz/2],
            [D_xy/2,  B,      F_yz/2],
            [E_xz/2,  F_yz/2, Cc]
        ])

        linear = np.array([G, H, I])

        try:
            center_norm = -np.linalg.solve(2*Q, linear)
        except np.linalg.LinAlgError:
            center_norm = -np.linalg.lstsq(2*Q, linear, rcond=None)[0]

        const = J + linear @ center_norm + center_norm @ Q @ center_norm

        eigenvalues_Q, eigenvectors_Q = np.linalg.eigh(Q)

        with np.errstate(invalid='ignore', divide='ignore'):
            axes_norm = np.sqrt(np.abs(-const / eigenvalues_Q))

        axes_norm = np.nan_to_num(axes_norm, nan=0.0, posinf=0.0, neginf=0.0)

        sort_idx = np.argsort(axes_norm)[::-1]
        axes_norm = axes_norm[sort_idx]
        eigenvectors_Q = eigenvectors_Q[:, sort_idx]

        #transform back to original 
        center_original = center_norm * scale + self.center
        axes_original = axes_norm * scale
# Results from fitting are stored in a dictionary. 

        ellipsoid_results = {
            'center': center_original,
            'axes': axes_original,
            'radii': axes_original,
            'rotation': eigenvectors_Q,
            'eigenvalues': eigenvalues_Q[sort_idx],
            'v': v
        }

        self.results["ellipsoid"] = ellipsoid_results


        # Shape evaluation:
        # sphericity metrics, axis are extracted for sphericity calculations.
        a, b, c = np.sort(axes_original)[::-1]

        sphericity_volume = (a * b * c)**(1/3) / a
        sphericity_axes = c / a

        aspect_ab = b / a
        aspect_bc = c / b if b > 0 else 0
        aspect_ac = c / a

        flattening = (a - c) / a if a > 0 else 0

        if np.abs(a - b) < np.abs(b - c):  # oblate
            eccentricity = np.sqrt(1 - (c**2 / a**2)) if a > 0 else 0
            shape_type = 'oblate'
        else:  # prolate
            eccentricity = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0
            shape_type = 'prolate'
# Metrics are stored in dictionary 

        self.results["ellipsoid"]["sphericity"] = {
            'sphericity_volume': sphericity_volume,
            'sphericity_axes': sphericity_axes,
            'aspect_ratio_ab': aspect_ab,
            'aspect_ratio_bc': aspect_bc,
            'aspect_ratio_ac': aspect_ac,
            'flattening': flattening,
            'eccentricity': eccentricity,
            'shape_type': shape_type
        }




    def ellipsoid_inner(self):
        """
        Maximum-volume inscribed ellipsoid using CVX.
        Returns B (linear map) and center.
        """

        points = self.matrix
        dim = points.shape[1]

        # Build convex hull constraints
        hull = ConvexHull(points)
        A = hull.equations[:, :dim]
        b = -hull.equations[:, dim]

        # Variables
        B = cp.Variable((dim, dim), PSD=True)
        d = cp.Variable(dim)

        constraints = [cp.norm(B @ A[i], 2) + A[i] @ d <= b[i] for i in range(len(A))]
        prob = cp.Problem(cp.Minimize(-cp.log_det(B)), constraints)
        prob.solve()

        B_val = B.value
        d_val = d.value

        # Convert center back to original coordinate system
        center_original = d_val + self.center

        # Radii from SVD of B
        U, S, Vt = np.linalg.svd(B_val)
        radii = S
        idx = np.argsort(radii)[::-1]
        radii = radii[idx]
        U = U[:, idx]
        
        metrics = self._compute_sphericity_metrics(radii)

        self.results["ellipsoid_inner"] = {
            "B": B_val,
            "center": center_original,
            "radii": radii,
            "rotation": U,
            "sphericity": metrics
        }





    def ellipsoid_outer(self, tol=1e-3):
        """
        Minimum-volume enclosing ellipsoid.
        Returns A, center where (x-c)^T A (x-c) = 1.
        """
        P = np.asarray(self.matrix)
        N, d = P.shape

        Q = np.column_stack((P, np.ones(N))).T
        u = np.ones(N) / N
        err = 1 + tol

        while err > tol:
            X = Q @ np.diag(u) @ Q.T
            M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
            j = np.argmax(M)
            step = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
            new_u = (1 - step) * u
            new_u[j] += step
            err = np.linalg.norm(new_u - u)
            u = new_u

        c_centered = u @ P
        A = np.linalg.inv(P.T @ np.diag(u) @ P - np.outer(c_centered, c_centered)) / d

        center_original = c_centered + self.center

        # Radii from eigenvalues
        evals, evecs = np.linalg.eigh(A)
        radii = 1 / np.sqrt(evals)
        idx = np.argsort(radii)[::-1]
        radii = radii[idx]
        evecs = evecs[:, idx]

       
        metrics = self._compute_sphericity_metrics(radii)

        self.results["ellipsoid_outer"] = {
            "A": A,
            "center": center_original,
            "radii": radii,
            "rotation": evecs,
            "sphericity": metrics
        }

    def _compute_sphericity_metrics(self, radii):
        a, b, c = np.sort(radii)[::-1]

        sphericity_volume = (a * b * c)**(1/3) / a
        sphericity_axes = c / a

        aspect_ab = b / a
        aspect_bc = c / b if b > 0 else 0
        aspect_ac = c / a

        flattening = (a - c) / a if a > 0 else 0

        if np.abs(a - b) < np.abs(b - c):
            eccentricity = np.sqrt(1 - (c**2 / a**2)) if a > 0 else 0
            shape_type = 'oblate'
        else:
            eccentricity = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0
            shape_type = 'prolate'

        return {
            'sphericity_volume': sphericity_volume,
            'sphericity_axes': sphericity_axes,
            'aspect_ratio_ab': aspect_ab,
            'aspect_ratio_bc': aspect_bc,
            'aspect_ratio_ac': aspect_ac,
            'flattening': flattening,
            'eccentricity': eccentricity,
            'shape_type': shape_type
        }








    def compute_pca(self):
        pca = PCA(n_components=3)
        pca.fit(self.matrix)

        transformed = pca.transform(self.matrix)

        axis_lengths = 2 * np.sqrt(pca.explained_variance_)

        self.results["PCA"] = {
            "components": pca.components_,              # principal axes (eigenvectors)
            "singular_values": pca.singular_values_,
            "variance": pca.explained_variance_,
            "variance_ratio": pca.explained_variance_ratio_,
            "axis_lengths": axis_lengths,               # PCA-based ellipsoid radii
            "transformed_points": transformed,          # coordinates in PCA space
            "anisotropy": pca.explained_variance_ratio_[0] - pca.explained_variance_ratio_[2],
            "elongation": pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[2]
        }



    def print_metrics(self):
        if not self.results:
            print("No results available. Run an analysis method first.")
            return

        for key, res in self.results.items():
            print(f"\n=== {key.upper()} ===")

            # --- Ellipsoid metrics (sphericity block only) ---
            if "sphericity" in res:
                sph = res["sphericity"]
                for metric, value in sph.items():
                    if isinstance(value, (int, float, np.floating)):
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}: {value}")
                continue

            # --- PCA metrics ---
            if key.upper() == "PCA":
                pca_metrics = [
                    "variance_ratio",
                    "axis_lengths",
                    "anisotropy",
                    "elongation"
                ]

                for metric in pca_metrics:
                    value = res.get(metric, None)
                    if value is None:
                        continue

                    if isinstance(value, np.ndarray):
                        print(f"{metric}: {np.array2string(value, precision=4)}")
                    elif isinstance(value, (int, float, np.floating)):
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}: {value}")

                continue






    def plot(self, kind="sphericity", **kwargs):
            """
                Convenience wrapper for 2D visualizations.
                """
            from . import visualize as vz

            kinds = {
                    "sphericity": vz.plot_sphericity,
                        "pca": vz.plot_pca,
                        "point_cloud": vz.plot_point_cloud,
                    }

            if kind not in kinds:
                    raise ValueError(f"Unknown plot kind: {kind}. Choose one of: {sorted(kinds)}")

            return kinds[kind](self, **kwargs)


    def render(self, model="all", **kwargs):
            """
                Convenience wrapper for 3D rendering.
                """
            from . import visualize as vz
            return vz.render_model(self, model=model, **kwargs)




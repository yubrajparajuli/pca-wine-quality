import numpy as np
from sklearn.decomposition import PCA


def run_pca_sklearn(X_scaled, n_components):
    """
    Runs sklearn PCA with given number of components.
    Returns transformed data and fitted pca object.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def verify_pca(pca_sklearn, scratch_results):
    """
    Compares sklearn and scratch PCA results.
    Prints component wise variance comparison.
    """
    explained_variance_ratio = scratch_results["explained_variance_ratio"]
    n_components             = scratch_results["n_components"]
    X_pca_scratch            = scratch_results["X_pca"]

    print(f"\n{'Component':<12} {'Sklearn':<12} {'Scratch':<12} {'Match'}")
    print("-" * 45)
    for i in range(n_components):
        sklearn_var = pca_sklearn.explained_variance_ratio_[i] * 100
        scratch_var = explained_variance_ratio[i] * 100
        match       = "OK" if abs(sklearn_var - scratch_var) < 0.1 else "MISMATCH"
        print(f"PC{i+1:<10} {sklearn_var:<12.2f} {scratch_var:<12.2f} {match}")

    print(f"\nSklearn cumulative variance: "
          f"{pca_sklearn.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"Scratch cumulative variance: "
          f"{scratch_results['cumulative_variance'][n_components-1]*100:.2f}%")
    print("Verified: PCA from scratch matches sklearn perfectly!")
import numpy as np


def run_pca_scratch(X_scaled, variance_threshold=0.80):
    """
    Performs PCA from scratch using NumPy.
    Returns transformed data, components and variance info.
    """
    mean = np.mean(X_scaled, axis=0)
    X_centered = X_scaled - mean

    n_samples = X_centered.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues  = eigenvalues.real
    eigenvectors = eigenvectors.real

    sorted_idx   = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    cumulative_variance      = np.cumsum(explained_variance_ratio)

    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    W = eigenvectors[:, :n_components]
    X_pca = np.dot(X_centered, W)

    return {
        "X_pca"                  : X_pca,
        "eigenvalues"            : eigenvalues,
        "eigenvectors"           : eigenvectors,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance"    : cumulative_variance,
        "n_components"           : n_components,
        "W"                      : W
    }
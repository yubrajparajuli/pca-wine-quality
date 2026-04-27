import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize(X):
    """
    Standardizes features to mean=0 and std=1.
    Returns scaled array and fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
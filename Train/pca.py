from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np

def train_pca(data, n_components=2):
    """
    Train PCA for dimensionality reduction and compute reconstruction errors.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    reconstructed_data = pca.inverse_transform(reduced_data)
    
    # Reconstruction error as anomaly score
    reconstruction_error = np.mean((data - reconstructed_data) ** 2, axis=1)
    return pca, reconstruction_error

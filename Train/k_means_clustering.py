from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances

def train_kmeans(data, n_clusters=5):
    """
    Train K-Means clustering and compute anomaly scores based on distances.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    
    # Compute anomaly scores as distance to nearest cluster centroid
    distances = euclidean_distances(data, kmeans.cluster_centers_)
    anomaly_scores = distances.min(axis=1)  # Minimum distance to centroids
    
    return kmeans, anomaly_scores

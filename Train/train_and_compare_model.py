
from Train.isolation_forest import train_isolation_forest
from Train.lof import train_lof
from Train.pca import train_pca
from Train.k_means_clustering import train_kmeans
from Train.model_evaluation import evaluate_unsupervised_model


def train_and_compare_models(data, contamination=0.01, n_clusters=5):
    """
    Train Isolation Forest, LOF, PCA, K-Means, and Spectral Clustering, then compare scores.
    """
    # Train Isolation Forest
    iso_model, iso_scores = train_isolation_forest(data)
    iso_score_mean = evaluate_unsupervised_model(iso_scores, "Isolation Forest")
    
    # Train LOF
    lof_model, lof_scores = train_lof(data)
    lof_score_mean = evaluate_unsupervised_model(lof_scores, "Local Outlier Factor")
    
    # Train PCA
    pca_model, pca_scores = train_pca(data, n_components=2)
    pca_score_mean = evaluate_unsupervised_model(pca_scores, "PCA")
    
    # Train K-Means
    kmeans_model, kmeans_scores = train_kmeans(data, n_clusters)
    kmeans_score_mean = evaluate_unsupervised_model(kmeans_scores, "K-Means")
    
    # Train Spectral Clustering
    # spectral_model, spectral_scores = train_spectral_clustering(data, n_clusters)
    # spectral_score_mean = evaluate_unsupervised_model(spectral_scores, "Spectral Clustering")
    
    # Compare models based on mean anomaly score
    scores = {
        "Isolation Forest": iso_score_mean,
        "LOF": lof_score_mean,
        "PCA": pca_score_mean,
        "K-Means": kmeans_score_mean,
    }
    best_model_name = max(scores, key=scores.get)
    print(f"Best Model: {best_model_name}")
    
    # Return the best model and its scores
    if best_model_name == "Isolation Forest":
        return iso_model, iso_scores
    elif best_model_name == "LOF":
        return lof_model, lof_scores
    elif best_model_name == "PCA":
        return pca_model, pca_scores
    elif best_model_name == "K-Means":
        return kmeans_model, kmeans_scores
    else:
        return spectral_model, spectral_scores

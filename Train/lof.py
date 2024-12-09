from sklearn.neighbors import LocalOutlierFactor

from sklearn.neighbors import LocalOutlierFactor

def train_lof(data, contamination=0.01):
    """
    Train Local Outlier Factor and generate anomaly scores.
    """
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    scores = -model.fit_predict(data)  # LOF does not provide a predict method; use scores
    return model, scores

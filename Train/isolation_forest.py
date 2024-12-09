from sklearn.ensemble import IsolationForest

def train_isolation_forest(data, contamination=0.01):
    """
    Train Isolation Forest and generate anomaly scores.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(data)
    scores = -model.decision_function(data)  # Higher score indicates anomaly
    return model, scores

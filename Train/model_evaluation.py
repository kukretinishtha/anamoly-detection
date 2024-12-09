import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def evaluate_unsupervised_model(scores, model_name, save_path="visualizations"):
    """
    Evaluate unsupervised model using anomaly scores.
    """
    # Visualize anomaly scores
    plt.figure(figsize=(10, 5))
    sns.histplot(scores, bins=50, kde=True)
    plt.title(f"Anomaly Scores Distribution - {model_name}")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name}_scores.png")
    plt.show()
    
    # Return mean anomaly score for ranking models
    mean_score = np.mean(scores)
    print(f"Mean Anomaly Score for {model_name}: {mean_score:.4f}")
    return mean_score

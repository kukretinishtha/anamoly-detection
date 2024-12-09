import pickle

def save_model(model, file_path="best_model.pkl"):
    """
    Save the best model to disk.
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def load_model(file_path="best_model.pkl"):
    """
    Load the model from disk.
    """
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model

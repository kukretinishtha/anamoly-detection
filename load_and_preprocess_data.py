import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_visualize_data(file_path: str):
    """
    Load dataset, visualize features, and analyze distributions.
    """
    data = pd.read_csv(file_path)
    
    # Dataset overview
    print(f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    print(data.info())
    print(data.describe())
    
    # Distribution of all features
    for col in data.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(data[col], bins=50, kde=True)
        plt.title(f"Feature Distribution: {col}")
        plt.savefig(f"visualizations/{col}_distribution.png")
        # plt.show()

    return data

def preprocess_data(data):
    """Preprocess the data by handling missing values, scaling, and encoding."""
    # Step 1: Handle missing values (using median for numerical columns)
    imputer = SimpleImputer(strategy="median")
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Step 2: Scaling numerical features
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)
    
    return data_scaled

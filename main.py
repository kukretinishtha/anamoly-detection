from load_and_preprocess_data import load_and_visualize_data, preprocess_data
from Train.train_and_compare_model import train_and_compare_models
from save_and_load_model import save_model, load_model
import pandas as pd

# Path to dataset
path_to_dataset = "Dataset/creditcard.csv"
path_to_saved_model = "Model/best_model.pkl"

# Load the file and visualize the dataset and save 
data = load_and_visualize_data(path_to_dataset)

data = pd.read_csv(path_to_dataset)
# Create the dataframe after
scaled_data = preprocess_data(data)

# Train different models and evalute
best_model, score = train_and_compare_models(scaled_data)

# Save the best model
saved_model = save_model(best_model, path_to_saved_model)

# Load the model
model = load_model(path_to_saved_model)

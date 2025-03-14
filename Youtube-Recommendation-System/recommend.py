# recommend.py

import torch
import pandas as pd
from model import ConcatNet, load_model
from data_loader import CollabDataset
from torch.utils.data import DataLoader
import config  # Importing the config.py file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def recommend(model_path, new_data, config, batch_size=2000):
    """
    Generate predictions for new user-item pairs.
    
    Args:
        model_path (str): Path to the trained model.
        new_data (list): List of dictionaries containing user-item pairs.
        config (dict): Configuration dictionary for the model.
        batch_size (int): Batch size for prediction.
    
    Returns:
        list: Predicted ratings for the input user-item pairs.
    """
    # Initialize the model and load the trained weights
    model = ConcatNet(config)
    model.to(device)
    model, optimizer, epoch = load_model(model, None, filename=model_path)
    model.eval()

    # Prepare the new data as a DataFrame, with user-item pairs
    new_df = pd.DataFrame(new_data, columns=['user_id', 'item_id'])
    
    # Convert to tensors
    new_df['user_id'] = new_df['user_id'] - 1  # Ensure 0-based indexing
    new_df['item_id'] = new_df['item_id'] - 1  # Ensure 0-based indexing
    
    # Create dataset without requiring a rating column
    new_dataset = CollabDataset(new_df, rating_col=2)  # rating_col=2 is out of bounds, so it will be ignored
    new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

    # List to store predictions
    predictions = []

    # Make predictions for new data
    with torch.no_grad():
        for u, i, _ in new_dataloader:
            u, i = u.to(device), i.to(device)
            r_pred = model(u, i)
            predictions.extend(r_pred.cpu().numpy())

    return predictions

# Example usage
if __name__ == "__main__":
    # Path to the saved model
    model_path = 'saved_models/best_model.pth'

    # New data (user-item pairs that you want predictions for)
    new_data = [
        {'user_id': 1, 'item_id': 2},
        {'user_id': 1, 'item_id': 3},
        {'user_id': 5, 'item_id': 10}
    ]
    
    # Get predictions
    predictions = recommend(model_path, new_data, config.config)  # Pass config.config
    
    # Print predictions
    print("Predicted Ratings:")
    for pred in predictions:
        print(pred)
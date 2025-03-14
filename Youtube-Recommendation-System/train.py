# main.py
from data_loader import load_data, get_dataloaders
from train import train
import config
import matplotlib.pyplot as plt
import os

# Ensure proper entry point protection for multiprocessing
if __name__ == '__main__':
    # Load data
    data_path = 'ml-100k/'
    id_val = 1
    train_df, valid_df = load_data(data_path, id_val)

    # Get dataloaders
    train_dataloader, valid_dataloader = get_dataloaders(train_df, valid_df)

    # Train the model
    model, train_losses, valid_losses, learning_rates = train(config.config, train_dataloader, valid_dataloader)

    # Optionally, plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')

    # Add labels and legend
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Save the plot to the 'plots' directory
    plt.savefig('plots/loss_plot.png')

    # Show the plot
    plt.show()


#https://data-flair.training/blogs/youtube-video-recommendation-system-ml/
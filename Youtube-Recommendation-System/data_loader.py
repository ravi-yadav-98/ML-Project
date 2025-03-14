# data_loader.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CollabDataset(Dataset):
    def __init__(self, df, user_col=0, item_col=1, rating_col=2):
        """
        Args:
            df (pd.DataFrame): Input DataFrame containing user-item interactions.
            user_col (int): Column index for user IDs.
            item_col (int): Column index for item IDs.
            rating_col (int): Column index for ratings. If None, ratings are not used.
        """
        self.df = df
        self.user_tensor = torch.tensor(self.df.iloc[:, user_col], dtype=torch.long, device=device)
        self.item_tensor = torch.tensor(self.df.iloc[:, item_col], dtype=torch.long, device=device)
        
        # Handle optional rating column
        if rating_col < len(self.df.columns):
            self.target_tensor = torch.tensor(self.df.iloc[:, rating_col], dtype=torch.float32, device=device)
        else:
            self.target_tensor = None  # No ratings provided

    def __getitem__(self, index):
        if self.target_tensor is not None:
            return (self.user_tensor[index], self.item_tensor[index], self.target_tensor[index])
        else:
            return (self.user_tensor[index], self.item_tensor[index], 0)  # Return dummy rating

    def __len__(self):
        return self.user_tensor.shape[0]

def load_data(data_path, id_val):
    """
    Load training and validation data from files.
    """
    train_df = pd.read_csv(f'{data_path}u{id_val}.base', sep='\t', header=None)
    train_df.columns = ['user_id', 'item_id', 'rating', 'ts']
    train_df['user_id'] -= 1
    train_df['item_id'] -= 1

    valid_df = pd.read_csv(f'{data_path}u{id_val}.test', sep='\t', header=None)
    valid_df.columns = ['user_id', 'item_id', 'rating', 'ts']
    valid_df['user_id'] -= 1
    valid_df['item_id'] -= 1

    return train_df, valid_df

def get_dataloaders(train_df, valid_df, batch_size=2000):
    """
    Create DataLoader objects for training and validation datasets.
    """
    train_dataset = CollabDataset(train_df)
    valid_dataset = CollabDataset(valid_df)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, valid_dataloader
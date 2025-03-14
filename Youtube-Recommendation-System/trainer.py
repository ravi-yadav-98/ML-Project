# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
from model import ConcatNet, save_model
import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Train function
def train(config, train_dataloader, valid_dataloader):
    # Initialize model, optimizer, and loss function
    model = ConcatNet(config)
    model.to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=1e-3,
                                                      patience=config['reduce_learning_rate'], min_lr=config['learning_rate'] / 10)

    best_loss = float('inf')
    
    best_weights = None
    no_improvements = 0
    train_losses = []
    valid_losses = []
    learning_rates = []

    for epoch in tqdm(range(config['num_epoch'])):
        model.train()
        train_loss = 0
        for u, i, r in train_dataloader:
            r_pred = model(u, i)
            r = r[:, None]
            loss = criterion(r_pred, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        current_learning_rate = optimizer.param_groups[0]['lr']
        learning_rates.append(current_learning_rate)
        train_loss /= len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        valid_loss = 0
        for u, i, r in valid_dataloader:
            u, i = u.to(device), i.to(device) 
            r_pred = model(u, i)
            r = r[:, None]
            loss = criterion(r_pred, r)
            valid_loss += loss.item()

        valid_loss /= len(valid_dataloader.dataset)
        valid_losses.append(valid_loss)
        print(f"Epoch {epoch+1} Train loss: {train_loss:.4f}; Valid loss: {valid_loss:.4f}; Learning rate: {current_learning_rate:.6f}")

        # Check for early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_weights = deepcopy(model.state_dict())
            no_improvements = 0
            save_model(model, optimizer, epoch)
        else:
            no_improvements += 1

        if no_improvements >= config['early_stopping']:
            print(f"Early stopping after epoch {epoch + 1}")
            break

        scheduler.step(valid_loss)

    return model, train_losses, valid_losses, learning_rates

# config.py
config = {
    'num_users': 943,
    'num_items': 1682,
    'emb_size': 50,
    'emb_dropout': 0.05,
    'fc_layer_sizes': [100, 512, 256],
    'dropout': [0.7, 0.35],
    'out_range': [0.8, 5.2],
    'learning_rate': 1e-2,
    'weight_decay': 5e-1,
    'num_epoch': 10,
    'reduce_learning_rate': 1,
    'early_stopping': 5
}

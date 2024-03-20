import os
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

# Data loader
def make_dataloader(X: torch.tensor, Y: torch.tensor, batch_size, shuffle=False):
    if Y is None:
        dataset = TensorDataset(X, torch.zeros(X.shape[0]))
    else:
        dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_valid_split(X: torch.tensor, Y: torch.tensor, valid_ratio=0.2):
    valid_size = int(X.shape[0] * valid_ratio)
    train_size = X.shape[0] - valid_size
    return (X[:train_size], Y[:train_size], X[train_size:], Y[train_size:])

# load imageset from path
def load_imageset(path, transform, max_cnt=None, shuffle=False):
    path += '/'
    file_list = os.listdir(path)
    if shuffle:
        random.shuffle(file_list)

    images = []
    metadata = {
        'mean': [0, 0, 0],
        'std': [0, 0, 0],
    }

    for idx, file in enumerate(file_list):
        if max_cnt and idx >= max_cnt:
            break
        img = plt.imread(path + file)

        metadata['mean'] += img.mean(axis=(0, 1))
        metadata['std'] += img.std(axis=(0, 1))

        images.append(transform(img))
    
    images = torch.stack(images)

    metadata['mean'] /= len(images)
    metadata['std'] /= len(images)
    return (images, file_list, metadata)


# Save & Load model
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path):
    model = torch.load(path)
    return model
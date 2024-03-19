import pandas as pd

# Tabular preprocessing



# Image preprocessing

import torchvision.transforms as transforms

def default_transformer(size: tuple=(128, 128), mean: list=None, std: list=None):
    e = []
    e.append(transforms.ToTensor())
    e.append(transforms.Resize(size))
    if mean and std:
        e.append(transforms.Normalize(mean, std))
    return transforms.Compose(e)

def augment_transformer(VertialFlip: bool=False, HorizontalFlip: bool=False, RandomRotation: int=0):
    e = []
    if VertialFlip:
        e.append(transforms.RandomVerticalFlip())
    if HorizontalFlip:
        e.append(transforms.RandomHorizontalFlip())
    if RandomRotation:
        e.append(transforms.RandomRotation(RandomRotation))
    return transforms.Compose(e)
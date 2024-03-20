import pandas as pd
import numpy as np

# Tabular preprocessing

# Drop missing values and outliers

def drop_nan_row(df: pd.DataFrame):
    return df.dropna(axis=0)

def drop_nan_col(df: pd.DataFrame):
    return df.dropna(axis=1)

def drop_outliers(df: pd.DataFrame, mode: str='IQR', columns: list=None):
    if columns:
        numericals = columns
    else:
        numericals = df.select_dtypes(exclude=['object']).columns.tolist()
    
    if mode == 'IQR':
        Q1 = df[numericals].quantile(0.25)
        Q3 = df[numericals].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[numericals] < (Q1 - 1.5 * IQR)) | (df[numericals] > (Q3 + 1.5 * IQR))
        return df[~outliers.any(axis=1)]

# Categorical to numerical

def categorical_to_order(df: pd.DataFrame, columns: list):
    for col in columns:
        df[col] = df[col].astype('category').cat.codes
    return df
def categorical_to_onehot(df: pd.DataFrame, columns: list, drop: bool=True):
    return pd.get_dummies(df, columns=columns).astype('int')

# Add new features

def add_power(df: pd.DataFrame, columns: list, power: int=2):
    new_col = col + '^' + str(power)
    for col in columns:
        df[new_col] = df[col].apply(lambda x: x ** power)
    return df
def add_log(df: pd.DataFrame, columns: list, power: int=2):
    new_col = col + '_log' + str(power)
    for col in columns:
        df[new_col] = df[col].apply(lambda x: np.log(x) if x > 0 else 0)
    return df

# Feature scaling

def normalize(df: pd.DataFrame, columns: list=None):
    if not columns:
        columns = df.select_dtypes(exclude=['object']).columns.tolist()
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def minmax_scale(df: pd.DataFrame, columns: list=None):
    if not columns:
        columns = df.select_dtypes(exclude=['object']).columns.tolist()
    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df






# Image preprocessing

import torchvision.transforms as transforms

def default_transformer(size: tuple=(128, 128), mean: list=None, std: list=None):
    e = []
    e.append(transforms.ToPILImage())
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
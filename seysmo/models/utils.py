import os

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from seysmo.models import model_class


def save_model(model, path):
    """
    Save the model to disk.

    Parameters:
    - model (nn.Module): Trained model.
    - path (str): Path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_class, path, *args, **kwargs):
    """
    Load the model from disk.

    Parameters:
    - model_class (nn.Module): Model class.
    - path (str): Path to the saved model.
    - args, kwargs: Arguments for initializing the model.

    Returns:
    - model (nn.Module): Loaded model.
    """
    model_class.load_state_dict(torch.load(path))
    model_class.eval()  # Switch model to evaluation mode
    print(f"Model loaded from {path}")
    return model_class


def count_parameters(model):
    """
    Count the total number of parameters in a model.

    Parameters:
    - model (nn.Module): The model to count parameters of.

    Returns:
    - int: Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class SignalSpeedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def give_data(path, num_of_train=3, fraction=0.1):
    with open(path, 'rb') as f:
        data_coord = pickle.load(f)
    coordinates = list(data_coord.keys())
    sorted_coordinates = sorted(coordinates, key=lambda x: x[1])
    sorted_array = np.array(sorted_coordinates)
    coord_train = sorted_array[::num_of_train]
    X_train = []
    y_train = []
    for key in coord_train:
        if data_coord[tuple(key.tolist())][1].shape == (10,):
            X_train.append(data_coord[tuple(key.tolist())][0])
            y_train.append(data_coord[tuple(key.tolist())][1])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    total_elements = sorted_array.shape[0]
    coord_ind = np.setdiff1d(np.arange(total_elements), np.arange(0, total_elements, num_of_train))
    coord_vt = sorted_array[coord_ind]
    X_vt = []
    y_vt = []
    for key in coord_vt:
        if data_coord[tuple(key.tolist())][1].shape == (10,):
            X_vt.append(data_coord[tuple(key.tolist())][0])
            y_vt.append(data_coord[tuple(key.tolist())][1])
    X_vt = np.array(X_vt)
    y_vt = np.array(y_vt)
    total_elements = X_vt.shape[0]
    num_indices = int(total_elements * fraction)
    random_indices = np.random.choice(total_elements, num_indices, replace=False)
    remaining_indices = np.setdiff1d(np.arange(total_elements), random_indices)
    X_val = X_vt[random_indices]
    y_val = y_vt[random_indices]
    coord_val = coord_vt[random_indices]
    X_test = X_vt[remaining_indices]
    y_test = y_vt[remaining_indices]
    coord_test = coord_vt[remaining_indices]
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    return X_train, y_train, coord_train, X_val, y_val, coord_val, X_test, y_test, coord_test


def get_model(cfg):
    ModelClass = getattr(model_class, cfg.model.name)
    return ModelClass(cfg)

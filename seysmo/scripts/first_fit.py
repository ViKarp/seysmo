import os.path

import pandas as pd
import pickle
import segyio
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from seysmo.models.model_class import LeNet5, LeNet5_2
from seysmo.models.train_model import train_model
from seysmo.models.utils import *
from seysmo.visualization.plotting import plot_map
from seysmo.features.mapping import do_array_for_mapping, compute_y_pred
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import MeanAbsolutePercentageError

import mlflow
import hydra
@hydra.main(config_path="../../config", config_name="config")
def main_line(cfg):
    X_train, y_train, coord_train, X_val, y_val, coord_val, X_test, y_test, coord_test = give_data(
        'C:/Users/vitya/Work/seysmo/data/processed/pickle_supervised/all_data.pkl', 2, 0.5)
    train_dataset = SignalSpeedDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataset = SignalSpeedDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_dataset = SignalSpeedDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    model = get_model(cfg)
    mlflow.set_experiment("/CNN_v0.1")
    loss_fn = nn.MSELoss()
    metric_fn = MeanAbsolutePercentageError().to(cfg.device)
    train_model(cfg, model, train_dataloader, val_dataloader, loss_fn, metric_fn)
    num_batches = len(test_dataloader)
    eval_loss, eval_mape = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            pred = model(X)
            eval_loss += loss_fn(torch.reshape(pred, (-1, 10)), torch.reshape(y, (-1, 10))).item()
            eval_mape += metric_fn(torch.reshape(pred, (-1, 10)), torch.reshape(y, (-1, 10))).item()

    eval_loss /= num_batches
    eval_mape /= num_batches
    print(f"Test loss: {eval_loss:4f}")
    print(f"Test MAPE: {eval_mape:4f}")


mlflow.set_tracking_uri("http://localhost:5000")
main_line()




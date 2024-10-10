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
    # X_train, y_train, coord_train, X_val, y_val, coord_val, X_test, y_test, coord_test = give_data(
    #     'C:/Users/vitya/Work/seysmo/data/processed/pickle_supervised/all_data.pkl', 2, 0.5)
    # train_dataset = SignalSpeedDataset(X_train, y_train)
    # train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    # val_dataset = SignalSpeedDataset(X_val, y_val)
    # val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    # test_dataset = SignalSpeedDataset(X_test, y_test)
    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    inputs = [torch.randn(32, 313, 27).to(cfg.device) for i in range(1000)]
    model = get_model(cfg)
    criterion = get_loss(cfg)
    mlflow.set_experiment("/Wav2vec")
    with mlflow.start_run() as run:
        mlflow.log_params(cfg)
        model = model.to(cfg.device)
        optimizer = getattr(torch.optim, cfg.training.name)(model.parameters(), lr=cfg.training.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.training.patience, factor=0.1)
        with open("model_P_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_P_summary.txt")
        for t in range(cfg.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            model.train()
            i = 0
            for X in inputs:
                pred = model(X)
                loss = criterion(*pred)

                # Backpropagation.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                if i % 1 == 0:
                    print(f"loss: {loss.item():2f}")

            scheduler.step(loss)
        mlflow.pytorch.log_model(model, "model")


mlflow.set_tracking_uri("http://localhost:5000")
main_line()




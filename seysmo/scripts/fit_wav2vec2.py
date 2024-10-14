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
    # dataset1 = BigDataset('../../../../../data/arrays_15/combined_batch_32.pkl')
    dataset2 = '../../../../../data/arrays_45/combined_batch_39.pkl'
    # dataset3 = BigDataset('../../../../../data/arrays_15/combined_batch_34.pkl')
    # loader1 = DataLoader(dataset1, batch_size=cfg.batch_size, shuffle=True)
    with open(dataset2, 'rb') as f:
        data = pickle.load(f)
    loader2 = DataLoader(SignalSpeedDataset(data), batch_size=cfg.batch_size, shuffle=True)
    model = get_model(cfg)
    model = load_model(model, '../../../model.pt')
    criterion = get_loss(cfg)
    mlflow.set_experiment("/Wav2vec")
    print(cfg.device)
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
            # for batch1 in loader1:
            #     optimizer.zero_grad()
            #     pred1 = model(batch1)
            #     loss1 = criterion(*pred1)
            #     loss1.backward()
            #     optimizer.step()

                # Обучаемся на втором файле
            for batch2 in loader2:
                optimizer.zero_grad()
                pred2 = model(batch2.to(cfg.device))
                loss2 = criterion(*pred2)
                loss2.backward()
                optimizer.step()
                i += 1
                if i % 10 == 0:
                    # print(f"loss: {loss1.item():2f}")
                    print(f"loss: {loss2.item():2f}")

            #scheduler.step(loss1)
        mlflow.pytorch.log_model(model, "model")
    torch.save(model.state_dict(), "model.pt")


mlflow.set_tracking_uri("http://localhost:5000")
main_line()




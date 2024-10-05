import torch
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.pytorch
from torch import nn
from torchinfo import summary
from torchmetrics import MeanAbsolutePercentageError

from seysmo.models.utils import *
import hydra


def train(dataloader, model, loss_fn, metrics_fn, optimizer, epoch, device, output_size):
    """Train the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
        epoch: an integer, the current epoch number.
        device: gpu if cuda is available, otherwise cpu.
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        # print(model, y)
        loss = loss_fn(torch.reshape(pred, (-1, output_size)), torch.reshape(y, (-1, output_size)))
        mape = metrics_fn(torch.reshape(pred, (-1, output_size)), torch.reshape(y, (-1, output_size)))

        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            step = batch // 100 * (epoch + 1)
            mlflow.log_metric("loss", f"{loss:2f}", step=step)
            mlflow.log_metric("MAPE", f"{mape:2f}", step=step)
            print(f"loss: {loss:2f} MAPE: {mape:2f} [{current} / {len(dataloader)}]")


def evaluate(dataloader, model, loss_fn, metrics_fn, epoch, device, output_size):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        epoch: an integer, the current epoch number.
        device: gpu if cuda is available, otherwise cpu.
    """
    num_batches = len(dataloader)
    model.eval()
    eval_loss, eval_mape = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            eval_loss += loss_fn(torch.reshape(pred, (-1, output_size)), torch.reshape(y, (-1, output_size))).item()
            eval_mape += metrics_fn(torch.reshape(pred, (-1, output_size)), torch.reshape(y, (-1, output_size))).item()

    eval_loss /= num_batches
    eval_mape /= num_batches
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric("eval_mape", f"{eval_mape:2f}", step=epoch)

    print(f"Eval metrics: \nMAPE: {eval_mape:.2f}, Avg loss: {eval_loss:2f} \n")
    return eval_loss, eval_mape


def train_model(cfg, model, train_dataloader, val_dataloader, loss_fn, metric_fn):
    with mlflow.start_run() as run:
        mlflow.log_params(cfg)
        model = model.to(cfg.device)
        with open("model_P_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_P_summary.txt")
        optimizer = getattr(torch.optim, cfg.training.name)(model.parameters(), lr=cfg.training.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.training.patience, factor=0.1)
        early_stopper = EarlyStopper(patience=cfg.training.patience, min_delta=cfg.training.min_delta)

        for t in range(cfg.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, metric_fn, optimizer, epoch=t, device=cfg.device,
                  output_size=10)
            validation_loss, validation_mape = evaluate(val_dataloader, model, loss_fn, metric_fn, epoch=t,
                                                        device=cfg.device,
                                                        output_size=10)
            if early_stopper.early_stop(validation_loss):
                break
            scheduler.step(validation_loss)
        mlflow.pytorch.log_model(model, "model")

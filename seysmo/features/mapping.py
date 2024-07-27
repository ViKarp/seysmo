import numpy as np
import torch


def do_array_for_mapping(coord_train, coord_val, coord_test, y_train, y_val, y_test):
    train_viz = np.column_stack((coord_train, y_train))
    val_viz = np.column_stack((coord_val, y_val))
    test_viz = np.column_stack((coord_test, y_test))
    true_slice = np.concatenate([train_viz, val_viz, test_viz])
    return true_slice


def compute_y_pred(model, test_dataloader, device):
    y_pred = np.array([])
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).to("cpu").detach().numpy()
            y_pred = np.concatenate([y_pred, pred.reshape((-1))])
    return y_pred

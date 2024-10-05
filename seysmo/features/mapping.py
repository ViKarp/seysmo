import numpy as np
import torch


def do_array_for_mapping(coord_train, coord_val, y_train, y_val, coord_test=None, y_test=None):
    train_viz = np.column_stack((coord_train, y_train))
    val_viz = np.column_stack((coord_val, y_val))
    if coord_test is not None and y_test is not None:
        test_viz = np.column_stack((coord_test, y_test))
        print(train_viz.shape, val_viz.shape, test_viz.shape)
        true_slice = np.concatenate([train_viz, val_viz, test_viz])
    else:
        true_slice = np.concatenate([train_viz, val_viz])
    return true_slice


def compute_y_pred(model, test_dataloader, device):
    y_pred = np.array([])
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).to("cpu").detach().numpy()
            y_pred = np.concatenate([y_pred, pred.reshape((-1))])
    return y_pred

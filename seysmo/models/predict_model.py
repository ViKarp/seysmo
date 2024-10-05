import torch

#TODO refactoring
def predict(model, data, device='cpu'):
    """
    Perform prediction using the model.

    Parameters:
    - model (nn.Module): Trained model.
    - data (np.array): Data for prediction.
    - device (str): device for model

    Returns:
    - torch.Tensor: Model predictions.
    """
    model.eval()
    with torch.no_grad():
        sh1, sh2 = data.shape
        data = torch.tensor(data.to_numpy().reshape((sh1, sh2, 1))).to(device)
        outputs = model(data).cpu().detach().numpy()
    return outputs

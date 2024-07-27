import torch
from torch.utils.data import Dataset


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
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()  # Switch model to evaluation mode
    print(f"Model loaded from {path}")
    return model


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
        # Добавляем размер канала
        self.X = torch.from_numpy(X).unsqueeze(1).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

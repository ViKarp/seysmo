import torch


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


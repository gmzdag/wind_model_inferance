import numpy as np
import torch

def run_inference(
    X: np.ndarray,
    model,
    device,
    batch_size: int = 128
) -> np.ndarray:
    """Runs batch model prediction and returns raw reconstruction errors in chunks.
    
    X shape: (Batch, SequenceLength, FeatureSize)
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            chunk = X[i : i + batch_size]
            x_tensor = torch.tensor(
                chunk,
                dtype=torch.float32,
                device=device
            )
            reconstruction = model(x_tensor)
            # Calculate MSE reconstruction error per sequence
            error = torch.mean((reconstruction - x_tensor) ** 2, dim=(1, 2))
            errors.extend(error.cpu().numpy().tolist())

    return np.array(errors, dtype=np.float32)

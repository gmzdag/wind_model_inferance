import numpy as np
import torch

def run_inference(
    sequence,
    model,
    scaler,
    threshold,
    device,
    sequence_length=20,
    feature_size=426
):
    # GCS'ten alınan scaler ve model ile inference yapar, hatayı threshold ile kıyaslar.
    sequence = np.asarray(sequence, dtype=np.float32)

    expected_shape = (sequence_length, feature_size)
    if sequence.shape != expected_shape:
        raise ValueError(
            f"Geçersiz dizi boyutu: {sequence.shape}, beklenen boyut {expected_shape}"
        )

    # Ölçeklendir
    sequence_scaled = scaler.transform(sequence)

    # PyTorch batch'i oluştur
    batch = torch.tensor(
        sequence_scaled,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        reconstruction = model(batch)
        # MSE hesapla
        error = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))

    reconstruction_error = float(error.item())
    is_anomaly = reconstruction_error > threshold

    return reconstruction_error, is_anomaly

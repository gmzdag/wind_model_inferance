import json
import torch
import joblib
import os
import logging
from app.lstm_autoencoder import LSTMAutoencoder

logger = logging.getLogger(__name__)

def load_inference_artifacts(checkpoint_path, scaler_path, metrics_path, device):
    # Model, scaler ve threshold değerini yükler. Threshold GCS'ten gelen metrics.json'dan okunur.
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint bulunamadı: {checkpoint_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler bulunamadı: {scaler_path}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics (Threshold) dosyası bulunamadı: {metrics_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Config'ten modeli oluştur
    model_config = dict(checkpoint["model_config"])
    model_config.pop("name", None)
    model_config.pop("num_channels", None)


    model = LSTMAutoencoder(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model başarıyla yüklendi.")

    # Scaler yükle
    scaler = joblib.load(scaler_path)
    logger.info("Scaler başarıyla yüklendi.")

    # GCS'ten gelen dinamik threshold değerini oku
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    threshold = float(metrics["threshold"]["value"])
    threshold_method = metrics["threshold"].get("method", "bilinmiyor")
    logger.info(f"Threshold yüklendi: Değer={threshold}, Yöntem={threshold_method}")

    return model, scaler, threshold, threshold_method, model_config

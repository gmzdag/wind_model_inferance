import logging
import time
import os
from collections import deque
import torch
from app.config import (
    MODEL_BUCKET_PATH, LOCAL_ARTIFACT_DIR, POLL_INTERVAL_SECONDS,
    SEQUENCE_LENGTH, FEATURE_SIZE, MODEL_VERSION
)
from app.gcs_downloader import download_artifacts
from app.artifact_loader import load_inference_artifacts
from app.db import get_db_connection
from app.db_reader import read_latest_sequence, result_exists
from app.db_writer import write_anomaly_result
from app.inference import run_inference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Inference servisi başlatılıyor...")
    
    # 1. GCS'ten Model, Scaler ve Threshold Dosyalarını İndir
    try:
        download_artifacts(MODEL_BUCKET_PATH, LOCAL_ARTIFACT_DIR)
    except Exception as e:
        logger.error(f"Artifact'ler indirilirken hata oluştu: {e}")
        # Gerekli dosyalar yoksa servis çalışmamalı, bu yüzden crash ediyoruz
        raise

    # 2. İndirilen Artifact'leri Yükle (Model, Scaler, Threshold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(LOCAL_ARTIFACT_DIR, "best_model.pt")
    scaler_path = os.path.join(LOCAL_ARTIFACT_DIR, "scaler.joblib")
    threshold_path = os.path.join(LOCAL_ARTIFACT_DIR, "threshold.json")
    
    # Dosyalardan threshold ve model ayarları okunuyor
    model, scaler, threshold, threshold_method, model_config = load_inference_artifacts(
        checkpoint_path, scaler_path, threshold_path, device
    )

    # 3. Google Cloud'daki TimescaleDB'ye Bağlan
    conn = get_db_connection()

    # 4. Sonsuz Inference Döngüsü
    recent_errors = deque(maxlen=5)  # 5 adımlık hareketli ortalama için kuyruk
    
    while True:
        try:
            # DB'den güncel feature_vector dizisini al (20 adet)
            latest_time, sequence = read_latest_sequence(conn, limit=SEQUENCE_LENGTH)
            
            # Yeterli veri birikmemişse bekle
            if sequence is None:
                logger.info("Yeni veri için bekleniyor (yeterli sequence yok)...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
                
            # Veri boyutu kontrolü (Örn: 20x426 olmalı)
            if sequence.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                logger.error(f"Geçersiz veri boyutu: {sequence.shape}, beklenen: {(SEQUENCE_LENGTH, FEATURE_SIZE)}. Inference atlanıyor.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Bu zaman damgasına ait veri daha önce işlendi mi?
            if result_exists(conn, latest_time, MODEL_VERSION):
                logger.info("Yeni veri için bekleniyor (mevcut veriler zaten işlendi)...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            logger.info(f"Son sequence okundu. Zaman damgası: {latest_time}")

            # LSTM Autoencoder üzerinden inference işlemini yap
            reconstruction_error, _ = run_inference(
                sequence=sequence,
                model=model,
                scaler=scaler,
                threshold=threshold,
                device=device,
                sequence_length=SEQUENCE_LENGTH,
                feature_size=FEATURE_SIZE
            )
            
            # 5 adımlık hareketli ortalama (rolling mean) düzleştirmesini uygula
            recent_errors.append(reconstruction_error)
            smoothed_error = sum(recent_errors) / len(recent_errors)
            is_anomaly = smoothed_error > threshold
            
            logger.info(
                f"Inference tamamlandı: Ham Hata={reconstruction_error:.6f}, "
                f"Düzleştirilmiş Hata={smoothed_error:.6f}, Eşik={threshold:.6f}, Anomali={is_anomaly}"
            )

            # Sonucu veritabanına yaz 
            write_anomaly_result(
                conn=conn,
                time=latest_time,
                reconstruction_error=smoothed_error,
                threshold=threshold,
                is_anomaly=is_anomaly,
                model_version=MODEL_VERSION
            )

        except Exception as e:
            logger.error(f"Inference döngüsünde bir hata oluştu: {e}")
            # DB bağlantısı kopmuşsa tekrar bağlanmayı dene
            if conn.closed:
                logger.info("Veritabanına tekrar bağlanılmaya çalışılıyor...")
                try:
                    conn = get_db_connection()
                except Exception as reconnect_err:
                    logger.error(f"Yeniden bağlanma başarısız oldu: {reconnect_err}")
            
        # Belirlenen süre kadar bekle
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()

import logging
import time
import os
import argparse
import sys
import numpy as np
import torch
from app.config import (
    MODEL_BUCKET_PATH, LOCAL_ARTIFACT_DIR, POLL_INTERVAL_SECONDS,
    SEQUENCE_LENGTH, SEQUENCE_STRIDE, FEATURE_SIZE, MODEL_VERSION
)
from app.gcs_downloader import download_artifacts
from app.artifact_loader import load_inference_artifacts
from app.db import get_db_connection
from app.db_reader import (
    get_last_processed_time, get_preceding_errors, read_feature_vectors_since
)
from app.db_writer import write_anomaly_results_batch
from app.inference import run_inference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_schema(conn):
    """anomaly_results tablosunun varlığını doğrular ve hypertable oluşturur."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_results (
                time                 TIMESTAMPTZ      NOT NULL,
                reconstruction_error DOUBLE PRECISION NOT NULL CHECK (reconstruction_error >= 0),
                threshold            DOUBLE PRECISION NOT NULL CHECK (threshold > 0),
                is_anomaly           BOOLEAN          NOT NULL,
                model_version        TEXT             DEFAULT 'v2'
            );
        """)
        try:
            cur.execute("SELECT create_hypertable('anomaly_results', 'time', if_not_exists => TRUE);")
        except Exception:
            # TimescaleDB uzantısı yüklü değilse normal tablo olarak kalır
            pass
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ar_time ON anomaly_results (time DESC);")
    conn.commit()
    logger.info("Database schema verified.")

def process_data(conn, model, threshold, device, args) -> str | None:
    # 1. Son işlenen zamanı al
    if args.force_backfill:
        logger.info("Force backfill aktif. Son tahmin zaman damgası yoksayılıyor, tüm veri sıfırdan işlenecek.")
        last_time = None
    else:
        last_time = get_last_processed_time(conn)
        logger.info(f"Son tahmin zaman damgası (last_time): {last_time}")
    
    # 2. Belirtilen zamandan sonrasını oku (çakışmayı koruyacak şekilde)
    rows = read_feature_vectors_since(conn, last_time, sequence_length=SEQUENCE_LENGTH)
    logger.info(f"Okunan feature_vectors satır sayısı: {len(rows)}")
    
    if len(rows) < SEQUENCE_LENGTH:
        logger.warning(f"Yetersiz veri: Okunan satır sayısı ({len(rows)}) < Beklenen sequence uzunluğu ({SEQUENCE_LENGTH}). İşlem atlanıyor.")
        return last_time

    times = [r[0] for r in rows]
    features = np.array([r[1] for r in rows], dtype=np.float32)

    # 3. Stride döngüsü için başlangıç indeksini bul
    if last_time is None:
        start_idx = 0
    else:
        try:
            idx = times.index(last_time)
            # Bir önceki pencerenin bitişi last_time idi. Stride=5 olduğu için,
            # yeni pencere last_time'dan SEQUENCE_STRIDE adım ileride bitmeli.
            # Dolayısıyla yeni pencerenin son indeksi: idx + SEQUENCE_STRIDE
            # Dolayısıyla başlangıç indeksi (end_idx - SEQUENCE_LENGTH + 1)
            start_idx = idx + SEQUENCE_STRIDE - SEQUENCE_LENGTH + 1
        except ValueError:
            # Veritabanı zaman damgası veya timezone uyuşmazlığı durumunda fallback
            start_idx = 0
            while start_idx + SEQUENCE_LENGTH <= len(times):
                if times[start_idx + SEQUENCE_LENGTH - 1] > last_time:
                    break
                start_idx += 1

    seqs = []
    seq_times = []
    end_idx = start_idx + SEQUENCE_LENGTH - 1

    while end_idx < len(times):
        seqs.append(features[end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1])
        seq_times.append(times[end_idx])
        end_idx += SEQUENCE_STRIDE

    if not seqs:
        return last_time

    X = np.stack(seqs)
    
    # Boyut kontrolü
    if X.shape[1] != SEQUENCE_LENGTH or X.shape[2] != FEATURE_SIZE:
        logger.error(f"Geçersiz veri boyutu: {X.shape}, beklenen: (Batch, {SEQUENCE_LENGTH}, {FEATURE_SIZE}). Inference atlanıyor.")
        return last_time

    # 4. LSTM Autoencoder üzerinden inference yap (ölçekleme kaldırıldı)
    raw_errors = run_inference(
        X=X,
        model=model,
        device=device
    )

    # 5. Düzleştirme (smoothing) uygula
    if not args.no_smoothing:
        preceding_count = args.smoothing_window - 1
        if last_time is not None:
            preceding_errors = get_preceding_errors(conn, seq_times[0], preceding_count)
        else:
            preceding_errors = []

        all_errors = preceding_errors + list(raw_errors)
        smoothed_errors = []
        for i in range(len(preceding_errors), len(all_errors)):
            window = all_errors[max(0, i - preceding_count) : i + 1]
            smoothed_errors.append(float(np.mean(window)))
        errors_to_save = smoothed_errors
    else:
        errors_to_save = [float(e) for e in raw_errors]

    # 6. Sonuçları topla ve DB'ye yaz
    results = []
    for t, err in zip(seq_times, errors_to_save):
        is_anomaly = bool(err >= threshold)
        results.append((t, err, threshold, is_anomaly, MODEL_VERSION))
        status = "ANOMALİ" if is_anomaly else "NORMAL"
        logger.info(f"Zaman: {t} | Hata: {err:.6f} | Eşik: {threshold:.6f} | Durum: {status}")

    logger.info(f"Yazılacak anomali tahmini sayısı: {len(results)}")
    write_anomaly_results_batch(conn, results)
    
    new_anomalies = sum(1 for r in results if r[3])
    logger.info(f"Veritabanına yazım başarılı. Tespit edilen anomali: {new_anomalies}")
    
    return seq_times[-1]

def main():
    parser = argparse.ArgumentParser(description="LSTM-AE Bulut Çıkarım Servisi")
    parser.add_argument("--mode", type=str, choices=["stream", "batch"], default="stream",
                        help="Çalışma modu: stream (akış) veya batch (toplu işleme)")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Hareketli ortalama ile düzleştirmeyi devre dışı bırakır")
    parser.add_argument("--smoothing-window", type=int, default=5,
                        help="Düzleştirme penceresi boyutu")
    parser.add_argument("--cpu", action="store_true",
                        help="Modeli CPU üzerinde çalışmaya zorlar")
    parser.add_argument("--force-backfill", action="store_true",
                        help="Son işlenen zamana bakılmaksızın tüm veritabanı verisini sıfırdan işler")
    
    args = parser.parse_args()
    
    logger.info("Inference servisi başlatılıyor...")
    
    # 1. GCS'ten Model, Scaler ve Threshold Dosyalarını İndir
    try:
        download_artifacts(MODEL_BUCKET_PATH, LOCAL_ARTIFACT_DIR)
    except Exception as e:
        logger.error(f"Artifact'ler indirilirken hata oluştu: {e}")
        raise

    # 2. İndirilen Artifact'leri Yükle (Model, Scaler, Threshold)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = os.path.join(LOCAL_ARTIFACT_DIR, "best_model.pt")
    scaler_path = os.path.join(LOCAL_ARTIFACT_DIR, "scaler.joblib")
    threshold_path = os.path.join(LOCAL_ARTIFACT_DIR, "threshold.json")
    
    model, scaler, threshold, threshold_method, model_config = load_inference_artifacts(
        checkpoint_path, scaler_path, threshold_path, device
    )

    # 3. Google Cloud'daki TimescaleDB'ye Bağlan
    conn = get_db_connection()
    
    # Şemayı doğrula
    verify_schema(conn)

    # 4. Tahmin Döngüsü
    if args.mode == "batch":
        logger.info("BATCH modunda çalıştırılıyor. Tüm geçmiş veri işlenecek.")
        try:
            process_data(conn, model, threshold, device, args)
        except Exception as e:
            logger.error(f"Batch işleminde hata oluştu: {e}")
        finally:
            logger.info("Batch işlemi tamamlandı. Çıkış yapılıyor.")
            if conn:
                conn.close()
        return

    logger.info(f"STREAM modunda çalıştırılıyor. Polling aralığı: {POLL_INTERVAL_SECONDS} saniye.")
    try:
        while True:
            try:
                process_data(conn, model, threshold, device, args)
            except Exception as e:
                logger.error(f"Inference döngüsünde bir hata oluştu: {e}")
                # DB bağlantısı kopmuşsa tekrar bağlanmeyi dene
                if conn.closed:
                    logger.info("Veritabanına tekrar bağlanılmaya çalışılıyor...")
                    try:
                        conn = get_db_connection()
                    except Exception as reconnect_err:
                        logger.error(f"Yeniden bağlanma başarısız oldu: {reconnect_err}")
            
            time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        if conn:
            conn.close()
            logger.info("Veritabanı bağlantısı kapatıldı.")

if __name__ == "__main__":
    main()

import os
import logging
from google.cloud import storage

logger = logging.getLogger(__name__)

def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    # GCS yolunu bucket ve prefix olarak ayırır.
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Geçersiz GCS yolu: {gcs_path}. 'gs://' ile başlamalıdır.")
    
    path_without_scheme = gcs_path[5:]
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix

def download_artifacts(gcs_path: str, local_dir: str) -> None:
    # Model ve config dosyalarını GCS'ten indirir. (Bulut tabanlı)
    logger.info("GCS'ten artifact'ler indiriliyor...")
    bucket_name, prefix = parse_gcs_path(gcs_path)
    
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # GCP client'ı başlat (VM içi yetkilendirme)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        logger.error(f"GCS istemcisi başlatılamadı. GOOGLE_APPLICATION_CREDENTIALS ayarlı mı? Hata: {e}")
        raise

    required_files = ["best_model.pt", "scaler.joblib", "threshold.json"]

    for file_name in required_files:
        blob_path = os.path.join(prefix, file_name).replace("\\", "/")
        blob = bucket.blob(blob_path)
        local_file_path = os.path.join(local_dir, file_name)
        
        if blob.exists():
            blob.download_to_filename(local_file_path)
            logger.info(f"{file_name} başarıyla indirildi.")
        else:
            raise FileNotFoundError(f"Gerekli dosya {file_name} GCS'te bulunamadı: {blob_path}")

    logger.info("Tüm artifact'ler GCS'ten başarıyla indirildi.")

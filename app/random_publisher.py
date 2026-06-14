import os
import sys
import time
import random
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Logging kurulumu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("random_publisher")

# wind-turbine-edge-processing projesini sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDGE_ROOT = PROJECT_ROOT / "wind-turbine-edge-processing"
sys.path.insert(0, str(EDGE_ROOT))

try:
    from edge.config.channel_groups import load_groups_from_config
    from edge.config.settings import load_config, validate_config
    from edge.data_ingestion.mat_loader import load_mat_file
    from edge.feature_extraction.vector_assembler import (
        EXPECTED_FEATURE_DIM,
        assemble_feature_vectors,
        prepare_vectors_for_model_db,
        get_feature_step_seconds,
    )
    from edge.main import _infer_scenario_label
except ImportError as e:
    logger.error(f"Edge modülleri yüklenemedi: {e}. Lütfen wind-turbine-edge-processing klasörünün mevcut olduğundan emin olun.")
    sys.exit(1)

from app.db import get_db_connection

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wind Turbine Vektör Gönderim Simülatörü (Doğrudan DB)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(EDGE_ROOT / "edge" / "config" / "config.yaml"),
        help="YAML yapılandırma dosyası yolu",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.5,
        help="İki dosya gönderimi arasındaki gecikme (saniye, varsayılan: 2.5)",
    )
    args = parser.parse_args()

    # Yapılandırmayı yükle ve doğrula
    config = load_config(args.config)
    validate_config(config)

    groups = load_groups_from_config(config)
    data_dir = config.get("data_source", {}).get("mat_directory", str(EDGE_ROOT / "data"))
    file_pattern = config.get("data_source", {}).get("file_pattern", "*.mat")

    # Mat dosyalarını bul
    root_path = Path(data_dir).resolve()
    if not root_path.is_dir():
        logger.error(f"Veri dizini mevcut değil: {root_path}")
        sys.exit(1)

    mat_files = list(root_path.rglob(file_pattern))
    if not mat_files:
        logger.error(f"'{file_pattern}' ile eşleşen dosya bulunamadı: {root_path}")
        sys.exit(1)

    logger.info(f"{root_path} içinde simülasyon için {len(mat_files)} kaynak dosya bulundu.")

    conn = get_db_connection()
    
    # Tablo varlığını kontrol et
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'feature_vectors';")
        if not cur.fetchone():
            logger.error("feature_vectors tablosu veritabanında bulunamadı!")
            conn.close()
            sys.exit(1)

    # Simülasyon saatini şu andan başlat
    current_sim_time = datetime.now(tz=timezone.utc).replace(microsecond=0)

    logger.info("Jetson doğrudan DB veri gönderim simülatörü başlatıldı...")
    try:
        while True:
            # Rastgele bir dosya seç
            file_path = random.choice(mat_files)
            scenario_label = _infer_scenario_label(file_path)
            
            logger.info(f"Dosya işleniyor: {file_path.name} | Senaryo: {scenario_label}")
            try:
                signals = load_mat_file(file_path)
                vectors = assemble_feature_vectors(signals, groups, config)
                
                if not vectors:
                    logger.warning(f"{file_path.name} dosyasından vektör üretilemedi. Geçiliyor.")
                    continue
                
                # Ölçeklendir ve Kırp (Model için hazır hale getir)
                vectors = prepare_vectors_for_model_db(vectors, config)
                feature_step_seconds = get_feature_step_seconds(groups, config)
                
                # DB kayıtlarını hazırla
                records = []
                for idx, vector in enumerate(vectors):
                    ts = current_sim_time + timedelta(seconds=idx * feature_step_seconds)
                    records.append((ts, scenario_label, vector.tolist()))
                
                logger.info(f"{len(records)} adet vektör feature_vectors tablosuna yazılıyor...")
                
                import psycopg2.extras
                with conn.cursor() as cur:
                    psycopg2.extras.execute_batch(
                        cur,
                        """
                        INSERT INTO feature_vectors (time, scenario_label, features)
                        VALUES (%s, %s, %s);
                        """,
                        records
                    )
                conn.commit()
                
                logger.info(f"Başarıyla yazıldı: {len(vectors)} vektör | Senaryo: {scenario_label}")
                
                # Simülasyon saatini ilerlet
                file_duration = len(vectors) * feature_step_seconds
                current_sim_time += timedelta(seconds=file_duration)
                
                time.sleep(args.interval)
                
            except Exception as e:
                logger.error(f"{file_path.name} işlenirken hata: {e}", exc_info=True)
                time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Simülatör durduruldu.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

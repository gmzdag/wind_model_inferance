import psycopg2
import logging
import time
from app.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

logger = logging.getLogger(__name__)

def get_db_connection():
    # Google Cloud VM üzerindeki Docker'dan DB'ye bağlanmayı dener (5 kez tekrar).
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info("TimescaleDB veritabanına başarıyla bağlanıldı.")
            return conn
        except psycopg2.OperationalError as e:
            logger.error(f"Veritabanı bağlantısı başarısız (Deneme {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("Birden fazla denemeye rağmen TimescaleDB'ye bağlanılamadı.")
                raise

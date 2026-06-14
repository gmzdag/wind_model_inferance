import logging
import numpy as np
from psycopg2.extensions import connection

logger = logging.getLogger(__name__)

def read_latest_sequence(conn: connection, limit: int = 20):
    # Veritabanından son sequence'i (örn: 20 adet) eskiden yeniye doğru okur.
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT time, features
            FROM feature_vectors
            ORDER BY time DESC
            LIMIT %s;
            """,
            (limit,)
        )
        rows = cursor.fetchall()

    if len(rows) < limit:
        return None, None

    # Zaman sırasını eskiden yeniye çevir (oldest -> newest)
    rows = list(reversed(rows))
    
    latest_time = rows[-1][0]
    
    sequence = [row[1] for row in rows]
    sequence_np = np.array(sequence, dtype=np.float32)
    
    return latest_time, sequence_np

def result_exists(conn: connection, time, model_version: str) -> bool:
    # Bu zaman damgası önceden işlendi mi kontrol eder.
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM anomaly_results
                WHERE time = %s AND model_version = %s
            );
            """,
            (time, model_version)
        )
        return cursor.fetchone()[0]

def get_last_processed_time(conn: connection):
    """anomaly_results tablosundaki en son tahmin edilen zaman damgasını döner."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT MAX(time) FROM anomaly_results;")
        val = cursor.fetchone()[0]
        return val

def get_preceding_errors(conn: connection, ref_time, count: int = 4) -> list[float]:
    """Hareketli ortalama kuyruğunu beslemek için geçmiş hata değerlerini okur."""
    if count <= 0:
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT reconstruction_error FROM anomaly_results 
            WHERE time < %s 
            ORDER BY time DESC 
            LIMIT %s;
            """,
            (ref_time, count)
        )
        rows = cursor.fetchall()
        # Oldest to newest
        return [float(r[0]) for r in reversed(rows)]

def read_feature_vectors_since(conn: connection, last_time, sequence_length: int = 20):
    """En son işlenen zamandan sonrasını, çakışmayı (overlap) korumak için 
    sequence_length kadar geriye giderek okur.
    """
    with conn.cursor() as cursor:
        if last_time is None:
            cursor.execute("SELECT time, features FROM feature_vectors ORDER BY time ASC;")
            return cursor.fetchall()
        
        # En az sequence_length kadar çakışan örnek olduğundan emin olmak için başlama zamanını bul
        cursor.execute(
            """
            SELECT MIN(time) FROM (
                SELECT time FROM feature_vectors 
                WHERE time <= %s 
                ORDER BY time DESC 
                LIMIT %s
            ) AS sub;
            """,
            (last_time, sequence_length)
        )
        start_time = cursor.fetchone()[0]
        if start_time is None:
            start_time = last_time

        cursor.execute(
            """
            SELECT time, features FROM feature_vectors 
            WHERE time >= %s 
            ORDER BY time ASC;
            """,
            (start_time,)
        )
        return cursor.fetchall()

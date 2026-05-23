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

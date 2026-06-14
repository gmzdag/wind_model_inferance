import logging
from psycopg2.extensions import connection

logger = logging.getLogger(__name__)

def write_anomaly_result(
    conn: connection,
    time,
    reconstruction_error: float,
    threshold: float,
    is_anomaly: bool,
    model_version: str
):
    # Anomali sonucunu DB'ye yazar (Google Cloud ortamı).
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO anomaly_results (
                time,
                reconstruction_error,
                threshold,
                is_anomaly,
                model_version
            )
            VALUES (%s, %s, %s, %s, %s);
            """,
            (time, reconstruction_error, threshold, is_anomaly, model_version)
        )
    conn.commit()
    logger.info("Sonuç anomaly_results tablosuna başarıyla yazıldı.")

def write_anomaly_results_batch(conn: connection, results: list[tuple]):
    # Anomali tahmin sonuçlarını toplu (batch) olarak veritabanına yazar.
    if not results:
        return
    import psycopg2.extras
    with conn.cursor() as cursor:
        psycopg2.extras.execute_batch(
            cursor,
            """
            INSERT INTO anomaly_results (
                time,
                reconstruction_error,
                threshold,
                is_anomaly,
                model_version
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            results
        )
    conn.commit()
    logger.info(f"{len(results)} adet sonuç anomaly_results tablosuna başarıyla yazıldı.")

import os
import sys
import logging
import psycopg2
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Logging kurulumu
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluator")

def print_table(headers, rows):
    """ASCII formatında güzel bir tablo yazdırır."""
    if not headers and not rows:
        return
        
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            val_str = str(val)
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val_str))
            else:
                col_widths.append(len(val_str))
                
    border = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_sep = "+" + "+".join(["=" * (w + 2) for w in col_widths]) + "+"
    
    print(border)
    header_cols = [f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)]
    print("|" + "|".join(header_cols) + "|")
    print(header_sep)
    
    for row in rows:
        row_cols = []
        for i, val in enumerate(row):
            w = col_widths[i] if i < len(col_widths) else 10
            row_cols.append(f" {str(val):<{w}} ")
        print("|" + "|".join(row_cols) + "|")
        print(border)

def connect_db():
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", 5432))
    name = os.getenv("DB_NAME", "windguard")
    user = os.getenv("DB_USER", "windguard_user")
    pwd = os.getenv("DB_PASSWORD", "ceren123")

    logger.info(f"Connecting to DB for evaluation at {host}:{port}/{name}...")
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=name,
            user=user,
            password=pwd,
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model Performans Değerlendirme")
    parser.add_argument("--version", type=str, default=os.getenv("MODEL_VERSION", "v2"),
                        help="Değerlendirilecek model sürümü (örn: v1, v2)")
    args = parser.parse_args()

    conn = connect_db()
    
    query = """
        SELECT 
            ar.time,
            fv.scenario_label,
            ar.reconstruction_error,
            ar.threshold,
            ar.is_anomaly
        FROM anomaly_results ar
        JOIN feature_vectors fv ON ar.time = fv.time
        WHERE ar.model_version = %s
        ORDER BY ar.time ASC
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, (args.version,))
            rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to execute evaluation query: {e}")
        conn.close()
        sys.exit(1)
        
    conn.close()
    
    if not rows:
        print(f"\n[!] Model sürümü '{args.version}' için eşleşen kayıt bulunamadı.")
        print("Lütfen veritabanında bu sürümle yazılmış anomali tahmini bulunduğundan emin olun.")
        return

    total = len(rows)
    print(f"\n==============================================================")
    print(f"Model Performans Değerlendirme Raporu (Model Sürümü: {args.version})")
    print(f"Toplam Pencere Sayısı: {total}")
    print(f"==============================================================\n")

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    scenario_stats = {}

    for row in rows:
        time_stamp, scenario_label, recon_err, threshold, is_anomaly = row
        is_actually_faulty = scenario_label.lower() != "healthy"
        
        if is_actually_faulty:
            if is_anomaly:
                tp += 1
            else:
                fn += 1
        else:
            if is_anomaly:
                fp += 1
            else:
                tn += 1

        if scenario_label not in scenario_stats:
            scenario_stats[scenario_label] = {"total": 0, "anomalies": 0, "normals": 0}
            
        scenario_stats[scenario_label]["total"] += 1
        if is_anomaly:
            scenario_stats[scenario_label]["anomalies"] += 1
        else:
            scenario_stats[scenario_label]["normals"] += 1

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    summary_data = [
        ["Accuracy", f"{accuracy:.4f} ({accuracy * 100:.2f}%)"],
        ["Precision (PPV)", f"{precision:.4f} ({precision * 100:.2f}%)"],
        ["Recall (Sensitivity)", f"{recall:.4f} ({recall * 100:.2f}%)"],
        ["F1-Score", f"{f1:.4f}"],
        ["False Positive Rate (FPR)", f"{fpr:.4f} ({fpr * 100:.2f}%)"],
    ]
    print("Temel Sınıflandırma Metrikleri:")
    print_table(["Metrik", "Değer"], summary_data)
    print()

    cm_data = [
        ["Actual Faulty", f"TP: {tp}", f"FN: {fn}", f"Total: {tp + fn}"],
        ["Actual Healthy", f"FP: {fp}", f"TN: {tn}", f"Total: {fp + tn}"],
    ]
    print("Confusion Matrix:")
    print_table(["", "Predicted Anomaly", "Predicted Normal", "Total"], cm_data)
    print()

    breakdown_data = []
    for sc, stats in sorted(scenario_stats.items()):
        tot = stats["total"]
        anom = stats["anomalies"]
        norm = stats["normals"]
        rate = anom / tot if tot > 0 else 0
        gt_type = "Healthy" if sc.lower() == "healthy" else "Faulty"
        rate_label = "False Alarm Rate" if sc.lower() == "healthy" else "Detection Rate"
        
        breakdown_data.append([
            sc,
            gt_type,
            tot,
            anom,
            norm,
            f"{rate * 100:.1f}% ({rate_label})"
        ])

    print("Detaylı Senaryo Dağılımı:")
    print_table(
        ["Senaryo Etiketi", "Tip", "Toplam Örnek", "Tahmin Edilen Anomali", "Tahmin Edilen Normal", "Oran"],
        breakdown_data
    )
    print()

if __name__ == "__main__":
    main()

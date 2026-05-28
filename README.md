# Wind Turbine Predictive Maintenance - Model Inference Service

Bu servis, rüzgar türbini için geliştirilmiş LSTM Autoencoder anomali tespit modelinin tahmin (inference) sürecini yürütür.

## Servisin Amacı
1. TimescaleDB içindeki `feature_vectors` tablosundan son sequence (20 feature vector) okumak.
2. Google Cloud Storage (GCS) üzerinden model eğitim artifact'lerini (model, scaler, threshold) indirmek (Model, threshold vs. GCS'ten alınır).
3. Feature verisini normalize ederek model üzerinden geçirmek (reconstruction error hesaplamak).
4. Hesaplanan hatayı threshold ile belirlenen anomali eşik değeriyle kıyaslayıp anomali kararını vermek.
5. Sonuçları `anomaly_results` tablosuna yazmak.

> **Önemli Not:** Şu an projede sadece `lstm_autoencoder.py` dosyası bulunmaktadır. `best_model.pt`, `scaler.joblib`, `threshold.json` artifact'leri Google Cloud Bucket içinde bulunmalıdır. Bu veritabanı (TimescaleDB) ve inference servisi, Google Cloud VM içindeki Docker ortamında çalışacak şekilde yapılandırılmıştır.

## Beklenen DB Tabloları

**Okunacak Tablo (feature_vectors):**
```sql
CREATE TABLE feature_vectors (
    time TIMESTAMPTZ NOT NULL,
    scenario_label TEXT DEFAULT 'unknown',
    features DOUBLE PRECISION[] NOT NULL
);
```

**Yazılacak Tablo (anomaly_results):**
```sql
CREATE TABLE anomaly_results (
    time TIMESTAMPTZ NOT NULL,
    reconstruction_error DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    is_anomaly BOOLEAN NOT NULL,
    model_version TEXT DEFAULT 'v1'
);
```

## Bucket Artifact Yapısı (Beklenen)

```text
gs://wind-turbine-pdm-models/lstm_autoencoder/v1/
├── best_model.pt
├── scaler.joblib
└── threshold.json
```
**Eksik Artifact'ler:** Bu dosyalar GCS'e yüklendiğinde servis `/app/artifacts/` dizini içine indirecektir.

## Environment Değişkenleri
` .env.example` dosyasında bulunan temel değişkenler:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`: Veritabanı bağlantı ayarları
- `MODEL_BUCKET_PATH`: GCS artifact yolu (Bucket Path)
- `MODEL_VERSION`: Yazılacak model sürümü (örn: v1)
- `LOCAL_ARTIFACT_DIR`: İndirilecek dosyaların konulacağı klasör
- `SEQUENCE_LENGTH`: Beklenen sequence uzunluğu (varsayılan: 20)
- `FEATURE_SIZE`: Beklenen feature sayısı (varsayılan: 426)
- `POLL_INTERVAL_SECONDS`: DB okuma periyodu

## Çalıştırma Komutları

### Docker Build
```bash
docker build -t wind-model-inference:latest .
```

### Google Cloud VM Üzerinde Çalıştırma
Google Cloud VM (Compute Engine) üzerinde çalıştırılırken genellikle "Service Account" otomatik olarak tanınır (GCS ve diğer Google servisleri için).
```bash
docker run -d \
  --name wind-model-inference \
  --env-file .env \
  --network host \
  wind-model-inference:latest
```

## Feature Order Uyarısı
DB'deki `features` array sırası eski eğitimdeki 426 feature sırası ile **aynı olmalıdır**. Bu servisin modeli (GCS üzerinden indirilen ağırlıklar) belirli bir feature sırasına göre eğitildiği için veritabanına veri yazan sistemin bunu koruması gerekir.

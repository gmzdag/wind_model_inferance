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

### 1. Docker Build (VM üzerinde kodlar güncellendikten sonra)
```bash
docker build -t wind-model-inference:latest .
```

### 2. Google Cloud VM Üzerinde Çalıştırma Seçenekleri

#### A. Toplu İşleme (Batch Mode) - Sıfırdan Tüm Geçmiş Verileri İşleme
Eğer veritabanındaki tüm geçmiş verileri sıfırdan anomali tespitinden geçirmek ve her adımın sonucunu terminalde canlı (`NORMAL` / `ANOMALİ`) olarak görmek istiyorsanız:
```bash
docker run --rm \
  --name wind-model-inference-batch \
  --env-file .env \
  --network host \
  wind-model-inference:latest \
  python app/main.py --mode batch --force-backfill
```
*(Bu komut çalışıp tüm verileri işledikten sonra otomatik olarak kapanır).*

#### B. Canlı Akış (Stream Mode) - Arka Planda Sürekli Çalıştırma
Veritabanına yeni veri geldikçe anomali tespitinin **canlı ve sürekli** olarak arka planda yapılması için:
```bash
# Eğer eski konteyner varsa durdurup silin:
docker stop wind-model-inference || true
docker rm wind-model-inference || true

# Konteyneri arka planda (detached) başlatın:
docker run -d \
  --name wind-model-inference \
  --env-file .env \
  --network host \
  wind-model-inference:latest
```
*Bu komut varsayılan olarak `--mode stream` ile çalışır ve her `POLL_INTERVAL_SECONDS` (varsayılan: 5 sn) sürede bir veritabanına yeni gelen verileri otomatik olarak işler.*

### 3. Logları Canlı Takip Etme (Stream Modu İçin)
Arka planda çalışan canlı akış servisinin loglarını terminalden anlık izlemek için:
```bash
docker logs -f wind-model-inference
```

### 4. Model Performansını Değerlendirme (F1, Precision, Recall vb.)
Veritabanındaki anomali sonuçlarının başarısını ölçmek ve F1 skorunu tablo halinde görmek için:
```bash
docker run --rm \
  --env-file .env \
  --network host \
  wind-model-inference:latest \
  python app/evaluate.py
```
### 5. Jetson Canlı Veri Yayını Simülatörü (Direct DB Ingestion)
Jetson cihazının sensör verilerini işleyip veritabanına canlı olarak attığı süreci simüle etmek için:
```bash
docker run --rm \
  --env-file .env \
  --network host \
  wind-model-inference:latest \
  python app/random_publisher.py --interval 2.5
```
*Bu komut Jetson edge cihazı gibi davranır: `wind-turbine-edge-processing/data` altındaki raw `.mat` dosyalarından rastgele birini seçer, FFT ve özellik çıkarımını yapar, veriyi ölçeklendirip doğrudan veritabanındaki `feature_vectors` tablosuna canlıymış gibi sırayla ekler.*

import unittest
import numpy as np
import torch
from app.lstm_autoencoder import LSTMAutoencoder
from app.inference import run_inference
from app.config import SEQUENCE_LENGTH, FEATURE_SIZE

class TestModelInference(unittest.TestCase):
    def setUp(self):
        """Test için kullanılacak cihazı ve LSTM Autoencoder modelini hazırlar."""
        self.device = torch.device("cpu")
        
        # Gerçek LSTMAutoencoder sınıfını test için örnek parametrelerle başlatıyoruz
        self.model = LSTMAutoencoder(
            input_size=FEATURE_SIZE,  # 426
            hidden_size=32,
            latent_size=16,
            num_layers=1,
            bidirectional=False
        )
        self.model.eval()

    def test_run_inference_valid_batch(self):
        """Modelin geçerli bir batch 3D veri aldığında doğru boyutta ve tipte hata ürettiğini test eder."""
        batch_size = 4
        # Rastgele 3D veri üret (Batch, SequenceLength, FeatureSize) -> (4, 20, 426)
        X = np.random.rand(batch_size, SEQUENCE_LENGTH, FEATURE_SIZE).astype(np.float32)

        # Çıkarım fonksiyonunu çalıştır
        errors = run_inference(X, self.model, self.device)

        # Çıktı doğrulamaları (Assertions)
        self.assertIsInstance(errors, np.ndarray, "Çıktı bir numpy array olmalıdır.")
        self.assertEqual(errors.shape, (batch_size,), f"Çıktı boyutu batch boyutuyla ({batch_size}) eşleşmelidir.")
        self.assertTrue(np.all(errors >= 0.0), "Yeniden yapılandırma hatası (MSE) negatif olamaz.")
        
        print(f"\n[OK] Batch Testi Başarılı! Girdi Boyutu: {X.shape} -> Çıktı Hata Sayısı: {len(errors)}")

    def test_run_inference_single_sequence(self):
        """Modelin tek bir dizi (Batch=1) aldığında başarıyla çalışıp çalışmadığını test eder."""
        batch_size = 1
        X = np.random.rand(batch_size, SEQUENCE_LENGTH, FEATURE_SIZE).astype(np.float32)

        errors = run_inference(X, self.model, self.device)

        self.assertEqual(errors.shape, (1,))
        self.assertTrue(errors[0] >= 0.0)
        
        print(f"[OK] Tekli Dizi Testi Başarılı! Girdi Boyutu: {X.shape} -> Hata: {errors[0]:.6f}")

if __name__ == "__main__":
    unittest.main()

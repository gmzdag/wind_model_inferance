"""LSTM Autoencoder uygulaması.

Mimari, Malhotra ve ark. (2016) "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"
makalesini temel alır ancak üç büyük iyileştirme içerir:
  1. Yalnızca son gizli (hidden) durumu almak yerine tüm encoder zaman adımları üzerinden
     ortalama havuzlama (mean-pooling) yapılır — bu sayede tüm penceredeki zamansal bağlam yakalanır.
  2. Sıkıştırılmış (latent) vektör, her bir decoder katmanı için (h_0, c_0) başlangıç durumlarına
     yansıtılır (projected). Böylece decoder LSTM, sıkıştırılmış temsilden doğru şekilde başlatılır.
  3. Decoder, diziyi zamansal olarak ters sırada yeniden oluşturur (reconstructs).
     Bu durum encoder'ı daha kompakt ve ayırt edici bir darboğaz (bottleneck) temsili öğrenmeye
     zorlar (Srivastava ve ark. 2015, "Unsupervised Learning of Video Representations").
     Çıktı döndürülmeden önce tekrar ileri (doğru) sıraya çevrilir.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """LSTM encoder ve decoder içeren dizi otoenkodlayıcı (sequence autoencoder)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Çift yönlü (Bidirectional) encoder, hidden_size * 2 boyutunda çıktılar üretir
        encoder_out_size = hidden_size * 2 if bidirectional else hidden_size

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.to_latent = nn.Linear(encoder_out_size, latent_size)
        self.latent_norm = nn.LayerNorm(latent_size)
        self.latent_dropout = nn.Dropout(p=dropout)

        # Latent vektörü → her bir decoder katmanı için başlangıç hidden (h0) ve cell (c0) durumlarına yansıt (project).
        # Şekil (Shape): (num_layers * hidden_size), sonrasında h_0 ve c_0 olarak ayrı ayrı bölünecek.
        self.latent_to_h0 = nn.Linear(latent_size, num_layers * hidden_size)
        self.latent_to_c0 = nn.Linear(latent_size, num_layers * hidden_size)

        # Decoder her zaman tek yönlüdür (otoregresif yeniden yapılandırma - autoregressive reconstruction)
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Bir girdi (input) batch'ini ortalama havuzlama (mean-pooling) ile latent (gizli) vektörlere kodlar."""
        encoded_sequence, _ = self.encoder(inputs)
        if self.bidirectional:
            # İleri ve geri (forward/backward) çıktıları ayır ve her bir yarıyı bağımsız olarak mean-pool yap.
            fwd = encoded_sequence[:, :, : self.hidden_size]        # (B, T, H)
            bwd = encoded_sequence[:, :, self.hidden_size :]        # (B, T, H)
            pooled = torch.cat([fwd.mean(dim=1), bwd.mean(dim=1)], dim=-1)  # (B, 2H)
        else:
            pooled = encoded_sequence.mean(dim=1)                   # (B, H)
        latent = self.to_latent(pooled)
        latent = self.latent_norm(latent)
        latent = self.latent_dropout(latent)
        return latent

    def decode(self, latent: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Girdi dizisini yeniden oluşturarak (reconstruct) bir latent vektörü çözer (decode).

        Eğitim (self.training=True):
            Tersine çevrilmiş girdi ile "Teacher-forcing" yapılır (Srivastava ve ark. 2015).
            Decoder, her adımda gerçek tersine çevrilmiş diziyi (ground-truth) alır,
            bu da darboğazın (bottleneck) öğrenilmesi için kararlı gradyanlar sağlar.

        Çıkarım / Tahmin (self.training=False):
            Serbest çalışma (otoregresif - auto-regressive) modu. Decoder sıfır token'ı ile başlar ve 
            kendi çıktısını bir sonraki girdi olarak besler. Bu durum, yeniden yapılandırma 
            kalitesinin tamamen latent (gizli) temsile bağlı olmasını zorlar — zayıf sıkıştırılan 
            anormal sinyaller daha yüksek yeniden yapılandırma hataları (reconstruction error) üretir,
            bu da ayrılabilirliği artırır.

        Her iki modda da çıktı ileri (doğru) zamansal sırada döndürülür.
        """
        batch_size, seq_len, _ = inputs.size()

        # Decoder LSTM başlangıç durumunu (state) latent vektörden başlat (initialise).
        h0 = self.latent_to_h0(latent).view(batch_size, self.num_layers, self.hidden_size)
        h0 = h0.permute(1, 0, 2).contiguous()   # (num_layers, B, H)
        c0 = self.latent_to_c0(latent).view(batch_size, self.num_layers, self.hidden_size)
        c0 = c0.permute(1, 0, 2).contiguous()   # (num_layers, B, H)

        if self.training:
            # Teacher-forcing: gerçek zamanlı tersine çevrilmiş diziyi (ground-truth) besle (Srivastava ve ark. 2015).
            reversed_inputs = torch.flip(inputs, dims=[1])
            decoded_sequence, _ = self.decoder(reversed_inputs, (h0, c0))
            reversed_output = self.output_layer(decoded_sequence)   # (B, T, input_size)
            # Orijinal ileri dizi ile hizalanması (align) için çıktıyı tekrar tersine çevir.
            return torch.flip(reversed_output, dims=[1])
        else:
            # Serbest çalışma (Free-run): decoder sıfır (zero) başlangıç token'ı ile otoregresif olarak üretim yapar.
            # Tahmin edilen her frame (kare/adım), bir sonraki decoder adımı için girdi olur.
            hidden = (h0, c0)
            current_input = torch.zeros(
                batch_size, 1, self.input_size, device=inputs.device, dtype=inputs.dtype
            )
            outputs: list[torch.Tensor] = []
            for _ in range(seq_len):
                out, hidden = self.decoder(current_input, hidden)   # (B, 1, H)
                pred = self.output_layer(out)                        # (B, 1, input_size)
                outputs.append(pred)
                current_input = pred
            # Çıktılar ters sırada üretildi (decoder geriye doğru yeniden yapılandırır);
            # bunları birleştir (concatenate) ve ileri (doğru) sıraya çevir (flip).
            return torch.flip(torch.cat(outputs, dim=1), dims=[1])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Bir girdi batch'ini yeniden oluştur (reconstruct)."""
        latent = self.encode(inputs)
        return self.decode(latent, inputs)

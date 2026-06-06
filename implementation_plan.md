# Rancangan Sistem: Next-Gen Automated Trading Bot dengan Deep Learning (CNN-LSTM-Transformer)

Dokumen ini merupakan rancangan pengembangan sistem (System Design Document) untuk sistem trading otomatis *end-to-end* yang diusulkan. Rancangan ini diformulasikan untuk memiliki **kebaruan (novelty)** yang signifikan dibandingkan dengan artikel referensi awal, menjadikannya layak sebagai sebuah inovasi maupun riset lanjutan.

## 1. Latar Belakang & Identifikasi Gap (Kebaruan)

Sistem pada artikel referensi menggunakan pendekatan yang solid (CNN-LSTM dengan siklus *retraining* otomatis). Namun, terdapat beberapa celah (gap) yang biasa ditemui dalam model finansial prediktif:
1. **Lagging Prediction:** Model regresi harga cenderung hanya meniru harga sebelumnya.
2. **Data Input Terbatas:** Hanya mengandalkan *Close Price* 1 jam (H1), buta terhadap tren makro dan volatilitas.
3. **Single Point of Failure:** Menggunakan satu arsitektur model tunggal, rentan terhadap noise.

### Kebaruan (Novelty) yang Diusulkan:
Untuk mengatasi gap di atas, sistem ini akan memperkenalkan 3 novelty utama:
1. **Multivariate & Multi-Timeframe Input Pipeline (Data Novelty):** 
   Tidak hanya harga *Close*, fitur input (tensor) akan diperkaya dengan 12+ indikator teknikal (RSI, MACD, Bollinger Bands) dan embedding tren dari Timeframe lebih besar (Daily/H4) sebagai konteks makro.
2. **Multi-Task Classification Objective (Model Novelty):**
   Alih-alih memprediksi harga eksak (regresi), model akan diubah menjadi klasifikasi probabilitas terkalibrasi (Naik Signifikan, Turun Signifikan, Sideways) menggunakan arsitektur gabungan **CNN-LSTM-Attention (Transformer mechanism)** untuk menangkap pola lokal dan dependensi spasial jangka panjang secara bersamaan.
3. **Adaptive Risk Sizing dengan Ensemble Voting (Execution Novelty):**
   Keputusan trading tidak hanya didasarkan pada satu model, melainkan 3 model *ensemble* (Majority Vote). Selain itu, batas *Take Profit* (TP) dan *Stop Loss* (SL) tidak kaku, melainkan menggunakan prediksi volatilitas prediktif berbasis AI, dikombinasikan dengan ATR (Average True Range).

---

## 2. Arsitektur Sistem Global

Sistem tetap mempertahankan otomatisasi *end-to-end* tanpa intervensi manusia, namun dengan *pipeline* yang lebih canggih.

### Komponen A: Data Gathering & Feature Engineering (MT5 -> Drive)
- **Sumber Data:** Script MQL5 akan mengekspor tidak hanya OHLCV, tetapi secara otomatis menghitung matriks indikator teknikal dari MT5 sebelum dikirim ke Colab.
- **Normalisasi:** Menggunakan *RobustScaler* atau *StandardScaler* daripada sekadar *MinMax* agar lebih kebal terhadap data pencilan (*outliers/spikes*).

### Komponen B: AI Training Pipeline (Google Colab / Cloud GPU)
- **Otomatisasi:** Menggunakan Google Apps Script untuk memicu *training* setiap akhir pekan.
- **Model Architecture:**
  - `Input Layer:` Tensor 3D (Batch, 120 Time Steps, 15 Features)
  - `Conv1D Layer:` Mengekstraksi fitur pola candlestick.
  - `LSTM Layer:` Memahami urutan sekuensial tren.
  - `Attention Mechanism (Novelty):` Fokus pada time-steps tertentu yang krusial (misal saat terjadi rilis berita ekonomi).
  - `Output Layer:` Softmax 3-class classification (Probabilitas Up, Down, Netral).
- **Format Export:** Dikonversi dan divalidasi ke format **ONNX v13** untuk deployment di MQL5.

### Komponen C: Trading Execution Engine (MQL5 / VPS)
- **Inference Module:** Expert Advisor (EA) MT5 me-load model ONNX untuk memproses data *real-time*.
- **Ensemble Logic:** EA menggabungkan prediksi dari beberapa skenario model untuk menekan *False Positives*.
- **Dynamic Risk Management:** Ukuran lot, TP, SL, dan *Trailing Stop* ditentukan secara dinamis berdasarkan prediksi volatilitas dari *pipeline*.

---

## 3. Langkah-Langkah Implementasi (Tahapan Pengerjaan)

> [!IMPORTANT]
> Pengembangan akan dibagi menjadi beberapa *milestone* yang bisa dites secara independen.

### Tahap 1: Data Pipeline & Feature Engineering
1. Membuat script MQL5 untuk mengekspor data OHLCV H1 beserta data teknikal indikator (RSI, Moving Averages, ATR, Volume).
2. Membangun script Python (di lokal/Colab) untuk membaca CSV, membersihkan data, menambahkan indikator *lagging/leading*, dan melakukan *data scaling*.

### Tahap 2: Model Development & Training (Python/TensorFlow)
1. Mendesain arsitektur model **CNN-LSTM-Attention**.
2. Mengubah target prediksi (Y) dari *Continuous Price* menjadi *Categorical Class* (1 = Uptrend, 0 = Sideways, -1 = Downtrend) berdasarkan ambang batas pips.
3. Menerapkan *Custom Loss Function* atau class weights untuk menangani ketidakseimbangan data (sideways biasanya dominan).
4. Menambahkan script ekspor ke format ONNX.

### Tahap 3: Orchestration (Google Apps Script)
1. Menulis Google Apps Script untuk otomatis memonitor file CSV baru di Google Drive.
2. Membangun mekanisme *trigger* untuk menjalankan notebook Colab secara *headless* via API setiap hari Sabtu.
3. Menerapkan pengiriman log report ke Telegram.

### Tahap 4: Execution & Risk Engine (MQL5)
1. Menulis kerangka utama Expert Advisor (EA) MQL5.
2. Mengintegrasikan load model ONNX dan memproses tensor dari 15+ input fitur.
3. Menerapkan logika *Ensemble* dan eksekusi pesanan dengan *Dynamic Risk Management*.
4. Menerapkan *Telegram Bot Integration* di MQL5 untuk notifikasi trade secara *real-time*.

---

## 4. Rencana Verifikasi (Verification Plan)

### Automated / Backtesting Verifikasi
- Melakukan **Walk-forward Backtesting** pada MT5 Strategy Tester menggunakan data pergerakan 1 tahun ke belakang untuk memastikan tidak ada *data leakage*.
- Membandingkan metrik rasio Sharpe, rasio Sortino, dan *Max Drawdown* antara model regresi klasik (seperti di artikel) vs model klasifikasi berfitur banyak (sistem usulan).

### Manual Verifikasi
- Menjalankan model pada *Paper Trading / Demo Account* di VPS selama 2 minggu untuk memonitor apakah logika *open/close position* dari ONNX sesuai dengan prediksi probabilitas pada saat model di-*train*.
- Verifikasi otomatisasi Colab: Mengubah jadwal *trigger* menjadi 1 jam ke depan untuk melihat apakah siklus pelatihan dapat berjalan secara mandiri dan mengirim pesan Telegram yang sukses.

---

## User Review Required / Open Questions

> [!TIP]
> Sebelum memulai eksekusi penulisan kode, saya membutuhkan konfirmasi Anda terkait beberapa keputusan teknis ini:

1. **Pemilihan Fitur Indikator:** Untuk fitur input tambahan, apakah Anda memiliki preferensi indikator teknikal spesifik (misal: MACD, RSI, EMA, Parabolic SAR), atau saya boleh menentukan kombinasi terbaik secara mandiri?
2. **Kategori Trading:** Apakah fokus simulasi trading dan pembentukan model ini masih untuk pair Forex yang sama (seperti EURUSD, GBPUSD), atau Anda berencana merambat ke aset lain (seperti Crypto atau Index/XAUUSD)?
3. **Mekanisme Eksekusi:** Apakah Anda lebih suka kita membangun ini tahap demi tahap (dimulai dari membuat script eksportir MT5 dahulu), atau langsung fokus membuat prototipe arsitektur Deep Learning di Python terlebih dahulu menggunakan *dummy data/historical data*?

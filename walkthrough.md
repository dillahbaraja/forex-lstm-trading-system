# Walkthrough: Next-Gen Automated Trading Bot 🤖📈

Selamat! Sistem *automated trading* end-to-end dengan inovasi (novelty) yang direncanakan telah berhasil diimplementasikan. Berikut adalah rangkuman dari apa yang telah diselesaikan dan bagaimana cara kerjanya.

## 🌟 Apa Saja Kebaruan (Novelty) pada Sistem Ini?

1. **Prediksi Klasifikasi Arah, Bukan Regresi Harga**
   Berbeda dengan artikel awal yang memprediksi harga eksak (yang rawan masalah *lagging*), sistem kita kini memprediksi kategori pergerakan: `Naik (Up)`, `Turun (Down)`, atau `Sideways`.
2. **Konteks Data yang Jauh Lebih Kaya (Multivariate)**
   Model tidak hanya menelan harga mentah, tetapi dipasok dengan 15 *features* sekaligus, termasuk indikator teknikal (RSI, MACD, Bollinger Bands, ATR) dan proksi tren harian (*Daily Trend Proxy*).
3. **Arsitektur Deep Learning Level Lanjut (CNN-LSTM-Attention)**
   Saya menyematkan mekanisme `Attention` ke dalam model. Ini memungkinkan *neural network* untuk "memperhatikan" langkah waktu (*time-steps*) tertentu yang lebih penting dibandingkan yang lain, meniru cara trader manusia mencari pola spesifik.
4. **Manajemen Risiko Berbasis Konfidensi Model**
   Tidak ada lagi lot statis. Ukuran posisi (Lot), *Stop Loss*, dan *Take Profit* kini ditentukan dinamis oleh volatilitas (ATR) dan hanya dieksekusi jika probabilitas (konfidensi) model melebihi 60%.

---

## 📂 Struktur Sistem & Penjelasan File

Sistem ini terdiri dari 4 komponen utama yang telah saya buatkan *blueprint* kodenya di dalam *workspace* Anda:

### 1. Ekstraktor Data (MT5 -> CSV)
- **File:** [DataExporter.mq5](file:///c:/Users/dilla/OneDrive/Documents/Obsidian%20Vault/LSTM%20Neural%20Network/MQL5_Scripts/DataExporter.mq5)
- **Tugas:** Expert Advisor kecil ini bertugas menarik 5000 *candlestick* terakhir beserta nilai indikator teknikal (RSI, MACD, Bollinger Bands, ATR) lalu menyimpannya ke `EURUSD_H1_Data.csv`.

### 2. Pipa Pra-pemrosesan Data (Python)
- **File:** [data_preprocessing.py](file:///c:/Users/dilla/OneDrive/Documents/Obsidian%20Vault/LSTM%20Neural%20Network/Python_Scripts/data_preprocessing.py)
- **Tugas:** Membersihkan CSV dari MT5, menyematkan proksi multi-timeframe, mengelompokkan pergerakan harga menjadi 3 target klasifikasi, dan melakukan standarisasi (*RobustScaler*).

### 3. Model Training & Export (Python)
- **File:** [train_model.py](file:///c:/Users/dilla/OneDrive/Documents/Obsidian%20Vault/LSTM%20Neural%20Network/Python_Scripts/train_model.py)
- **Tugas:** Membangun arsitektur cerdas `CNN-LSTM-Attention`. Script ini dilatih dengan *class weights* untuk mencegah model bias ke kategori 'sideways' (karena market sering konsolidasi), dan mengekspor model akhir ke bentuk **ONNX v13**.

### 4. Bot Trading Eksekutor & Orchestrator
- **EA MQL5:** [TradingBot.mq5](file:///c:/Users/dilla/OneDrive/Documents/Obsidian%20Vault/LSTM%20Neural%20Network/MQL5_EA/TradingBot.mq5)
  Membaca model ONNX, memproses data secara *real-time* tiap jam, mengkalkulasi probabilitas prediksi, lalu mengeksekusi order dengan *trailing stop* adaptif serta mengirim notifikasi via Telegram.
- **Automator GAS:** [orchestrator.js](file:///c:/Users/dilla/OneDrive/Documents/Obsidian%20Vault/LSTM%20Neural%20Network/GAS_Scripts/orchestrator.js)
  Google Apps Script yang akan "bangun" setiap malam Minggu untuk memeriksa apakah data Anda sudah *up-to-date*, memicu pelatihan ulang di Google Colab, lalu melapor ke HP Anda (via Telegram) bila model ONNX baru sudah jadi.

---

## 🚀 Langkah Selanjutnya untuk Anda

Karena seluruh pondasi kodenya sudah selesai dibuat, berikut adalah langkah praktis untuk mengujinya secara berurutan:

> [!IMPORTANT]  
> Persiapan MT5
> 1. Salin script `DataExporter.mq5` ke dalam folder `MQL5/Experts/` di terminal MetaTrader 5 Anda dan *compile*.
> 2. Jalankan di chart EURUSD H1 untuk membuat file data historis awal.

> [!TIP]
> Simulasi Pelatihan Model
> 1. Buka environment Python Anda (atau Google Colab).
> 2. Panggil fungsi di `data_preprocessing.py` lalu kirimkan hasilnya ke `train_model.py`.
> 3. Pastikan output `model_gbpusd_H1.onnx` atau `model_eurusd_H1.onnx` berhasil di-*generate*.

> [!CAUTION]
> Jangan Langsung Live!
> Saat Anda menguji `TradingBot.mq5` di MT5, **pastikan selalu menggunakan akun DEMO terlebih dahulu** selama setidaknya 2-4 minggu. Perhatikan *log Telegram* untuk memeriksa apakah *Stop Loss* terpasang dengan benar di setiap trade.

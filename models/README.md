# Direktori Models

File model yang dibutuhkan aplikasi ini.

## Sudah tersedia di folder ini
- `viral_classifier.pkl` - hasil training model klasifikasi & regresi viralitas (MLP + scaler + label encoder)
- `topic_results.pkl` - hasil ekstraksi topik dari BERTopic (cache, opsional dimuat ulang lewat tombol "Load Hasil Topik Tersimpan")
- `forecast_results.pkl` - hasil forecasting LightGBM (cache, opsional dimuat ulang lewat tombol "Load Hasil Forecast Tersimpan")

## Belum tersedia, perlu di-download manual
- `bertopic(1).pkl` (~400 MB) - model BERTopic terlatih yang dipakai di halaman Model Forecasting (`BERTOPIC_MODEL_PATH` pada `modules/forecasting.py`). File ini tidak disertakan di repo karena ukurannya di atas batas GitHub, dihosting terpisah sebagai asset di **GitHub Releases** repo ini.

Cara mendapatkannya:
1. Buka halaman **Releases** repo ini di GitHub.
2. Download file `bertopic(1).pkl` dari asset release yang tersedia.
3. Letakkan file tersebut di folder `models/` pada clone lokal kamu, dengan nama persis `bertopic(1).pkl`.

Aplikasi akan otomatis mendeteksi model ini saat halaman Model Forecasting dibuka. Jika file belum ada, akan muncul pesan error yang mengarahkan untuk menyiapkan file model tersebut.

## Model IndoBERTweet & RoBERTa
Model teks (IndoBERTweet untuk embedding, RoBERTa untuk sentimen) dimuat langsung dari Hugging Face lewat `AutoModel.from_pretrained(...)` saat aplikasi berjalan, jadi tidak perlu disimpan manual di folder ini. Pastikan environment yang menjalankan aplikasi memiliki akses internet saat pertama kali dijalankan agar bobot model bisa diunduh (lalu di-cache secara lokal).

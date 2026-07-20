# VIRAL-Net

**Virality Integration and Representation Analysis Learning Network**

VIRAL-Net adalah dashboard berbasis Streamlit yang menggabungkan dua alur analisis data tweet:

1. **Klasifikasi Viralitas & Prediksi Engagement** - memprediksi apakah sebuah tweet berpotensi viral, sekaligus memperkirakan jumlah engagement yang akan didapatkan, lengkap dengan evaluasi model dan feature importance.
2. **Forecasting Frekuensi Topik, Hashtag, dan Kata** - mengelompokkan topik dari kumpulan tweet dan memprediksi tren frekuensi kemunculan topik, hashtag, serta kata di masa depan.

## Arsitektur Model

| Komponen | Model | Fungsi |
|---|---|---|
| Text encoder | IndoBERTweet | Ekstraksi representasi semantik dari teks tweet berbahasa Indonesia |
| Topic modeling | BERTopic | Identifikasi dan klasterisasi topik dari setiap tweet |
| Sentiment | RoBERTa | Ekstraksi fitur polaritas emosi pada tweet |
| Forecasting | LightGBM | Time series forecasting frekuensi topik, hashtag, dan kata |
| Fusion & klasifikasi | Multi-Layer Perceptron (MLP) | Menggabungkan embedding teks, sentimen, dan metadata akun untuk klasifikasi viralitas dan regresi engagement |

## Struktur Proyek

```
viral-net/
├── app.py                     # Entry point Streamlit
├── modules/
│   ├── data_preprocessing.py  # Upload, cleaning, dan eksplorasi data tweet
│   ├── classification.py      # Training, evaluasi, dan prediksi viralitas
│   └── forecasting.py         # Ekstraksi topik & forecasting frekuensi
├── models/
│   ├── viral_classifier.pkl   # Model klasifikasi & regresi viralitas terlatih
│   ├── topic_results.pkl      # Cache hasil ekstraksi topik
│   ├── forecast_results.pkl   # Cache hasil forecasting
│   └── README.md              # Catatan file model tambahan yang perlu disiapkan manual
├── requirements.txt
└── README.md
```

## Instalasi

Disarankan menggunakan Python 3.10 atau 3.11 di dalam virtual environment.

```bash
git clone <url-repo-anda>
cd viral-net

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Catatan:
- `torch` di `requirements.txt` menginstal versi CPU secara default. Jika ingin memakai GPU, install `torch` sesuai instruksi resmi di [pytorch.org](https://pytorch.org/get-started/locally/) sebelum menjalankan `pip install -r requirements.txt`.
- Model IndoBERTweet (`indolem/indobertweet-base-uncased`) dan RoBERTa sentimen (`w11wo/indonesian-roberta-base-sentiment-classifier`) diunduh otomatis dari Hugging Face saat aplikasi pertama kali dijalankan, jadi koneksi internet dibutuhkan pada run pertama.
- Lihat `models/README.md` untuk file model `bertopic(1).pkl` yang belum disertakan di repo ini dan perlu ditambahkan manual.

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501` dengan empat halaman navigasi:
- **Home** - ringkasan sistem dan arsitektur model
- **Data & Preprocessing** - upload data mentah dan proses pembersihan
- **Model Forecasting** - ekstraksi topik dengan BERTopic dan forecasting dengan LightGBM
- **Analisis Pendukung** - training/evaluasi model klasifikasi viralitas dan prediksi pada tweet baru

## Data yang Dibutuhkan

Data tweet yang sudah dipreprocessing minimal berisi kolom berikut untuk halaman klasifikasi:
- `final_tweet` - teks tweet yang sudah dibersihkan
- `followers`, `following` - metadata akun
- `like`, `retweet`, `reply`, `quote` - metrik interaksi
- `verified_status` - status verifikasi akun

## Lisensi

Tambahkan berkas lisensi sesuai kebutuhan (misalnya MIT License) sebelum repo dipublikasikan.

---
Developed by Deannisa SP

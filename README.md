# 🌿 Smart Farming Bayam Brazil

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![CNN 1D](https://img.shields.io/badge/CNN_1D-84.38%25-1D9E75?style=for-the-badge)
![CNN 2D](https://img.shields.io/badge/CNN_2D-100%25-1D9E75?style=for-the-badge)

**Sistem monitoring kelembapan tanah berbasis CNN & IoT dengan deteksi kesiapan panen menggunakan computer vision.**

[▶️ Buka di Google Colab](https://colab.research.google.com/github/diklatbkpsdm23-blip/Smart-Farming-Bayam-Brazil/blob/main/Kel3bayam.ipynb) · [📁 Dataset](#-dataset) · [📖 Cara Menjalankan](#%EF%B8%8F-cara-menjalankan)

</div>

---

## 📋 Daftar Isi
- [Gambaran Umum](#-gambaran-umum)
- [Dataset](#-dataset)
- [CNN 1D — Klasifikasi Status Tanah](#-subsistem-1--cnn-1d-klasifikasi-status-tanah)
- [Monitoring Realtime & LSCM](#-subsistem-2--monitoring-realtime--tren-lscm)
- [CNN 2D — Deteksi Kesiapan Panen](#-subsistem-3--cnn-2d-deteksi-kesiapan-panen)
- [Struktur File](#-struktur-file)
- [Cara Menjalankan](#%EF%B8%8F-cara-menjalankan)
- [Teknologi](#-teknologi)
- [Tim](#-tim-pengembang)

---

## 🌱 Gambaran Umum

Proyek ini membangun sistem **Smart Farming** untuk tanaman Bayam Brazil menggunakan dua model CNN:

1. **CNN 1D** — Mengklasifikasikan status kelembapan tanah dari sensor IoT secara realtime
2. **CNN 2D** — Mendeteksi kesiapan panen berdasarkan citra visual daun tanaman

Sistem dilengkapi algoritma **LSCM (Least Squares Curve Method)** berbasis regresi linear untuk memprediksi tren kelembapan berikutnya, serta logika otomasi pompa air.

---

## 📊 Dataset

**File:** `Data_Kelembapan_Bayam_Brazil_1440.xlsx`

### Sampel Data (5 Baris Pertama)

| No | Tanggal | Jam | Kelembapan Tanah (%) | Suhu (°C) | Status Tanah |
|----|---------|-----|----------------------|-----------|--------------|
| 1 | 01 May 2026 | 00:00 | 71 | 24.1 | Ideal |
| 2 | 01 May 2026 | 00:31 | 76 | 25.0 | Ideal |
| 3 | 01 May 2026 | 01:02 | 72 | 26.9 | Ideal |
| 4 | 01 May 2026 | 01:33 | 85 | 24.3 | Basah |
| 5 | 01 May 2026 | 02:04 | 81 | 24.1 | Basah |

### Spesifikasi Dataset

| Atribut | Detail |
|---------|--------|
| Total data | 1.440 baris |
| Interval pembacaan | 30 menit |
| Durasi | 30 hari (Mei 2026) |
| Fitur input model | Kelembapan Tanah (%), Suhu (°C) |
| Label target | Status Tanah (3 kelas) |

### Aturan Klasifikasi Status Tanah

```
Kelembapan < 60%         →  🔥 KERING   →  Pompa Air MENYALA
60% ≤ Kelembapan ≤ 79%   →  ✅ IDEAL    →  Pompa Air MATI
Kelembapan ≥ 80%         →  💧 BASAH    →  Pompa Air MATI
```

### Split Data Training / Testing

| Set | Jumlah Data | Persentase |
|-----|------------|------------|
| Training | 1.152 data | 80% |
| Testing  | 288 data   | 20% |
| **Total**    | **1.440 data** | **100%** |

---

## 🤖 Subsistem 1 — CNN 1D: Klasifikasi Status Tanah

### Arsitektur Model

```
Input Shape: (2, 1)  ←  [Kelembapan Tanah (%), Suhu (°C)]
     │
     ▼
Conv1D(filters=32, kernel_size=1, activation='relu')
     │
     ▼
MaxPooling1D(pool_size=1)
     │
     ▼
Flatten()
     │
     ▼
Dense(64, activation='relu')
     │
     ▼
Dropout(rate=0.3)
     │
     ▼
Dense(3, activation='softmax')
     │
     ▼
Output: [Kering | Ideal | Basah]
```

**Optimizer:** Adam | **Loss:** Sparse Categorical Crossentropy | **Batch Size:** 16

---

### 📈 Hasil Training — Epoch per Epoch (30 Epochs)

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|-------|---------------|------------|--------------|----------|
| 1  | 42.10% | 2.7515 | 51.74% | 0.8753 |
| 2  | 52.78% | 1.0071 | 76.04% | 0.6268 |
| 3  | 71.35% | 0.6250 | 85.07% | 0.5285 |
| 4  | 75.09% | 0.5695 | 76.74% | 0.5054 |
| 5  | 77.60% | 0.5348 | 76.74% | 0.4768 |
| 6  | 76.91% | 0.5085 | 84.03% | 0.4594 |
| 7  | 75.78% | 0.5255 | 72.22% | 0.5107 |
| 8  | 78.99% | 0.4829 | 84.72% | 0.4097 |
| 9  | 77.69% | 0.4794 | 83.33% | 0.4086 |
| 10 | 79.60% | 0.4555 | 85.07% | 0.3934 |
| 11 | 80.12% | 0.4526 | 81.25% | 0.4248 |
| 12 | 80.90% | 0.4319 | 85.42% | 0.3705 |
| 13 | 77.86% | 0.4580 | 85.42% | 0.3707 |
| 14 | 79.77% | 0.4532 | 84.38% | 0.3645 |
| 15 | 80.56% | 0.4386 | 82.64% | 0.3926 |
| 16 | 80.38% | 0.4325 | 82.29% | 0.3689 |
| 17 | 80.90% | 0.4121 | 85.07% | 0.3430 |
| 18 | 79.86% | 0.4484 | 85.07% | 0.3483 |
| 19 | 80.99% | 0.4138 | 85.07% | 0.3433 |
| 20 | 80.73% | 0.4164 | 83.68% | 0.3481 |
| 21 | 81.16% | 0.4007 | 83.68% | 0.3544 |
| 22 | 80.21% | 0.4109 | 85.42% | 0.3314 |
| 23 | 80.38% | 0.4042 | 79.86% | 0.4021 |
| 24 | 80.56% | 0.4245 | 85.76% | 0.3183 |
| 25 | 81.34% | 0.3970 | 85.76% | 0.3292 |
| 26 | 80.38% | 0.4136 | 83.33% | 0.3527 |
| 27 | 82.03% | 0.4055 | **86.11%** | 0.3501 |
| 28 | 78.39% | 0.4477 | 83.68% | 0.3577 |
| 29 | 80.73% | 0.4074 | 79.86% | 0.4051 |
| 30 | 80.21% | 0.4100 | 84.38% | 0.3663 |

> 🏆 **Val Accuracy terbaik: 86.11%** dicapai pada Epoch 27

### Hasil Evaluasi Final

```
✅ Akurasi Pengujian Model CNN: 84.38%
✅ File tersimpan: model_cnn_bayam_brazil.h5
✅ File tersimpan: encoder_bayam_brazil.pkl
```

---

## 📡 Subsistem 2 — Monitoring Realtime & Tren LSCM

### Cara Kerja Sistem

```
Sensor IoT (DHT11 + Soil Moisture)
         │
         ▼
   Baca tiap 2 detik
   Kelembapan: 45–90%  |  Suhu: 25.0–33.0°C
         │
         ▼
   Slide Window (100 data berjalan terakhir)
         │
    ┌────┴────┐
    ▼         ▼
CNN 1D     LSCM (Linear Regression)
Klasifikasi  Prediksi tren berikutnya
    │         │
    └────┬────┘
         ▼
   Keputusan Otomasi Pompa Air
         │
         ▼
   Render Grafik Animasi Realtime
```

### Contoh Output Dashboard Realtime (Log ke-87)

```
======================================================================
 DATA DASHBOARD MONITORING REALTIME - LOG KE-87
======================================================================
 🟢 BACA SENSOR -> Kelembapan Tanah: 88%  |  Suhu Udara: 27.7°C
 🤖 MODEL CNN   -> Klasifikasi Status Tanah  : Basah
 📈 MODEL LSCM  -> Proyeksi Tren Berikutnya  : 69.50%
 ⚡ AUTOMATION  -> Keputusan Sakelar Pompa   : ❌ POMPA AIR MATI (Tanah Terlalu Basah)
======================================================================
```

### Logika Otomasi Pompa Air

| Kondisi | Status Pompa | Keterangan |
|---------|-------------|------------|
| Kelembapan < 60% | 🔥 **MENYALA** | Tanah kering, perlu irigasi |
| 60% ≤ Kelembapan ≤ 79% | 🛑 **MATI** | Kondisi ideal, tidak perlu irigasi |
| Kelembapan ≥ 80% | ❌ **MATI** | Tanah terlalu basah, hentikan irigasi |

---

## 🍃 Subsistem 3 — CNN 2D: Deteksi Kesiapan Panen

### Arsitektur Model

```
Input: Citra Daun (150 × 150 × 3 piksel)
     │
     ▼
Conv2D(32, 3×3, relu) → MaxPooling2D(2×2)
     │
     ▼
Conv2D(64, 3×3, relu) → MaxPooling2D(2×2)
     │
     ▼
Conv2D(128, 3×3, relu) → MaxPooling2D(2×2)   ← Ekstraksi fitur batang
     │
     ▼
Flatten() → Dense(256, relu) → Dropout(0.4)
     │
     ▼
Dense(3, softmax)
     │
     ▼
Output: [Belum Panen | Mendekati Panen | Siap Panen]
```

**Optimizer:** Adam | **Loss:** Categorical Crossentropy | **Data Augmentation:** Rotasi 20°, Zoom 20%, Horizontal Flip

### Dataset Visual Daun

| Kelas | Jumlah Gambar | Karakteristik Visual |
|-------|--------------|----------------------|
| `belum_panen` | 40 gambar | Daun sempit, hijau muda transparan, renggang |
| `mendekati_panen` | 40 gambar | Daun sedang berkembang, hijau normal |
| `siap_panen` | 40 gambar | Daun lebar, hijau tua pekat, rimbun penuh |
| **Total** | **120 gambar** | Split 80% train / 20% validasi |

### Hasil Training CNN 2D (12 Epochs)

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|-------|---------------|------------|--------------|----------|
| 1  | 61.46% | 0.7855 | 66.67% | 0.3431 |
| 2  | 91.67% | 0.1657 | 100.00% | 0.0061 |
| 3  | 100.00% | 0.0047 | 100.00% | 0.000056 |
| 4  | 100.00% | 0.000073 | 100.00% | 0.0000083 |
| 5  | 100.00% | 0.000016 | 100.00% | 0.0000057 |
| 6  | 100.00% | 0.000045 | 100.00% | 0.0000014 |
| 7  | 100.00% | 0.0000080 | 100.00% | 0.00000087 |
| 8  | 100.00% | 0.000011 | 100.00% | 0.00000066 |
| 9  | 100.00% | 0.000093 | 100.00% | 0.00000050 |
| 10 | 100.00% | 0.0000035 | 100.00% | 0.00000050 |
| 11 | 100.00% | 0.0000050 | 100.00% | 0.00000037 |
| 12 | 100.00% | 0.00014 | 100.00% | 0.00000016 |

### Hasil Evaluasi Inference (Contoh: `siap_panen`)

```
=================================================================
        HASIL EKSTRAKSI GEOMETRI & WARNA TANAMAN (CNN 2D)
=================================================================
 HASIL DETEKSI SISTEM : SIAP_PANEN
 TINGKAT KEYAKINAN AI : 100.00% Confidence Score
-----------------------------------------------------------------
 KESIMPULAN & ANALISIS FITUR KLASIFIKASI:
 🍃 STATUS REKOMENDASI : [ SEGERA LAKUKAN PEMANENAN ]
 📌 Indikator Daun     : Daun bagian atas sudah MELEBAR (min. selebar jempol/koin)
 📌 Indikator Warna    : Hijau segar pekat dan tekstur daun tebal/renyah
 📌 Indikator Rimbun   : Kondisi rimbun penuh, lebat, dan saling menutupi pot
=================================================================
```

### Panduan Interpretasi Hasil Deteksi

| Status | Rekomendasi | Indikator Daun | Indikator Warna | Indikator Rimbun |
|--------|------------|----------------|-----------------|-----------------|
| ❌ Belum Panen | Jangan dipanen | Masih menguncup kecil / sempit | Hijau sangat muda transparan | Renggang, belum penuh |
| ⏳ Mendekati Panen | Monitoring 1–2 minggu lagi | Ukuran mulai sedang | Gradasi hijau normal | Cabang samping mulai tumbuh |
| 🍃 Siap Panen | Segera panen | Melebar (≥ selebar koin) | Hijau pekat, tekstur tebal | Lebat, saling menutupi pot |

---

## 🗂️ Struktur File

```
Smart-Farming-Bayam-Brazil/
│
├── 📓 Kel3bayam.ipynb                         ← Notebook utama (semua subsistem)
├── 📊 Data_Kelembapan_Bayam_Brazil_1440.xlsx  ← Dataset sensor 1.440 data
│
├── 🧠 model_cnn_bayam_brazil.h5               ← Model CNN 1D tersimpan
├── 🧠 model_cnn2d_bayam_3kelas.h5             ← Model CNN 2D tersimpan
├── 🏷️  encoder_bayam_brazil.pkl               ← Label encoder status tanah
├── 🏷️  label_map_panen.pkl                    ← Label map kelas panen
│
├── 📁 dataset_daun_v2/                        ← Folder dataset gambar daun
│   ├── belum_panen/      (40 gambar)
│   ├── mendekati_panen/  (40 gambar)
│   └── siap_panen/       (40 gambar)
│
└── 📄 README.md
```

---

## ⚙️ Cara Menjalankan

### Metode 1 — Google Colab (Direkomendasikan)

**Langkah 1** — Klik tombol berikut:

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diklatbkpsdm23-blip/Smart-Farming-Bayam-Brazil/blob/main/Kel3bayam.ipynb)

**Langkah 2** — Upload dataset ke tab Files (ikon folder di kiri):
```
Data_Kelembapan_Bayam_Brazil_1440.xlsx
```

**Langkah 3** — Jalankan semua cell secara berurutan:
```
Runtime → Run all   (Ctrl + F9)
```

**Langkah 4** — Untuk menghentikan monitoring realtime:
```
Klik tombol Stop (■) di kiri cell yang sedang berjalan
```

### Metode 2 — Clone Lokal

```bash
git clone https://github.com/diklatbkpsdm23-blip/Smart-Farming-Bayam-Brazil.git
cd Smart-Farming-Bayam-Brazil
pip install tensorflow scikit-learn pandas openpyxl matplotlib
jupyter notebook Kel3bayam.ipynb
```

---

## 🛠️ Teknologi

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

| Library | Versi | Kegunaan |
|---------|-------|----------|
| TensorFlow / Keras | 2.x | Membangun & melatih model CNN 1D dan 2D |
| Scikit-learn | latest | Label encoding, train-test split, regresi linear LSCM |
| Pandas | latest | Membaca & memproses dataset Excel |
| NumPy | latest | Operasi matriks dan array sensor |
| Matplotlib | latest | Visualisasi grafik training & monitoring |
| OpenPyXL | latest | Membaca file `.xlsx` |

---

## 👥 Tim Pengembang

**Kelompok 3 — STMIK Kaputama Binjai**

> Proyek ini dikembangkan sebagai bagian dari penelitian Smart Farming berbasis kecerdasan buatan untuk optimasi pertanian Bayam Brazil menggunakan teknologi IoT dan Deep Learning.

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik — STMIK Kaputama Binjai © 2026

---

<div align="center">

Dibuat dengan ❤️ oleh **Kelompok 3** · STMIK Kaputama Binjai · 2026

</div>

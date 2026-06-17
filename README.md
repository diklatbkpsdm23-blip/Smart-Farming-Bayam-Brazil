[README.md](https://github.com/user-attachments/files/29028473/README.md)
# 🌿 Smart Farming Bayam Brazil

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Akurasi](https://img.shields.io/badge/Akurasi_CNN-84.38%25-1D9E75?style=for-the-badge)

**Sistem monitoring kelembapan tanah berbasis CNN & IoT dengan deteksi kesiapan panen menggunakan computer vision.**

[▶️ Buka di Google Colab](https://colab.research.google.com/github/diklatbkpsdm23-blip/Smart-Farming-Bayam-Brazil/blob/main/Kel3bayam.ipynb) · [📁 Lihat Dataset](#dataset) · [📖 Dokumentasi](#cara-menjalankan)

</div>

---

## 📊 Hasil Model

| Metrik | Nilai |
|--------|-------|
| 🎯 Akurasi CNN 1D (Kelembapan) | **84.38%** |
| 🍃 Akurasi CNN 2D (Deteksi Panen) | **100%** |
| 📦 Total Data Sensor | **1.440 data** |
| 🔁 Epochs Training | **30 epochs** |
| ✂️ Split Data | **80% train / 20% test** |

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 📡 **Monitoring Realtime** | Pembacaan sensor kelembapan & suhu dengan grafik animasi berjalan tiap 2 detik |
| 🤖 **Klasifikasi CNN 1D** | Deteksi status tanah (Kering / Ideal / Basah) secara otomatis |
| 🍃 **Deteksi Panen CNN 2D** | Analisis visual daun untuk menentukan kesiapan panen (3 kelas) |
| 📈 **Tren LSCM** | Prediksi tren kelembapan berikutnya menggunakan regresi linear |
| ⚡ **Otomasi Pompa Air** | Keputusan nyala/mati pompa berdasarkan ambang batas aturan |
| 📊 **Visualisasi Menarik** | Dashboard dark mode dengan grafik kurva akurasi, loss, dan distribusi data |

---

## 🧠 Arsitektur Model

### CNN 1D — Klasifikasi Status Tanah
```
Input (Kelembapan %, Suhu °C)
    ↓
Conv1D(32, kernel=1, relu)
    ↓
MaxPooling1D
    ↓
Flatten
    ↓
Dense(64, relu) → Dropout(0.3)
    ↓
Dense(3, softmax) → [Kering | Ideal | Basah]
```

### CNN 2D — Deteksi Kesiapan Panen
```
Input Citra Daun (150x150x3)
    ↓
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
    ↓
Flatten → Dense(256, relu) → Dropout(0.4)
    ↓
Dense(3, softmax) → [Belum Panen | Mendekati Panen | Siap Panen]
```

---

## 🗂️ Struktur File

```
Smart-Farming-Bayam-Brazil/
│
├── 📓 Kel3bayam.ipynb                        # Notebook utama Google Colab
├── 📊 Data_Kelembapan_Bayam_Brazil_1440.xlsx # Dataset sensor (1.440 data)
├── 🧠 model_cnn_bayam_brazil.h5              # Model CNN 1D tersimpan
├── 🧠 model_cnn2d_bayam_3kelas.h5            # Model CNN 2D tersimpan
├── 🏷️ encoder_bayam_brazil.pkl               # Label encoder status tanah
├── 🏷️ label_map_panen.pkl                    # Label map kelas panen
└── 📄 README.md
```

---

## ⚙️ Cara Menjalankan

### Metode 1 — Google Colab (Direkomendasikan)

**Langkah 1** — Klik tombol di bawah untuk membuka notebook:

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diklatbkpsdm23-blip/Smart-Farming-Bayam-Brazil/blob/main/Kel3bayam.ipynb)

**Langkah 2** — Upload dataset ke tab Files Colab:
```
Data_Kelembapan_Bayam_Brazil_1440.xlsx
```

**Langkah 3** — Jalankan semua cell:
```
Runtime → Run all  (Ctrl+F9)
```

### Metode 2 — Clone via Git

```bash
git clone https://github.com/diklatbkpsdm23-blip/Smart-Farming-Bayam-Brazil.git
cd Smart-Farming-Bayam-Brazil
pip install tensorflow scikit-learn pandas openpyxl matplotlib
jupyter notebook Kel3bayam.ipynb
```

---

## 📦 Dataset

Dataset berisi **1.440 data** hasil simulasi sensor IoT selama 30 hari (interval 30 menit).

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `No` | Integer | Nomor urut data |
| `Tanggal` | Date | Tanggal pembacaan sensor |
| `Jam` | Time | Waktu pembacaan |
| `Kelembapan Tanah (%)` | Float | Persentase kelembapan tanah |
| `Suhu (°C)` | Float | Suhu udara sekitar |
| `Status Tanah` | String | Kering / Ideal / Basah |

**Aturan klasifikasi status tanah:**

```
Kelembapan < 60%        → 🔥 KERING   (Pompa MENYALA)
60% ≤ Kelembapan ≤ 79%  → ✅ IDEAL    (Pompa MATI)
Kelembapan ≥ 80%        → 💧 BASAH    (Pompa MATI)
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

---

## 👥 Tim Pengembang

**Kelompok 3 — STMIK Kaputama Binjai**

> Proyek ini dikembangkan sebagai bagian dari penelitian Smart Farming berbasis kecerdasan buatan untuk optimasi pertanian Bayam Brazil menggunakan teknologi IoT dan Deep Learning.

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik — STMIK Kaputama Binjai © 2026

---

<div align="center">
  Dibuat dengan ❤️ oleh Kelompok 3 · STMIK Kaputama Binjai · 2026
</div>

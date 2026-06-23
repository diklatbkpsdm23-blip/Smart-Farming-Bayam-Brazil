import streamlit as st
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set halaman Streamlit menjadi lebar (Wide)
st.set_page_config(page_title="Smart Farming Bayam Brazil", layout="wide")

st.title("🌿 Smart Farming Bayam Brazil — Dashboard Realtime")
st.write("Aplikasi ini membaca model CNN & simulasi data LSCM secara realtime.")

# --- TUNGGU DATA & MODEL SELESAI DIUNGGAH ---
@st.cache_resource
def load_models_and_data():
    # Load model dan encoder
    model_realtime = tf.keras.models.load_model("model_cnn_bayam_brazil.h5")
    with open("encoder_bayam_brazil.pkl", "rb") as f:
        encoder_realtime = pickle.load(f)
    # Load dataset historis
    data_historis = pd.read_excel("Data_Kelembapan_Bayam_Brazil_1440.xlsx")
    return model_realtime, encoder_realtime, data_historis

try:
    model_realtime, encoder_realtime, data_historis = load_models_and_data()
    st.success("✅ Model dan Dataset Historis Berhasil Dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat file. Pastikan file model .h5, encoder .pkl, dan file Excel sudah di-upload ke GitHub. Error: {e}")
    st.stop()

# --- FUNGSI SIMULASI SENSOR ---
def dapatkan_data_sensor_realtime():
    kelembapan_sekarang = np.random.randint(45, 90)
    suhu_sekarang = round(np.random.uniform(25.0, 33.0), 1)
    return kelembapan_sekarang, suhu_sekarang

# --- FUNGSI VISUALISASI STREAMLIT (Modifikasi dari kode asli Anda) ---
def plot_realtime_streamlit(kelembapan_history, lscm_predictions, kelembapan, suhu, status_tanah_cnn, status_pompa, tren_berikutnya, counter):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor('#0d1117')

    warna_panel = '#161b22'
    warna_text  = '#e6edf3'
    warna_muted = '#8b949e'
    warna_bdr   = '#30363d'

    # PANEL KIRI: GRAFIK KELEMBAPAN
    ax1.set_facecolor(warna_panel)
    for sp in ax1.spines.values():
        sp.set_edgecolor(warna_bdr)

    x = range(len(kelembapan_history))
    ax1.plot(x, kelembapan_history, color='#58a6ff', linewidth=1.8, label='Sensor Realtime', zorder=3)
    ax1.fill_between(x, kelembapan_history, 30, alpha=0.1, color='#58a6ff')
    ax1.plot(x, lscm_predictions, color='#f0883e', linewidth=2, linestyle='--', label='Tren LSCM', zorder=4)

    ax1.axhline(60, color='#f78166', linestyle=':', linewidth=1.4, alpha=0.8, label='Batas Kering (60%)')
    ax1.axhline(80, color='#3fb950', linestyle=':', linewidth=1.4, alpha=0.8, label='Batas Basah (80%)')

    ax1.set_title(f'📡 Monitoring Kelembapan Realtime — Log ke-{counter}', color=warna_text, fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylim(30, 110)
    ax1.tick_params(colors=warna_muted, labelsize=9)
    ax1.grid(axis='y', alpha=0.12, color=warna_bdr)
    ax1.legend(facecolor=warna_panel, edgecolor=warna_bdr, labelcolor=warna_text, fontsize=8, loc='upper left')

    # PANEL KANAN: CARD STATUS
    ax2.set_facecolor(warna_panel)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.axis('off')

    if 'MENYALA' in status_pompa:
        warna_pompa, icon_pompa = '#f78166', '🔥'
    elif 'BASAH' in status_pompa:
        warna_pompa, icon_pompa = '#58a6ff', '💧'
    else:
        warna_pompa, icon_pompa = '#3fb950', '✅'

    warna_cnn = {'Ideal': '#3fb950', 'Basah': '#58a6ff', 'Kering': '#f78166'}
    w_cnn = warna_cnn.get(status_tanah_cnn, '#e6edf3')

    items = [
        (0.88, '🌡️', 'Kelembapan Tanah', f'{kelembapan}%', '#58a6ff'),
        (0.70, '🌡️', 'Suhu Udara', f'{suhu}°C', '#f0883e'),
        (0.52, '🤖', 'CNN Klasifikasi', status_tanah_cnn, w_cnn),
        (0.34, '📈', 'Tren LSCM', f'{tren_berikutnya:.1f}%', '#d2a8ff'),
        (0.14, icon_pompa, 'Pompa Air', 'MENYALA' if 'MENYALA' in status_pompa else 'MATI', warna_pompa),
    ]

    for y, icon, label, nilai, warna in items:
        ax2.text(0.08, y + 0.04, icon, fontsize=14, va='center')
        ax2.text(0.22, y + 0.06, label, color=warna_muted, fontsize=8, va='center')
        ax2.text(0.22, y - 0.02, nilai, color=warna, fontsize=13, fontweight='bold', va='center')
        ax2.axhline(y - 0.09, color=warna_bdr, linewidth=0.5, alpha=0.5)

    st.pyplot(fig) # Trik Utama: Mengirim gambar matplotlib ke halaman Streamlit
    plt.close(fig)

# --- TOMBOL KONTROL SIMULASI ---
mulai_simulasi = st.checkbox("▶️ Jalankan Monitoring Sensor Realtime", value=False)

# Inisialisasi awal data historis 100 terakhir
kelembapan_history = data_historis['Kelembapan Tanah (%)'].iloc[-100:].values.tolist()

# Wadah kosong (Container) tempat grafik akan terus di-update
grafik_placeholder = st.empty()

if mulai_simulasi:
    counter = 1
    while mulai_simulasi:
        kelembapan, suhu = dapatkan_data_sensor_realtime()
        kelembapan_history.append(kelembapan)

        x = np.arange(len(kelembapan_history)).reshape(-1, 1)
        regresi = LinearRegression().fit(x, kelembapan_history)
        lscm_predictions = regresi.predict(x).tolist()
        tren_berikutnya = lscm_predictions[-1]

        if len(kelembapan_history) > 100:
            kelembapan_history.pop(0)
            lscm_predictions.pop(0)

        input_sensor = np.array([[kelembapan, suhu]]).reshape(1, 2, 1)
        prediksi_output = model_realtime.predict(input_sensor, verbose=0)
        kelas_id = np.argmax(prediksi_output)
        status_tanah_cnn = encoder_realtime.inverse_transform([kelas_id])[0]

        if kelembapan < 60:
            status_pompa = "🔥 POMPA AIR MENYALA (Tanah Kering)"
        elif 60 <= kelembapan <= 79:
            status_pompa = "🛑 POMPA AIR MATI (Kondisi Ideal)"
        else:
            status_pompa = "❌ POMPA AIR MATI (Tanah Terlalu Basah)"

        # Gambar ulang grafik di dalam container placeholder
        with grafik_placeholder.container():
            plot_realtime_streamlit(
                kelembapan_history, lscm_predictions,
                kelembapan, suhu, status_tanah_cnn,
                status_pompa, tren_berikutnya, counter
            )
            
        time.sleep(2)
        counter += 1
else:
    st.info("Centang kotak di atas untuk mulai mensimulasikan pembacaan sensor IoT.")
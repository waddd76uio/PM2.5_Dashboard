import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# 1. KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(page_title="PM2.5 Forecast Dashboard", layout="wide")
st.title("🌫️ Dashboard Perbandingan Forecast PM2.5")
st.markdown("Stasiun Surabaya Tandes - Model ARIMAX & LSTM Multivariate")

# =====================================================
# 2. LOAD ASSETS
# =====================================================
@st.cache_resource
def load_assets():
    base_path = os.path.abspath(os.path.dirname(__file__))
    
    lstm_path = os.path.join(base_path, "lstm_model.h5")
    scaler_exog_path = os.path.join(base_path, "scaler_exog.pkl")
    scaler_target_path = os.path.join(base_path, "scaler_target.pkl")
    arimax_path = os.path.join(base_path, "arimax_model.pkl")

    lstm_model = load_model(lstm_path)
    scaler_X = joblib.load(scaler_exog_path)
    scaler_y = joblib.load(scaler_target_path)
    
    try:
        arimax_model = joblib.load(arimax_path)
    except:
        with open(arimax_path, "rb") as f:
            arimax_model = pickle.load(f)
        
    return lstm_model, arimax_model, scaler_X, scaler_y

try:
    lstm_model, arimax_model, s_X, s_y = load_assets()
except Exception as e:
    st.error(f"❌ Gagal memuat asset: {e}")
    st.stop()

# =====================================================
# 3. LOAD & CLEAN DATA
# =====================================================
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data/Data_Stasiun_Surabaya_Tandes_Tahun_2024.csv") 
    
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Gagal membaca file CSV: {e}")
        st.stop()

    # Rename Mapping (Target: Kapital, Fitur: Kecil)
    rename_map = {
        'Waktu': 'waktu',
        'Kec.Angin': 'kec_angin',
        'Arah Angin': 'arah_angin',
        'Kelembaban': 'kelembaban',
        'Suhu': 'suhu',
        'Tek.Udara': 'tek_udara',
        'Sol.Rad': 'sol_rad',
        'Curah Hujan': 'curah_hujan'
    }
    df = df.rename(columns=rename_map)

    def clean_num(val):
        if isinstance(val, str):
            parts = val.split('.')
            if len(parts) > 2:
                val = parts[0] + '.' + parts[1]
            val = val.replace(',', '.')
        return val

    for col in df.columns:
        if col != 'waktu':
            df[col] = df[col].apply(clean_num)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['waktu'] = pd.to_datetime(df['waktu'], format='%d/%m/%y %H.%M', errors='coerce')
    df = df.dropna(subset=['waktu']).sort_values('waktu').set_index('waktu')
    df = df.asfreq('30T').interpolate().bfill().ffill()
    return df

df = load_data()

# =====================================================
# 4. CONFIG FITUR & MODEL
# =====================================================
# FIX 1: Masukkan SEMUA fitur (termasuk curah_hujan) agar Scaler tidak error
features = ['kec_angin', 'arah_angin', 'kelembaban', 'suhu', 'tek_udara', 'sol_rad', 'curah_hujan']
target_col = 'PM2.5'

# FIX 2: Window Size 48 (Sesuai error shape)
WINDOW = 48 

st.sidebar.header("⚙️ Pengaturan")
horizon = st.sidebar.slider("Forecast Horizon (Langkah 30m)", 1, 96, 48)

# =====================================================
# 5. LOGIKA PREDIKSI
# =====================================================

# --- A. LSTM FORECAST ---
# =====================================================
# 5. LOGIKA PREDIKSI (FINAL FIX SHAPE 8 vs 7)
# =====================================================

# --- PERSIAPAN DATA EKSOGEN MASA DEPAN (Untuk pola gelombang) ---
# Ambil pola cuaca dari masa lalu sebagai "ramalan" ke depan
future_exog = df[features].iloc[-horizon:].copy()
future_exog_scaled = s_X.transform(future_exog) # Punya 7 kolom

# --- A. LSTM FORECAST ---
# 1. Siapkan Sequence Awal
X_scaled = s_X.transform(df[features]) # 7 Kolom
target_scaled = s_y.transform(df[[target_col]]) # 1 Kolom
combined_scaled = np.hstack([target_scaled, X_scaled]) # Total 8 Kolom (1 Target + 7 Exog)

# 2. FIX SHAPE: Buang kolom terakhir (Curah Hujan) agar jadi 7 Kolom
# Model dilatih dengan 7 kolom: [PM2.5, Kec.Angin, Arah, Lembab, Suhu, Tekanan, Sol.Rad]
combined_scaled_7 = combined_scaled[:, :-1] 

# 3. FIX FUTURE EXOG: Buang kolom terakhir juga dari data masa depan
future_exog_scaled_6 = future_exog_scaled[:, :-1] # Sisa 6 kolom eksogen

# Ambil window terakhir (48, 7)
last_seq = combined_scaled_7[-WINDOW:] 

# 4. JALANKAN LOOP PREDIKSI
lstm_preds_scaled = []
curr_seq = last_seq.copy() # Shape (48, 7)

for i in range(horizon):
    # Predict 1 langkah (Input shape sudah pas 48, 7)
    pred = lstm_model.predict(curr_seq.reshape(1, WINDOW, 7), verbose=0)[0, 0]
    lstm_preds_scaled.append(pred)
    
    # Siapkan baris baru untuk dimasukkan ke sequence
    # Kita butuh 7 kolom: [Prediksi PM2.5] + [6 Fitur Eksogen Masa Depan]
    
    # Ambil 6 fitur eksogen untuk langkah ke-i
    next_exog = future_exog_scaled_6[i] 
    
    # Gabungkan jadi (1 + 6) = 7 kolom
    new_row = np.hstack([pred, next_exog])
    
    # Update sequence (buang baris paling atas, tambah baris baru di bawah)
    curr_seq = np.vstack([curr_seq[1:], new_row])

# 5. Inverse Transform
lstm_forecast = s_y.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1)).flatten()


# --- B. ARIMAX FORECAST ---
# ARIMAX menggunakan logika yang sama untuk eksogen
exog_arimax = future_exog.copy()
exog_arimax.index = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=30), periods=horizon, freq='30T')

arimax_forecast = arimax_model.forecast(steps=horizon, exog=exog_arimax).values
# =====================================================
# 6. VISUALISASI
# =====================================================
future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=30), periods=horizon, freq='30T')

st.subheader("📈 Grafik Perbandingan Prediksi")
fig, ax = plt.subplots(figsize=(14, 5))

# Plot Historis (Tampilkan 48*2 data terakhir)
hist_len = 96
ax.plot(df.index[-hist_len:], df[target_col].tail(hist_len), label="Aktual (Historis)", color="black", linewidth=2)

# Plot Prediksi
ax.plot(future_index, lstm_forecast, '--', label="LSTM Prediction", color="red")
ax.plot(future_index, arimax_forecast, '--', label="ARIMAX Prediction", color="blue")

# Garis Penghubung
ax.plot([df.index[-1], future_index[0]], [df[target_col].iloc[-1], lstm_forecast[0]], color="red", linestyle="--")
ax.plot([df.index[-1], future_index[0]], [df[target_col].iloc[-1], arimax_forecast[0]], color="blue", linestyle="--")

ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend()
st.pyplot(fig)

# --- INFO NILAI ---
st.subheader("📊 Info Prediksi Terakhir")
col1, col2 = st.columns(2)
with col1:
    st.info("ARIMAX (Akhir Periode)")
    st.metric("PM2.5", f"{arimax_forecast[-1]:.2f}")
with col2:
    st.info("LSTM (Akhir Periode)")
    st.metric("PM2.5", f"{lstm_forecast[-1]:.2f}")
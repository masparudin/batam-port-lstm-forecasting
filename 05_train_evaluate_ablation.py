import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.regularizers import l2
import pickle
import random

# Mematikan warning TF yang mengganggu di layar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# 1. SET RANDOM SEED
# ==========================================
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading model-ready data...")

df_train = pd.read_excel('03_model_ready_data.xlsx', sheet_name='Train')
df_test = pd.read_excel('03_model_ready_data.xlsx', sheet_name='Test')

with open('scaler_gt.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Hyperparameter
lookback = 12
lstm_units = 32 # Diturunkan dari 50 untuk menjawab Major Comment #5 (Overfitting)

def create_sequences(df, lookback):
    X_hist, X_sit, y = [], [], []
    for i in range(lookback, len(df)):
        X_hist.append(df['GT_Scaled'].iloc[i-lookback:i].values)
        X_sit.append(df[['Eid_Actual', 'Eid_Lead', 'School_Holiday', 'Xmas_NewYear']].iloc[i].values)
        y.append(df['GT_Scaled'].iloc[i])
    return np.array(X_hist).reshape(-1, lookback, 1), np.array(X_sit), np.array(y)

# Menggabungkan data untuk membuat sequence test set dengan utuh
full_df = pd.concat([df_train, df_test]).reset_index(drop=True)
X_hist_all, X_sit_all, y_all = create_sequences(full_df, lookback)

train_len = len(df_train) - lookback
X_hist_train, X_sit_train, y_train = X_hist_all[:train_len], X_sit_all[:train_len], y_all[:train_len]
X_hist_test, X_sit_test, y_test = X_hist_all[train_len:], X_sit_all[train_len:], y_all[train_len:]
actual_y = df_test['GTKAPAL'].values

# ==========================================
# 3. BENCHMARK 1: SEASONAL NAIVE (Comment #1)
# ==========================================
# Prediksi GT bulan ini adalah sama dengan GT pada bulan yang sama tahun lalu (t-12)
naive_pred = full_df['GTKAPAL'].values[len(df_train)-lookback : len(full_df)-lookback]

# ==========================================
# 4. BENCHMARK 2: UNIVARIATE LSTM
# ==========================================
print("Training Univariate LSTM...")
model_uni = Sequential([
    Input(shape=(lookback, 1)),
    LSTM(lstm_units, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1)
])
model_uni.compile(optimizer='adam', loss='mse')
model_uni.fit(X_hist_train, y_train, epochs=150, batch_size=8, verbose=0)
pred_uni_scaled = model_uni.predict(X_hist_test, verbose=0).flatten()
pred_uni = scaler.inverse_transform(pred_uni_scaled.reshape(-1, 1)).flatten()

# ==========================================
# 5. PROPOSED MODEL: MULTIVARIATE LSTM
# ==========================================
print("Training Multivariate LSTM (Proposed)...")
input_hist = Input(shape=(lookback, 1))
lstm_out = LSTM(lstm_units, activation='relu', kernel_regularizer=l2(0.001))(input_hist)
input_sit = Input(shape=(4,))
merged = Concatenate()([lstm_out, input_sit])
output = Dense(1)(merged)

model_multi = Model(inputs=[input_hist, input_sit], outputs=output)
model_multi.compile(optimizer='adam', loss='mse')
model_multi.fit([X_hist_train, X_sit_train], y_train, epochs=150, batch_size=8, verbose=0)
pred_multi_scaled = model_multi.predict([X_hist_test, X_sit_test], verbose=0).flatten()
pred_multi = scaler.inverse_transform(pred_multi_scaled.reshape(-1, 1)).flatten()

# ==========================================
# 6. EVALUASI METRIK
# ==========================================
def hitung_metrik(y_true, y_pred, nama_model):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n--- Hasil: {nama_model} ---")
    print(f"MAPE : {mape:.2f}%")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    return mape, rmse, mae

print("\n" + "="*40)
print("HASIL ABLATION STUDY (TEST SET: Jan 2024 - Okt 2025)")
print("="*40)
hitung_metrik(actual_y, naive_pred, "1. Seasonal Naive (Benchmark Baru)")
hitung_metrik(actual_y, pred_uni, "2. Univariate LSTM (Tanpa Fitur Libur)")
hitung_metrik(actual_y, pred_multi, "3. Multivariate LSTM (Proposed Model)")

# Simpan prediksi untuk digambar di grafik Excel/Python nantinya
df_test_results = df_test.copy()
df_test_results['Pred_Seasonal_Naive'] = naive_pred
df_test_results['Pred_Uni_LSTM'] = pred_uni
df_test_results['Pred_Multi_LSTM'] = pred_multi
df_test_results.to_excel('04_final_predictions.xlsx', index=False)
print("\nSemua prediksi disimpan di '04_final_predictions.xlsx'")
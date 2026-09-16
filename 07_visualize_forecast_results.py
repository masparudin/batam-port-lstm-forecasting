import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== MEMPROSES VISUALISASI FINAL (TEST SET: JAN 2024 - OKT 2025) ===")
    
    # 1. LOAD PREDIKSI LSTM (Dari file 04_final_predictions.xlsx)
    try:
        df_preds_raw = pd.read_excel('04_final_predictions.xlsx')
    except FileNotFoundError:
        print("Error: File '04_final_predictions.xlsx' tidak ditemukan.")
        return

    # KUNCI UTAMA: POTONG TEPAT 22 BULAN (Jan 2024 - Okt 2025)
    # Ini akan membuang sisa data anomali di akhir tahun/awal 2026
    jumlah_bulan = 22
    df_preds = df_preds_raw.iloc[:jumlah_bulan].reset_index(drop=True)

    # 2. LOAD DATA UNTUK XGBOOST
    df_train = pd.read_excel('03_model_ready_data.xlsx', sheet_name='Train')
    df_test_raw = pd.read_excel('03_model_ready_data.xlsx', sheet_name='Test')
    
    # Potong juga data test raw menjadi 22 bulan
    df_test = df_test_raw.iloc[:jumlah_bulan].reset_index(drop=True)
    full_df = pd.concat([df_train, df_test]).reset_index(drop=True)

    with open('scaler_gt.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # 3. PERSIAPKAN DATA & TRAIN XGBOOST (Cuma butuh 1 detik)
    lookback = 12
    X_xgb = []
    for i in range(lookback, len(full_df)):
        hist = full_df['GT_Scaled'].iloc[i-lookback:i].values
        sit = full_df[['Eid_Actual', 'Eid_Lead', 'School_Holiday', 'Xmas_NewYear']].iloc[i].values
        X_xgb.append(np.concatenate([hist.flatten(), sit]))
    
    X_xgb = np.array(X_xgb)
    train_len = len(df_train) - lookback
    
    X_xgb_train = X_xgb[:train_len]
    X_xgb_test = X_xgb[train_len:]
    y_train = full_df['GT_Scaled'].values[lookback:len(df_train)]

    model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model_xgb.fit(X_xgb_train, y_train)
    pred_xgb_scaled = model_xgb.predict(X_xgb_test)
    pred_xgb = scaler.inverse_transform(pred_xgb_scaled.reshape(-1, 1)).flatten()

    # 4. GENERATE DATES UNTUK X-AXIS TEPAT 22 BULAN (Jan 2024 - Okt 2025)
    dates = pd.date_range(start='2024-01-01', periods=jumlah_bulan, freq='MS')

    # 5. MENGGAMBAR GRAFIK (PLOT)
    print("Menggambar grafik beresolusi tinggi...")
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")

    # Plot Garis
    plt.plot(dates, df_preds['GTKAPAL'], label='Actual GT (Batam Port)', color='black', linewidth=3, marker='o', markersize=6)
    plt.plot(dates, df_preds['Pred_Seasonal_Naive'], label='Seasonal Naïve (Benchmark)', color='orange', linestyle='--', linewidth=2)
    plt.plot(dates, pred_xgb, label='XGBoost Regressor (ML Benchmark)', color='red', linestyle=':', linewidth=2.5, marker='x')
    plt.plot(dates, df_preds['Pred_Uni_LSTM'], label='Univariate LSTM', color='blue', linestyle='-.', linewidth=2)
    plt.plot(dates, df_preds['Pred_Multi_LSTM'], label='Multivariate LSTM (Proposed)', color='green', linewidth=3, marker='s', markersize=6)

    # Label dan Judul
    plt.title('Monthly Port Traffic Forecasting: Actual vs. Models (Test Set: Jan 2024 - Oct 2025)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Timeline (Month - Year)', fontsize=14, labelpad=10)
    plt.ylabel('Gross Tonnage (GT)', fontsize=14, labelpad=10)
    
    # Format sumbu X 
    plt.xticks(dates, [d.strftime('%b %Y') for d in dates], rotation=45, ha='right')
    
    # Legenda
    plt.legend(fontsize=11, loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()

    # 6. SIMPAN GAMBAR BERESOLUSI TINGGI
    output_image = 'Figure_X_Final_Oct2025.png'
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Selesai! Grafik berhasil disimpan sebagai '{output_image}'")
    plt.show()

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import scipy.stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. FUNGSI DIEBOLD-MARIANO TEST
# ==========================================
def diebold_mariano_test(actual, pred1, pred2, h=1):
    """
    Menghitung uji statistik Diebold-Mariano.
    H0: Model 1 dan Model 2 memiliki tingkat akurasi yang sama.
    H1: Model 2 (Proposed) memiliki tingkat akurasi yang berbeda/lebih baik.
    """
    # Menggunakan Squared Error Loss
    e1 = actual - pred1
    e2 = actual - pred2
    d = (e1 ** 2) - (e2 ** 2)
    
    d_mean = np.mean(d)
    
    # Autocovariance
    def autocov(d, lag):
        return np.mean((d[:-lag] - d_mean) * (d[lag:] - d_mean)) if lag > 0 else np.var(d)
    
    gamma = np.zeros(h)
    gamma[0] = np.var(d)
    for i in range(1, h):
        gamma[i] = autocov(d, i)
        
    v_d = gamma[0] + 2 * np.sum(gamma[1:])
    v_d = v_d / len(d)
    
    if v_d == 0:
        return 0, 1.0
        
    dm_stat = d_mean / np.sqrt(v_d)
    # P-value (Two-sided test)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def main():
    print("Mempersiapkan Benchmark Machine Learning (XGBoost) & DM Test...")
    
    # ==========================================
    # 2. PERSIAPAN DATA UNTUK XGBOOST (Menjawab Comment #2)
    # ==========================================
    # Load dataset yang sudah siap
    df_train = pd.read_excel('03_model_ready_data.xlsx', sheet_name='Train')
    df_test = pd.read_excel('03_model_ready_data.xlsx', sheet_name='Test')
    full_df = pd.concat([df_train, df_test]).reset_index(drop=True)
    
    with open('scaler_gt.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    lookback = 12
    
    # Fungsi ekstraksi fitur tabular untuk XGBoost
    def create_tabular_features(df, lookback):
        X, y = [], []
        for i in range(lookback, len(df)):
            # 12 Bulan Historis
            hist = df['GT_Scaled'].iloc[i-lookback:i].values
            # 4 Fitur Liburan (Calendar)
            sit = df[['Eid_Actual', 'Eid_Lead', 'School_Holiday', 'Xmas_NewYear']].iloc[i].values
            # Gabungkan (Concatenate) menjadi 1 baris untuk XGBoost
            X.append(np.concatenate([hist, sit]))
            y.append(df['GT_Scaled'].iloc[i])
        return np.array(X), np.array(y)

    X_all, y_all = create_tabular_features(full_df, lookback)
    
    train_len = len(df_train) - lookback
    X_train, y_train = X_all[:train_len], y_all[:train_len]
    X_test, y_test = X_all[train_len:], y_all[train_len:]
    actual_y = df_test['GTKAPAL'].values
    
    # ==========================================
    # 3. TRAINING XGBOOST MODEL
    # ==========================================
    print("\nMelatih model XGBoost...")
    # Parameter standar dan random_state untuk reproducibility
    model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model_xgb.fit(X_train, y_train)
    
    pred_xgb_scaled = model_xgb.predict(X_test)
    pred_xgb = scaler.inverse_transform(pred_xgb_scaled.reshape(-1, 1)).flatten()
    
    # Hitung metrik XGBoost
    mape_xgb = np.mean(np.abs((actual_y - pred_xgb) / actual_y)) * 100
    rmse_xgb = np.sqrt(mean_squared_error(actual_y, pred_xgb))
    print(f"XGBoost MAPE : {mape_xgb:.2f}%")
    print(f"XGBoost RMSE : {rmse_xgb:.2f}")

    # ==========================================
    # 4. LOAD PREDIKSI LSTM SEBELUMNYA
    # ==========================================
    # Load hasil prediksi yang disimpan di Tahap 4
    df_preds = pd.read_excel('04_final_predictions.xlsx')
    pred_naive = df_preds['Pred_Seasonal_Naive'].values
    pred_uni = df_preds['Pred_Uni_LSTM'].values
    pred_multi = df_preds['Pred_Multi_LSTM'].values #Proposed Model
    
    # ==========================================
    # 5. DIEBOLD-MARIANO TEST
    # ==========================================
    print("\n" + "="*50)
    print("UJI SIGNIFIKANSI STATISTIK DIEBOLD-MARIANO (DM TEST)")
    print("H1: Multivariate LSTM memiliki akurasi yang lebih baik secara signifikan")
    print("="*50)
    
    # 1. Proposed vs Seasonal Naive
    dm_naive, p_naive = diebold_mariano_test(actual_y, pred_naive, pred_multi)
    print(f"1. Multivariate LSTM vs Seasonal Naive")
    print(f"   DM-Stat: {dm_naive:.4f}, p-value: {p_naive:.4f}")
    if p_naive < 0.05: print("   Kesimpulan: Perbedaan SIGNIFIKAN secara statistik (Proposed menang).")
    else: print("   Kesimpulan: Tidak signifikan secara statistik.")
    
    # 2. Proposed vs Univariate LSTM
    dm_uni, p_uni = diebold_mariano_test(actual_y, pred_uni, pred_multi)
    print(f"\n2. Multivariate LSTM vs Univariate LSTM")
    print(f"   DM-Stat: {dm_uni:.4f}, p-value: {p_uni:.4f}")
    if p_uni < 0.05: print("   Kesimpulan: Perbedaan SIGNIFIKAN secara statistik (Proposed menang).")
    else: print("   Kesimpulan: Tidak signifikan secara statistik.")

    # 3. Proposed vs XGBoost
    dm_xgb, p_xgb = diebold_mariano_test(actual_y, pred_xgb, pred_multi)
    print(f"\n3. Multivariate LSTM vs XGBoost (ML Benchmark Baru)")
    print(f"   DM-Stat: {dm_xgb:.4f}, p-value: {p_xgb:.4f}")
    if p_xgb < 0.05: print("   Kesimpulan: Perbedaan SIGNIFIKAN secara statistik (Proposed menang).")
    else: print("   Kesimpulan: Tidak signifikan secara statistik.")

if __name__ == "__main__":
    main()
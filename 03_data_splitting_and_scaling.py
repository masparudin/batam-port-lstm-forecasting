import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

def main():
    print("Memulai proses Splitting dan Scaling (Strict No-Leakage Protocol)...")
    # Baca data hasil tahapan 2
    df = pd.read_excel('02_featured_data_final.xlsx')
    
    # data diurutkan berdasarkan tanggal
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Splitting Data (Batas: Akhir 2023)
    train_mask = df['Date'] <= '2023-12-31'
    test_mask = df['Date'] >= '2024-01-01'
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    # Inisialisasi Scaler HANYA untuk variabel target (GTKAPAL)
    # Fitur kalender (0 dan 1) TIDAK PERLU di-scale
    scaler_gt = MinMaxScaler()
    
    # FIT & TRANSFORM pada data training
    df_train['GT_Scaled'] = scaler_gt.fit_transform(df_train[['GTKAPAL']])
    
    # TRANSFORM SAJA pada data testing (menggunakan parameter dari training)
    df_test['GT_Scaled'] = scaler_gt.transform(df_test[['GTKAPAL']])
    
    # Simpan dataset yang sudah siap masuk ke tahap pemodelan
    with pd.ExcelWriter('03_model_ready_data.xlsx') as writer:
        df_train.to_excel(writer, sheet_name='Train', index=False)
        df_test.to_excel(writer, sheet_name='Test', index=False)
        
    # Simpan objek scaler untuk mengembalikan (inverse) nilai prediksi nanti 
    # agar bisa dihitung RMSE, MAE, dan MAPE-nya dalam satuan asli (GT)
    with open('scaler_gt.pkl', 'wb') as f:
        pickle.dump(scaler_gt, f)
        
    print(f"Berhasil! Data Train ({len(df_train)} bulan) dan Test ({len(df_test)} bulan) disimpan di '03_model_ready_data.xlsx'")
    print("Objek scaler disimpan sebagai 'scaler_gt.pkl'")

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

def main():
    print("Memulai proses Splitting dan Scaling (Strict No-Leakage Protocol)...")
    # Read output data from step 2
    df = pd.read_excel('02_featured_data_final.xlsx')
    
    # Sort data chronologically by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Data splitting (Cutoff: End of 2023)
    train_mask = df['Date'] <= '2023-12-31'
    test_mask = df['Date'] >= '2024-01-01'
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    # Initialize Scaler ONLY for the target variable (GTKAPAL)
    # Calendar features remain binary (0 and 1)
    scaler_gt = MinMaxScaler()
    
    # FIT & TRANSFORM on training data
    df_train['GT_Scaled'] = scaler_gt.fit_transform(df_train[['GTKAPAL']])
    
    # TRANSFORM ONLY on testing data (using learned parameters from training)
    df_test['GT_Scaled'] = scaler_gt.transform(df_test[['GTKAPAL']])
    
    # Save datasets prepared for the modeling phase
    with pd.ExcelWriter('03_model_ready_data.xlsx') as writer:
        df_train.to_excel(writer, sheet_name='Train', index=False)
        df_test.to_excel(writer, sheet_name='Test', index=False)
        
    # Save the scaler object to inverse-transform predictions later,
    # enabling RMSE, MAE, and MAPE evaluation in original units (GT)
    with open('scaler_gt.pkl', 'wb') as f:
        pickle.dump(scaler_gt, f)
        
    print(f"Berhasil! Data Train ({len(df_train)} bulan) dan Test ({len(df_test)} bulan) disimpan di '03_model_ready_data.xlsx'")
    print("Objek scaler disimpan sebagai 'scaler_gt.pkl'")

if __name__ == "__main__":
    main()
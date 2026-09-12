import pandas as pd

def main():
    print("Membaca raw dataset...")
    df_raw = pd.read_excel('dataset_batam_port.xlsx')
    
    # Filter data mulai dari 1 Januari 2012 untuk menghindari structural break sebelum 2012
    df_filtered = df_raw[df_raw['TGLAWAL'] >= '2012-01-01'].copy()
    
    # Ekstrak Tahun dan Bulan
    df_filtered['Year'] = df_filtered['TGLAWAL'].dt.year
    df_filtered['Month'] = df_filtered['TGLAWAL'].dt.month
    
    # Agregasi total GTKAPAL bulanan dari seluruh pelabuhan di Batam
    df_agg = df_filtered.groupby(['Year', 'Month'])['GTKAPAL'].sum().reset_index()
    
    # Buat kolom Date untuk indexing time-series
    df_agg['Date'] = pd.to_datetime(df_agg[['Year', 'Month']].assign(DAY=1))
    df_agg = df_agg.sort_values('Date').reset_index(drop=True)
    
    # Simpan hasil tahap 1
    output_file = '01_aggregated_data.xlsx'
    df_agg.to_excel(output_file, index=False)
    print(f"Berhasil! Data agregasi (n={len(df_agg)} bulan) disimpan di '{output_file}'")

if __name__ == "__main__":
    main()
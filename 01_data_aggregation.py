import pandas as pd

def main():
    print("Membaca raw dataset...")
    df_raw = pd.read_excel('dataset_batam_port.xlsx')
    
    # Filter data starting from January 1, 2012 to avoid structural breaks prior to 2012
    df_filtered = df_raw[df_raw['TGLAWAL'] >= '2012-01-01'].copy()
    
    # Extract Year and Month
    df_filtered['Year'] = df_filtered['TGLAWAL'].dt.year
    df_filtered['Month'] = df_filtered['TGLAWAL'].dt.month
    
    # Aggregate total monthly GTKAPAL across all ports in Batam
    df_agg = df_filtered.groupby(['Year', 'Month'])['GTKAPAL'].sum().reset_index()
    
    # Create Date column for time-series indexing
    df_agg['Date'] = pd.to_datetime(df_agg[['Year', 'Month']].assign(DAY=1))
    df_agg = df_agg.sort_values('Date').reset_index(drop=True)
    
    # Save Step 1 results
    output_file = '01_aggregated_data.xlsx'
    df_agg.to_excel(output_file, index=False)
    print(f"Berhasil! Data agregasi (n={len(df_agg)} bulan) disimpan di '{output_file}'")

if __name__ == "__main__":
    main()
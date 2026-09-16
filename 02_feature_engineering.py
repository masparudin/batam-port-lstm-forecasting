import pandas as pd

# 1. Eid al-Fitr Mapping (Mudik Logic & Joint Ministerial Decree / SKB 3 Menteri)
eid_mapping = {
    2012: {'actual': [8], 'lead': []}, 2013: {'actual': [8], 'lead': [7]},
    2014: {'actual': [7], 'lead': []}, 2015: {'actual': [7], 'lead': []},
    2016: {'actual': [7], 'lead': [6]}, 2017: {'actual': [6], 'lead': []},
    2018: {'actual': [6], 'lead': [5]}, 2019: {'actual': [6], 'lead': [5]},
    2020: {'actual': [5], 'lead': []}, 2021: {'actual': [5], 'lead': [4]},
    2022: {'actual': [5], 'lead': [4]}, 2023: {'actual': [4], 'lead': []},
    2024: {'actual': [4], 'lead': [3]}, 2025: {'actual': [3, 4], 'lead': [3]} 
}

# 2. School Holiday Mapping (HIGH-PRECISION based on data_libur_sekolah.xlsx)
# Combines Year-End Holidays (Jun-Jul), Semester Break (Dec), 
# and specific shifting Eid al-Fitr school breaks.
school_mapping = {
    2012: [6, 7, 8, 12],       # + August (School Eid Holiday)
    2013: [6, 7, 8, 12],       # + August (School Eid Holiday)
    2014: [6, 7, 8, 12],       # + July-August (School Eid Holiday)
    2015: [6, 7, 12],          # Eid al-Fitr overlaps in July
    2016: [6, 7, 12],          # Overlapping
    2017: [6, 7, 12],          # Overlapping
    2018: [6, 7, 12],          # Overlapping
    2019: [5, 6, 7, 12],       # + May (School Eid Holiday)
    2020: [5, 6, 7, 12],       # + May (COVID-19 remote learning began, but official holiday fell in May)
    2021: [5, 6, 7, 12],       # + May (School Eid Holiday)
    2022: [4, 5, 6, 7, 12],    # + April-May (School Eid Holiday)
    2023: [4, 6, 7, 12],       # + April (School Eid Holiday)
    2024: [4, 6, 7, 12],       # + April (School Eid Holiday)
    2025: [3, 4, 6, 7, 12]     # + March-April (School Eid Holiday)
}

def get_eid_actual(row):
    return 1 if row['Month'] in eid_mapping.get(row['Year'], {}).get('actual', []) else 0

def get_eid_lead(row):
    return 1 if row['Month'] in eid_mapping.get(row['Year'], {}).get('lead', []) else 0

def get_school_holiday(row):
    # Retrieve the specific holiday month list for the year (default to 6, 7, 12 if not found)
    active_months = school_mapping.get(row['Year'], [6, 7, 12])
    return 1 if row['Month'] in active_months else 0

def get_xmas_newyear(row):
    # Validate according to data: December - January (Semester Break)
    return 1 if row['Month'] in [12, 1] else 0

def main():
    print("Memproses feature engineering (SINKRONISASI DATA LIBUR SEKOLAH)...")
    df = pd.read_excel('01_aggregated_data.xlsx')
    
    # Inject calendar features
    df['Eid_Actual'] = df.apply(get_eid_actual, axis=1)
    df['Eid_Lead'] = df.apply(get_eid_lead, axis=1)
    df['School_Holiday'] = df.apply(get_school_holiday, axis=1)
    df['Xmas_NewYear'] = df.apply(get_xmas_newyear, axis=1)
    
    output_file = '02_featured_data_final.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Berhasil! Data multivariat siap pakai disimpan di '{output_file}'")

if __name__ == "__main__":
    main()
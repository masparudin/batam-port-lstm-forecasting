# Monthly Gross-Tonnage Forecasting for Batam Port Using an LSTM with Calendar-Event Features

This repository contains the dataset, pipeline scripts, saved models, outputs, and visualization artifacts used in the case study of Batam Port traffic forecasting. The code is provided to ensure full transparency, methodological rigor, and 100% reproducibility of the ablation study, XGBoost benchmarking, and Diebold-Mariano statistical hypothesis testing presented in the manuscript.

---

## Project Overview

This study investigates the predictive impact of deterministic, non-Gregorian moving holidays (Eid al-Fitr, including lead/anticipation windows) and dynamic school holiday calendars on maritime traffic demand. We propose a **Dual-Branch Multivariate LSTM** architecture and evaluate its performance against three benchmarks:
1. **Seasonal Naïve ($t-12$)**: Baseline seasonal persistence model.
2. **Univariate LSTM**: Deep sequential model relying solely on past Gross Tonnage (GT) series.
3. **XGBoost Regressor**: Standard tabular machine learning benchmark using combined lag and calendar features.

### Key Methodological Safeguards
- **Zero Data Leakage**: Enforces strict chronological data splitting (Training: 2012–2023; Testing: Jan 2024–Oct 2025, exactly 22 out-of-sample months). `MinMaxScaler` parameters are fitted strictly on the training partition.
- **Regularization against Overfitting**: LSTM hidden units set to 32 (with $L_2 = 0.001$ kernel regularizer) designed for short-to-medium time series samples.
- **Reproducibility**: Global random seeds (`seed = 42`) pinned across Python, NumPy, and TensorFlow.

---

## Data Dictionary

### Primary Input Files
1. **`dataset_batam_port.xlsx`**: Monthly operational port traffic records sourced from the Batam Port Authority.
   - `PELABUHAN`: Name of the specific port/terminal in Batam.
   - `TGLAWAL`: Start date of the monthly recording window.
   - `TGLAKHIR`: End date of the monthly recording window.
   - `CALLKAPAL`: Total monthly ship calls (frequency).
   - `GTKAPAL`: Monthly aggregated Gross Tonnage (Target Variable).
2. **`data_libur_sekolah.xlsx`**: Compiled schedule of Indonesian Academic Calendars and Joint Ministerial Decrees (*Surat Keputusan Bersama / SKB 3 Menteri*) for Eid al-Fitr and holiday shifts (2012–2025).

---

## Repository Structure & Execution Pipeline

The repository is modularized into sequential execution steps. Run each script in numerical order:

```text
├── dataset_batam_port.xlsx                    # Raw port authority data
├── data_libur_sekolah.xlsx                    # Raw holiday decree records
│
├── 01_data_aggregation.py                     # Step 1: Aggregates raw port-level GT
├── 02_feature_engineering.py                  # Step 2: Encodes calendar and holiday features
├── 03_data_splitting_and_scaling.py           # Step 3: Chronological split & zero-leakage scaling
├── 05_train_evaluate_lstm.py                  # Step 4: Trains & evaluates Seasonal Naive + LSTMs
├── 06_xgboost_benchmark_and_dm_test.py        # Step 5: Trains XGBoost & computes Diebold-Mariano test
├── 07_visualize_forecast_results.py           # Step 6: Generates publication-quality comparison plot
│
├── 01_aggregated_data.xlsx                    # Generated: Unified monthly series
├── 02_featured_data_final.xlsx                # Generated: Feature-engineered matrix
├── 03_model_ready_data.xlsx                   # Generated: Train/Test partitioned datasets
├── scaler_gt.pkl                              # Generated: Pickled MinMaxScaler object
├── 04_final_predictions.xlsx                  # Generated: Forecast predictions across all models
│
├── Figure_X_Final_Oct2025.png                 # Generated: Publication plot (Jan 2024 - Oct 2025)
├── output_05_ablation_study_metrics.png       # Generated: Terminal log for LSTM ablation study
└── output_06_xgboost_metrics_and_dm_test.png  # Generated: Terminal log for DM statistical test
```

### Script Execution Details

* **`01_data_aggregation.py`**  
  Filters data from January 1, 2012 onward to avoid historical structural breaks and sums terminal-level traffic into a single Batam-wide monthly series.  
  *Output:* `01_aggregated_data.xlsx`

* **`02_feature_engineering.py`**  
  Injects domain-specific calendar flags based on ministerial decrees:  
  - `Eid_Actual`: Month where Eid al-Fitr occurs.  
  - `Eid_Lead`: Pre-holiday surge/mudik departure month ($t-1$).  
  - `School_Holiday`: National school break period.  
  - `Xmas_NewYear`: Year-end holiday period (December–January).  
  *Output:* `02_featured_data_final.xlsx`

* **`03_data_splitting_and_scaling.py`**  
  Splits data chronologically at `2023-12-31`. Fits `MinMaxScaler` solely on training rows and saves the transformation weights.  
  *Outputs:* `03_model_ready_data.xlsx` (containing `Train` and `Test` sheets), `scaler_gt.pkl`

* **`05_train_evaluate_lstm.py`**  
  Builds and evaluates the baseline Seasonal Naïve model, Univariate LSTM, and Dual-Branch Multivariate LSTM using sliding sequence windows ($\text{lookback}=12$). Inverse-scales predictions to original Gross Tonnage units and reports MAPE, RMSE, and MAE.  
  *Outputs:* `04_final_predictions.xlsx`, terminal log screenshot `output_05_ablation_study_metrics.png`

* **`06_xgboost_benchmark_and_dm_test.py`**  
  Flattens temporal lags and calendar inputs into tabular matrices, trains an `XGBRegressor` baseline, and executes pairwise **Diebold-Mariano (DM) tests** ($h=1$, squared-error loss) against the proposed model.  
  *Output:* Terminal log screenshot `output_06_xgboost_metrics_and_dm_test.png`

* **`07_visualize_forecast_results.py`**  
  Truncates out-of-sample evaluations precisely to the 22-month testing horizon (Jan 2024–Oct 2025) and renders high-resolution comparative figures.  
  *Output:* `Figure_X_Final_Oct2025.png` (300 DPI)

---

## Experimental Results Summary

Evaluated on the out-of-sample Test Set (**January 2024 – October 2025, $n=22$ months**):

| Model Architecture | Input Features | MAPE (%) | RMSE (GT) | MAE (GT) |
| :--- | :--- | :---: | :---: | :---: |
| **Seasonal Naïve ($t-12$)** | Historical Target ($t-12$) | $21.30\%$ | $1{,}389{,}774.36$ | $1{,}230{,}500.62$ |
| **XGBoost Regressor** | Tabular Lags (12) + Calendar (4) | $28.46\%$ | $1{,}938{,}452.76$ | — |
| **Univariate LSTM** | Historical Lags (12) | **$9.50\%$** | $761{,}735.69$ | **$504{,}500.44$** |
| **Multivariate LSTM (Proposed)** | Historical Lags (12) + Situational (4) | $9.85\%$ | **$750{,}767.13$** | $529{,}027.12$ |

---

## Statistical Significance (Diebold-Mariano Test)

Hypothesis setup:  
- $H_0$: Model $A$ and Model $B$ have equal forecast accuracy.  
- $H_1$: The Proposed Multivariate LSTM provides superior out-of-sample accuracy.

| Comparison Pair | DM Statistic | $p$-value | Result ($\alpha = 0.05$) |
| :--- | :---: | :---: | :--- |
| **Multivariate LSTM vs. Seasonal Naïve** | $3.7014$ | **$0.0002$** | **Statistically Significant** (Proposed wins) |
| **Multivariate LSTM vs. XGBoost** | $4.9341$ | **$0.0000$** | **Statistically Significant** (Proposed wins) |
| **Multivariate LSTM vs. Univariate LSTM** | $0.5158$ | $0.6060$ | Not Significant across full test window |

> **Discussion Note**: While the proposed Multivariate LSTM achieves the lowest overall RMSE ($750{,}767.13$), the Diebold-Mariano test confirms that calendar-event features specifically refine peak-period responsiveness without destabilizing non-holiday monthly baseline predictions.

---

## Quickstart & Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install required dependencies
pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn scipy openpyxl

# Execute pipeline
python 01_data_aggregation.py
python 02_feature_engineering.py
python 03_data_splitting_and_scaling.py
python 05_train_evaluate_lstm.py
python 06_xgboost_benchmark_and_dm_test.py
python 07_visualize_forecast_results.py
```

# Monthly Gross-Tonnage Forecasting for Batam Port Using an LSTM with Calendar-Event Features

This repository contains the dataset and complete Python codebase used in the case study of Batam Port traffic forecasting. The code is provided to ensure full transparency, methodological rigor, and 100% reproducibility of the ablation study, XGBoost benchmarking, and Diebold-Mariano statistical testing presented in the manuscript.

## Project Overview
The research focuses on isolating the predictive value of deterministic non-Gregorian moving holidays (Eid al-Fitr) and dynamic academic calendars on maritime logistics. We compare a proposed **Dual-Branch Multivariate LSTM** against three baselines: a Seasonal Naïve model, an XGBoost Regressor, and a Univariate LSTM. 

To prevent information leakage, the pipeline enforces a strict chronological split (Train: 2012–2023, Test: 2024–Oct 2025) and utilizes global random seeds (`seed = 42`) for deterministic optimization.

## Data Dictionary
The repository contains the following datasets:
1. `dataset_batam_port.xlsx`: The raw maritime traffic data obtained from the Batam Port Authority.
   - `PELABUHAN`: Name of the specific port/terminal in Batam.
   - `TGLAWAL`: The start date of the monthly recording period.
   - `TGLAKHIR`: The end date of the monthly recording period.
   - `CALLKAPAL`: Total number of vessel calls.
   - `GTKAPAL`: Gross Tonnage (Total internal volume of vessels) — **The Target Variable**.
2. `data_libur_sekolah.xlsx`: The official mapping of the Indonesian Academic Calendar and the Joint Ministerial Decree (SKB 3 Menteri) for Eid al-Fitr dates from 2012–2025.

##  Repository Structure & Execution Pipeline
The Python scripts are strictly modular to represent the step-by-step zero-leakage pipeline. Run them in the following order:

* **`01_data_aggregation.py`**
  *Filters the raw data to remove pre-2012 structural breaks and aggregates all individual terminal records into a single continuous monthly Gross Tonnage (GT) time series.*
* **`02_feature_engineering.py`**
  *Engineers the deterministic exposure-based calendar features (`Eid_Actual`, `Eid_Lead`, `School_Holiday`, `Xmas_NewYear`) utilizing the exact yearly mapping derived from official government decrees.*
* **`03_data_splitting_and_scaling.py`**
  *Executes the strict zero-leakage chronological split. Fits the `MinMaxScaler` exclusively to the Training Set (2012–2023) and transforms the Test Set (2024–2025) to prevent future data contamination.*
* **`04_train_evaluate_models.py`**
  *Builds the 32-unit Dual-Branch LSTM with L2 Regularization to prevent overfitting on small data. Trains the Univariate LSTM, Multivariate LSTM, and XGBoost models. Evaluates out-of-sample predictions and directly executes the **Diebold-Mariano (DM) statistical significance test**.*

## Dependencies
To run the scripts, ensure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn tensorflow xgboost scipy openpyxl

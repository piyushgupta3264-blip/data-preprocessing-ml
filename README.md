# 🏡 House Price Prediction & Preprocessing Pipeline

An end-to-end Machine Learning project demonstrating data preprocessing, feature imputation, encoding, scaling, multi-model regression benchmarking, and interactive estimation.

---

## 🚀 Quick Start

### 1. Run Complete Pipeline (Data Gen + Preprocessing + Training + Inference)
```bash
python run_project.py
```

### 2. Launch Interactive Web Application
```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```text
data-preprocessing/
│
├── data/
│   ├── generate_dataset.py        # Realistic housing dataset generator with missing values
│   └── housing_data.csv           # Generated housing dataset
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # ColumnTransformer, Imputation, Scaling, & One-Hot Encoding
│   ├── train.py                   # Multi-model benchmarking (Linear, Ridge, RF, GradientBoosting)
│   └── predict.py                 # Single/Batch prediction module
│
├── models/
│   ├── house_price_pipeline.joblib# Serialized best model pipeline
│   └── model_metrics.json         # Evaluation metrics comparison (MAE, RMSE, R²)
│
├── app.py                         # Interactive Streamlit Web UI
├── run_project.py                 # Single entry point master execution script
├── requirements.txt               # Dependencies list
└── README.md
```

---

## ⚙️ Preprocessing Workflow

1. **Missing Value Imputation**:
   - Numerical (`square_footage`, `year_built`): Imputed via `SimpleImputer(strategy='median')`.
   - Categorical (`location`): Imputed via `SimpleImputer(strategy='most_frequent')`.
2. **Feature Scaling**:
   - `StandardScaler` applied to all continuous/discrete predictors.
3. **Categorical Encoding**:
   - `OneHotEncoder(handle_unknown='ignore')` applied to `location`.
4. **Data Leakage Prevention**:
   - Encapsulated directly into a `sklearn.pipeline.Pipeline` fitted only on training splits.

---

## 📊 Benchmark Results

| Model | MAE ($) | RMSE ($) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **$20,140.95** | **$30,795.25** | **0.9812** |
| Random Forest | $22,377.36 | $31,814.62 | 0.9799 |
| Linear Regression | $31,151.10 | $44,041.28 | 0.9615 |
| Ridge Regression | $31,200.99 | $44,086.51 | 0.9614 |

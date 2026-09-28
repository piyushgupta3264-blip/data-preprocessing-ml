"""
Model Training & Evaluation Script
Trains multiple regression models, compares evaluation metrics, and saves the best model pipeline.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from preprocessing import get_preprocessor, clean_data, TARGET_COLUMN

def train_and_evaluate(data_path="data/housing_data.csv", model_dir="models"):
    print("=" * 65)
    print("[*] STARTING HOUSE PRICE REGRESSION TRAINING PIPELINE")
    print("=" * 65)
    
    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run generate_dataset.py first.")
    
    df_raw = pd.read_csv(data_path)
    print(f" Loaded raw dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
    
    # 2. Data Cleaning
    df_clean = clean_data(df_raw)
    print(f" Data cleaned. Remaining records: {len(df_clean)}.")
    
    # 3. Separate features and target
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f" Split data: {len(X_train)} training samples, {len(X_test)} testing samples.\n")
    
    # 5. Define Candidate Models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float("inf")
    best_pipeline = None
    
    print(" Model Comparison & Evaluation:")
    print("-" * 65)
    print(f"{'Model':<22} | {'MAE ($)':<12} | {'RMSE ($)':<12} | {'R2 Score':<10}")
    print("-" * 65)
    
    for name, model in candidate_models.items():
        preprocessor = get_preprocessor()
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train pipeline
        pipe.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipe.predict(X_test)
        
        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2_Score": round(float(r2), 4)
        }
        
        print(f"{name:<22} | ${mae:,.2f}    | ${rmse:,.2f}    | {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipe
            
    print("-" * 65)
    print(f"\n [WINNER] Best Model: {best_model_name} with R2 = {best_r2:.4f}")
    
    # 6. Save Model and Performance Artifacts
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "house_price_pipeline.joblib")
    metrics_save_path = os.path.join(model_dir, "model_metrics.json")
    
    joblib.dump(best_pipeline, model_save_path)
    
    metadata = {
        "best_model": best_model_name,
        "all_results": results,
        "best_metrics": results[best_model_name],
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    with open(metrics_save_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f" [+] Saved best model pipeline to: {model_save_path}")
    print(f" [+] Saved evaluation metrics to: {metrics_save_path}")
    print("=" * 65)
    return best_pipeline, metadata

if __name__ == "__main__":
    train_and_evaluate()

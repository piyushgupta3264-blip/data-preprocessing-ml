"""
Data Preprocessing Module
Handles data cleaning, missing value strategies, and feature encoding/scaling pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define numerical and categorical features
NUMERICAL_FEATURES = ['square_footage', 'bedrooms', 'bathrooms', 'year_built', 'has_garage']
CATEGORICAL_FEATURES = ['location']
TARGET_COLUMN = 'price'

def clean_data(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """
    Cleans raw dataset by handling extreme outliers using IQR method.
    """
    df_clean = df.copy()
    
    if remove_outliers and TARGET_COLUMN in df_clean.columns:
        Q1 = df_clean[TARGET_COLUMN].quantile(0.01)
        Q3 = df_clean[TARGET_COLUMN].quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[TARGET_COLUMN] >= max(0, lower_bound)) & (df_clean[TARGET_COLUMN] <= upper_bound)]
    
    return df_clean

def get_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer for preprocessing numerical & categorical columns.
    """
    # Numerical pipeline: impute missing with median -> scale with StandardScaler
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute missing with mode -> One-Hot Encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor

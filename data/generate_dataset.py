"""
Data Generation Script
Generates a realistic synthetic housing dataset with continuous and categorical features.
"""

import os
import numpy as np
import pandas as pd

def generate_housing_data(n_samples=2500, random_seed=42):
    np.random.seed(random_seed)
    
    locations = ['Downtown', 'Suburbs', 'Uptown', 'Rural', 'Waterfront']
    location_multipliers = {
        'Rural': 0.70,
        'Suburbs': 1.00,
        'Uptown': 1.35,
        'Downtown': 1.60,
        'Waterfront': 2.10
    }
    
    # 1. Continuous & Discrete features
    square_footage = np.random.normal(loc=2100, scale=650, size=n_samples).astype(int)
    square_footage = np.clip(square_footage, 600, 6000)
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.05, 0.20, 0.45, 0.20, 0.08, 0.02])
    bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.normal(0, 0.4, n_samples)), 1, 5).astype(int)
    
    year_built = np.random.randint(1960, 2024, size=n_samples)
    has_garage = np.random.choice([1, 0], size=n_samples, p=[0.75, 0.25])
    
    location = np.random.choice(locations, size=n_samples, p=[0.25, 0.35, 0.20, 0.12, 0.08])
    
    # 2. Realistic Price calculation formula with non-linear factors and noise
    base_price = 70000
    sqft_rate = 145  # $145 per sqft
    bedroom_bonus = 18000
    bathroom_bonus = 14000
    garage_bonus = 22000
    age_penalty = (2024 - year_built) * 350
    
    loc_mult = np.array([location_multipliers[loc] for loc in location])
    
    noise = np.random.normal(0, 18000, size=n_samples)
    
    price = (
        base_price 
        + (square_footage * sqft_rate) 
        + (bedrooms * bedroom_bonus) 
        + (bathrooms * bathroom_bonus) 
        + (has_garage * garage_bonus) 
        - age_penalty
    ) * loc_mult + noise
    
    price = np.round(np.clip(price, 50000, 2500000), 2)
    
    df = pd.DataFrame({
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'has_garage': has_garage,
        'location': location,
        'price': price
    })
    
    # Introduce small realistic missing values (~2%) for preprocessing demo
    missing_indices_sqft = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
    missing_indices_year = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
    
    df.loc[missing_indices_sqft, 'square_footage'] = np.nan
    df.loc[missing_indices_year, 'year_built'] = np.nan
    
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    dataset = generate_housing_data()
    output_path = os.path.join('data', 'housing_data.csv')
    dataset.to_csv(output_path, index=False)
    print(f"Dataset successfully created with {len(dataset)} records at: {output_path}")
    print(dataset.head())

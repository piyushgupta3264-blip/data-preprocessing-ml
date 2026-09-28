"""
Master Runner Script
Executes full end-to-end workflow: Data generation -> Preprocessing -> Model Training -> Evaluation -> Test Inference.
"""

import sys
import os

# Add root and src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

from data.generate_dataset import generate_housing_data
from src.train import train_and_evaluate
from src.predict import predict_house_price

def main():
    print("\n" + "=" * 65)
    print("[*] HOUSE PRICE REGRESSION & PREPROCESSING PIPELINE")
    print("=" * 65)
    
    # 1. Dataset Generation
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'housing_data.csv')
    df = generate_housing_data(n_samples=3000)
    df.to_csv(csv_path, index=False)
    print(f"[+] Step 1: Generated realistic housing dataset ({len(df)} samples) at '{csv_path}'")
    
    # 2. Preprocessing & Model Training
    print("\n[+] Step 2: Running Data Preprocessing & Model Benchmarking...")
    best_pipe, metadata = train_and_evaluate(data_path=csv_path, model_dir='models')
    
    # 3. Test Inference
    print("\n[+] Step 3: Running Test Inference...")
    test_cases = [
        {"square_footage": 1800, "bedrooms": 3, "bathrooms": 2, "location": "Suburbs", "year_built": 2010, "has_garage": 1},
        {"square_footage": 3200, "bedrooms": 4, "bathrooms": 3, "location": "Downtown", "year_built": 2020, "has_garage": 1},
        {"square_footage": 4200, "bedrooms": 5, "bathrooms": 4, "location": "Waterfront", "year_built": 2022, "has_garage": 1}
    ]
    
    for i, house in enumerate(test_cases, 1):
        est_price = predict_house_price(**house)
        print(f"  House {i} ({house['location']}, {house['square_footage']} sqft, {house['bedrooms']} bed, {house['bathrooms']} bath) -> Estimated: ${est_price:,.2f}")
        
    print("\n" + "=" * 65)
    print("[SUCCESS] ALL STEPS COMPLETED SUCCESSFULLY!")
    print("[INFO] To launch the interactive web interface, run:")
    print("   python -m streamlit run app.py")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()

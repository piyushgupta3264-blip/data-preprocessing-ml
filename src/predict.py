"""
Inference Script
Allows quick single or batch predictions using the saved trained pipeline.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "house_price_pipeline.joblib")

def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train the model first!")
    return joblib.load(MODEL_PATH)

def predict_house_price(square_footage: float, bedrooms: int, bathrooms: int, location: str, year_built: int = 2015, has_garage: int = 1) -> float:
    """
    Predict the estimated price for a house given its attributes.
    """
    pipeline = load_pipeline()
    
    input_df = pd.DataFrame([{
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'has_garage': has_garage,
        'location': location
    }])
    
    predicted_price = pipeline.predict(input_df)[0]
    return float(predicted_price)

if __name__ == "__main__":
    # Example prediction
    sample_house = {
        "square_footage": 2400,
        "bedrooms": 3,
        "bathrooms": 2,
        "location": "Uptown",
        "year_built": 2018,
        "has_garage": 1
    }
    
    print("Evaluating sample house:")
    for k, v in sample_house.items():
        print(f" - {k}: {v}")
        
    est_price = predict_house_price(**sample_house)
    print(f"\n[ESTIMATION] Estimated House Price: ${est_price:,.2f}")

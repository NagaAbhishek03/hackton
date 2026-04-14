import pandas as pd
import numpy as np
import os

def fetch_and_clean_data(output_path="clean_heart_disease.csv"):
    """
    Downloads the UCI Cleveland Heart Disease dataset, cleans missing values,
    performs feature engineering, and saves it for modeling.
    """
    print("Downloading dataset from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    
    df = pd.read_csv(url, names=columns)
    
    print("Initial shape:", df.shape)
    
    # Clean missing values which are represented as '?'
    df.replace('?', np.nan, inplace=True)
    
    # Convert 'ca' and 'thal' to numeric
    df['ca'] = pd.to_numeric(df['ca'])
    df['thal'] = pd.to_numeric(df['thal'])
    
    # Impute missing values with median for numeric columns
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].median())
    
    # Target column contains values 0, 1, 2, 3, 4. 
    # 0 = no disease, 1,2,3,4 = presence of disease. Convert to binary.
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # --- Feature Engineering ---
    # Create Age Groups
    bins = [0, 45, 60, 100]
    labels = [0, 1, 2] # 0: Young, 1: Middle-aged, 2: Senior
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    print("Missing values handled and features engineered.")
    print("Final shape:", df.shape)
    print("Target distribution:\n", df['target'].value_counts())
    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    fetch_and_clean_data()

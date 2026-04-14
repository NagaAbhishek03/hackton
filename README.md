# Heart Disease Prediction using Machine Learning 🫀

An end-to-end Machine Learning project to predict the probability of heart disease using medical attributes. This project implements a full ML pipeline: from automated data downloading, feature engineering, managing class imbalance, to model evaluation and deploying an Explainable AI Dashboard using Streamlit.

## Features
- **Data Engineering**: Cleans missing data ('?') and engineers meaningful features like `age_group` automatically.
- **Model Training Framework**: Evaluates Logistic Regression, Random Forest, XGBoost, and KNN. Validates metrics using cross-validation schemas. Handles imbalanced targets using SMOTE.
- **Explainable AI (XAI)**: SHAP-integrated Streamlit Interface for interpretable ML output.
- **Dynamic Application**: Calculates probability risk and categorizes into Low, Medium, and High-risk brackets, dispatching contextual health recommendations along with dashboard capabilities.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data
Extract and preprocess the dataset from the public UCI database:
```bash
python data.py
```

### 3. Train Models
Run the ML training job to create the `.pkl` models and scalers:
```bash
python model.py
```

### 4. Launch the Web Application
Run the Streamlit interactive dashboard:
```bash
streamlit run app.py
```

## Deployment Guide (Render/Heroku/AWS)

### Render (Recommended & Free)
1. Commit the project to a GitHub repository.
2. Sign in to Render -> New Web Service.
3. Link your GitHub. 
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `streamlit run app.py --server.port $PORT`
6. Deploy!

### AWS Elastic Beanstalk
1. Install AWS CLI and `awsebcli`.
2. Initialize with `eb init -p python-3.9`.
3. Create an environment using `eb create heart-disease-env`.
4. Deploy the application with `eb deploy`.

## Interview Preparation & Conceptual Explanations

1. **Why SMOTE?** Heart disease datasets are often imbalanced. Accuracy alone is misleading on imbalanced data. Synthetic Minority Over-sampling Technique (SMOTE) balances the classes so models like Random Forest aren't biased towards the majority outcome.
2. **Feature Engineering**: Features like Age can present non-linear risks. Putting age into buckets (`age_group`) gives tree-based algorithms easier splits to correlate aging to heart diseases.
3. **Explainable AI (Feature Importance)**: Medical professionals simply cannot trust black-box algorithms. Explaining exactly how much each feature (like high blood pressure) contributed to a prediction builds trust.
4. **Why tree-based models (XGBoost)?**: Trees models handle non-linear combinations well while remaining interpretable.

# hackton

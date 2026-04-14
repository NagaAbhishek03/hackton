import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate():
    print("Loading cleaned dataset...")
    df = pd.read_csv("clean_heart_disease.csv")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for inference
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as scaler.pkl")
    
    # Handle Class Imbalance using SMOTE on training data only
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "KNN": KNeighborsClassifier()
    }
    
    # Evaluate models
    results = {}
    best_model = None
    best_f1 = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
        
        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": roc}
        
        print(f"Results for {name}:")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    print(f"\nBest Model elected: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    # Optional: Hyperparameter tuning for XGBoost if it's the best (or RF)
    # We will just save the best found model directly due to dataset size (it's fast)
    joblib.dump(best_model, 'heart_model.pkl')
    print(f"Best model ({best_model_name}) saved as heart_model.pkl")
    
    return results

if __name__ == "__main__":
    train_and_evaluate()

# Final AML Detection Model

"""
Final model training scriots.
This script loads the cleaned dataset, applies the feature engineering
pipeline, trains the model and saves the fitted model to outputs/.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from src.data_loader import load_transaction_data
from src.feature_engineering import engineer_features

def main():
        df = load_transaction_data("../data/raw/HI-Small_Trans.csv.gz")
        df_trans = engineer_features(df)
    # Define target
        y = df_trans['is_laundering']
    # Define features 
        X = df_trans.drop(columns=['is_laundering'])
    # 70% train, 30% test stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    # Select only numeric columns
    X_train = X_train.select_dtypes(include=["number", "bool"])
    X_test = X_test.select_dtypes(include=["number", "bool"])

    # Define pipeline
    pipeline = Pipeline([
        ("var_thresh", VarianceThreshold(threshold=0.001)),
        ("model", xgb.XGBClassifier(
            n_estimators=25,
            max_depth=5,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    
    # Get Diagnostics
    y_pred = pipeline.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, "outputs/models/final_model.joblib")

    if __name__ == "__main__":
            main()

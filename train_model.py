import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, recall_score

import xgboost as xgb

def train_and_tune_xgboost(file_path):
    print("1. Memuat Dataset...")
    df = pd.read_csv("assets/stroke.csv")
    
    # Filter kategori 'Other' pada gender jika ada (sangat jarang)
    df = df[df['gender'] != 'Other']
    
    # Memisahkan Fitur dan Target
    X = df.drop(columns=['id', 'stroke'])
    y = df['stroke']
    
    # Definisi kolom
    num_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    # Preprocessing Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Menghitung ratio imbalanced class untuk scale_pos_weight
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"Rasio Imbalance Class (scale_pos_weight): {scale_pos:.2f}")
    
    # Model XGBoost Classifier
    xgb_base = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos
    )
    
    # Pipeline Utuh
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_base)
    ])
    
    # Ruang Parameter untuk Tuning
    param_distributions = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [3, 4, 5, 6, 8],
        'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2, 0.3],
        'classifier__min_child_weight': [1, 3, 5]
    }
    
    print("2. Melakukan Hyperparameter Tuning (RandomizedSearchCV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print("\n Hyperparameter Terbaik Found:")
    for param, val in search.best_params_.items():
        print(f" - {param}: {val}")
        
    best_model = search.best_estimator_
    
    # Evaluasi pada Test Set
    print("\n3. Evaluasi Model pada Data Uji (Test Set):")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"ROC-AUC Score : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Recall Score  : {recall_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Simpan Model Pipeline ke File
    model_filename = 'xgboost_stroke_pipeline.joblib'
    joblib.dump(best_model, model_filename)
    print(f"\n Pipeline model berhasil disimpan ke: '{model_filename}'")

if __name__ == '__main__':
    train_and_tune_xgboost('stroke.csv')
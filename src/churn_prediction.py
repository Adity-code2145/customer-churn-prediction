"""
Customer Churn Prediction Using Machine Learning
=================================================

This script provides a complete pipeline for:
1. Loading and preprocessing the Telco Customer Churn dataset
2. Training machine learning models
3. Evaluating and comparing models
4. Making predictions on new customer data

Author: Student
Date: 2026
"""

# =============================================
# Import Required Libraries
# =============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import joblib
import warnings
import os

warnings.filterwarnings('ignore')


# =============================================
# Step 1: Load Dataset
# =============================================
def load_dataset(filepath):
    """Load the Telco Customer Churn dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    return df


# =============================================
# Step 2: Data Preprocessing
# =============================================
def preprocess_data(df):
    """
    Preprocess the dataset:
    - Convert TotalCharges to numeric
    - Handle missing values
    - Remove customerID
    - Encode target variable
    - One-hot encode categorical features
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Convert TotalCharges to numeric (some values are blank/spaces)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing TotalCharges with median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Remove customerID (not useful for prediction)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Encode target variable: Yes -> 1, No -> 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-hot encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print(f"Preprocessing complete. Shape: {df_encoded.shape}")
    return df_encoded


# =============================================
# Step 3: Split and Scale Data
# =============================================
def split_and_scale(df_encoded):
    """
    Split data into train/test sets and apply feature scaling.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    # Separate features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    feature_names = X.columns.tolist()

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Testing set: {X_test_scaled.shape[0]} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# =============================================
# Step 4: Train Models
# =============================================
def train_models(X_train, y_train):
    """
    Train three ML models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier

    Returns:
        Dictionary of trained models
    """
    models = {}

    # Model 1: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    print("Logistic Regression trained.")

    # Model 2: Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    print("Decision Tree trained.")

    # Model 3: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    print("Random Forest trained.")

    return models


# =============================================
# Step 5: Evaluate Models
# =============================================
def evaluate_model(name, model, X_test, y_test):
    """Evaluate a model and return its performance metrics."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"Model: {name}")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    return {
        'Model': name,
        'Accuracy': round(accuracy * 100, 2),
        'Precision': round(precision * 100, 2),
        'Recall': round(recall * 100, 2),
        'F1 Score': round(f1 * 100, 2)
    }


def compare_models(models, X_test, y_test):
    """Evaluate all models and return comparison table."""
    results = []
    for name, model in models.items():
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    results_df = pd.DataFrame(results)
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON TABLE")
    print(f"{'=' * 60}")
    print(results_df.to_string(index=False))
    print(f"{'=' * 60}")

    return results_df


# =============================================
# Step 6: Select Best Model
# =============================================
def select_best_model(results_df, models):
    """Select the best model based on F1 Score."""
    best_idx = results_df['F1 Score'].idxmax()
    best_row = results_df.loc[best_idx]
    best_name = best_row['Model']
    best_model = models[best_name]

    print(f"\n{'=' * 50}")
    print("BEST MODEL")
    print(f"{'=' * 50}")
    print(f"Model:     {best_name}")
    print(f"Accuracy:  {best_row['Accuracy']}%")
    print(f"Precision: {best_row['Precision']}%")
    print(f"Recall:    {best_row['Recall']}%")
    print(f"F1 Score:  {best_row['F1 Score']}%")
    print(f"{'=' * 50}")

    return best_name, best_model


# =============================================
# Step 7: Prediction Function
# =============================================
def predict_customer_churn(customer_data, model, scaler):
    """
    Predict whether a customer is likely to churn.

    Parameters:
    -----------
    customer_data : dict
        Dictionary containing customer features.
    model : trained sklearn model
        The trained ML model.
    scaler : StandardScaler
        The fitted scaler object.

    Returns:
    --------
    str : Prediction result
    """
    # Create DataFrame from customer data
    customer_df = pd.DataFrame([customer_data])

    # Scale features
    customer_scaled = scaler.transform(customer_df)

    # Make prediction
    prediction = model.predict(customer_scaled)
    probability = model.predict_proba(customer_scaled)

    if prediction[0] == 1:
        result = "Customer is likely to CHURN"
        confidence = probability[0][1] * 100
    else:
        result = "Customer is likely to STAY"
        confidence = probability[0][0] * 100

    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    return result


# =============================================
# Step 8: Save Model
# =============================================
def save_model(model, scaler, feature_names, output_dir='../models'):
    """Save the trained model, scaler, and feature names."""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, 'churn_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.pkl'))

    print(f"Model saved to {output_dir}/churn_model.pkl")
    print(f"Scaler saved to {output_dir}/scaler.pkl")
    print(f"Feature names saved to {output_dir}/feature_names.pkl")


# =============================================
# Main Pipeline
# =============================================
def main():
    """Run the complete churn prediction pipeline."""
    print("=" * 60)
    print("  Customer Churn Prediction Using Machine Learning")
    print("=" * 60)

    # Determine the correct path to the dataset
    # Works whether run from src/ or project root
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                             'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Step 1: Load data
    print("\n--- Step 1: Loading Dataset ---")
    df = load_dataset(data_path)

    # Step 2: Preprocess
    print("\n--- Step 2: Preprocessing Data ---")
    df_encoded = preprocess_data(df)

    # Step 3: Split and scale
    print("\n--- Step 3: Splitting and Scaling Data ---")
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df_encoded)

    # Step 4: Train models
    print("\n--- Step 4: Training Models ---")
    models = train_models(X_train, y_train)

    # Step 5: Evaluate and compare
    print("\n--- Step 5: Evaluating Models ---")
    results_df = compare_models(models, X_test, y_test)

    # Step 6: Select best
    print("\n--- Step 6: Selecting Best Model ---")
    best_name, best_model = select_best_model(results_df, models)

    # Step 7: Save model
    print("\n--- Step 7: Saving Model ---")
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    save_model(best_model, scaler, feature_names, output_dir=model_dir)

    # Final summary
    best_row = results_df[results_df['Model'] == best_name].iloc[0]
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Best Model: {best_name}")
    print(f"Accuracy: {best_row['Accuracy']}%")
    print(f"F1 Score: {best_row['F1 Score']}%")
    print("=" * 50)
    print("\nProject completed successfully!")

    return models, best_model, scaler, feature_names


if __name__ == '__main__':
    main()

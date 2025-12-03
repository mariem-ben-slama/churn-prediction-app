# utils/preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

def preprocess_telco_data(filepath):
    """
    Load, clean, encode, and scale the Telco customer churn dataset.
    Returns a clean dataframe ready for modeling.
    """
    # Load data
    df = pd.read_csv(filepath)

    # Drop customerID (not useful for modeling)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    #check for NaN values:
    nan_columns = df.columns[df.isnull().any()].tolist()
    if nan_columns:
        print(f"Found NaN values in columns: {nan_columns}")
        for col in nan_columns:
            nan_count = df[col].isnull().sum()
            print(f"   - {col}: {nan_count} NaN values")
            
            # Fill categorical columns with mode, numeric with median
            if df[col].dtype == 'object':
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"     Filled with mode: {mode_val}")
            else:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"     Filled with median: {median_val}")

    # Binary mapping
    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes':1, 'No':0, 'Female':0, 'Male':1})

    # SeniorCitizen is already numeric (0,1) - Just ensure it's integer type
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    # One-hot encoding for multi-class categorical variables
    multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Convert all boolean columns to integers
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    
    # Feature scaling - USING MinMaxScaler to avoid negative values
    scaler = MinMaxScaler()
    numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    joblib.dump(scaler, 'models/scaler.pkl')

    # CRITICAL: Final NaN check
    final_nan_count = df.isnull().sum().sum()
    if final_nan_count > 0:
        print(f"CRITICAL: Still have {final_nan_count} NaN values after preprocessing!")
        print("Columns with remaining NaN values:")
        print(df.columns[df.isnull().any()].tolist())
        # Emergency fix: fill any remaining NaN with 0
        df = df.fillna(0)
        print("Emergency fix: Filled all remaining NaN with 0")
    else:
        print("No NaN values remaining in the dataset!")

    print(f"Final clean data shape: {df.shape}")
    print("Preprocessing completed successfully!")
    
    return df


def preprocess_single_input(input_dict, scaler, training_columns):
    """
    Preprocess a single customer input so it has the EXACT SAME columns
    as the training data.
    """

    # Convert dict -> DataFrame
    df = pd.DataFrame([input_dict])

    # Check for NaN in single input
    if df.isnull().any().any():
        print("Warning: NaN values found in user input")
        df = df.fillna(0)  # Fill with 0 for single inputs

    # Binary mapping
    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','SeniorCitizen']
    for col in binary_cols:
        df[col] = df[col].map({'Yes':1, 'No':0, 'Female':0, 'Male':1})

    # One-hot encoding for multi-class
    multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                  'Contract','PaymentMethod']

    df = pd.get_dummies(df, columns=multi_cols)

    # Add missing columns (set to 0)
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns, in the same order
    df = df[training_columns]

    numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Scale numeric features
    numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

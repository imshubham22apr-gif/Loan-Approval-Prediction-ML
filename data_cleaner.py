import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    return df

def clean_data(df):
    # Strip spaces from string values
    for col in df.columns:
        if df[col].dtype in ['object', 'string']:
            df[col] = df[col].astype(str).str.strip()
            
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['object', 'string']:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df

def fit_preprocessing(df, categorical_cols, numerical_cols):
    """
    Fits encoders and scalers on the dataframe and returns them.
    Note: We fit on X and y separately.
    """
    preprocessors = {
        'encoders': {},
        'scaler': None
    }
    
    # Fit LabelEncoder for each categorical column
    for col in categorical_cols:
        if col in df.columns and col != 'loan_status':
            le = LabelEncoder()
            le.fit(df[col])
            preprocessors['encoders'][col] = le
            
    # Fit MinMaxScaler for numerical columns
    if numerical_cols:
        scaler = MinMaxScaler()
        # Ensure we only fit on columns that exist in the dataframe
        existing_num_cols = [c for c in numerical_cols if c in df.columns and c != 'loan_id']
        scaler.fit(df[existing_num_cols])
        preprocessors['scaler'] = scaler
        preprocessors['scaled_cols'] = existing_num_cols
        
    return preprocessors

def transform_data(df, preprocessors, categorical_cols, numerical_cols):
    """
    Transforms df using pre-fitted preprocessors.
    """
    df_transformed = df.copy()
    
    # Transform categorical features
    for col in categorical_cols:
        if col in df_transformed.columns:
            if col == 'loan_status':
                # Explicit mapping for target label to prevent inconsistency: Approved -> 1, Rejected -> 0
                # Wait, to maintain consistency with previous logic (Approved -> 0, Rejected -> 1):
                # Let's map Approved to 1 and Rejected to 0, it makes more sense as 1 is positive (approved).
                # But let's check: we can just map it explicitly:
                df_transformed[col] = df_transformed[col].map({'Approved': 1, 'Rejected': 0})
            elif col in preprocessors['encoders']:
                le = preprocessors['encoders'][col]
                # Handle unseen categories gracefully
                df_transformed[col] = df_transformed[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                df_transformed[col] = le.transform(df_transformed[col])
                
    # Transform numerical features
    if preprocessors['scaler'] is not None:
        scaled_cols = preprocessors['scaled_cols']
        df_transformed[scaled_cols] = preprocessors['scaler'].transform(df_transformed[scaled_cols])
        
    return df_transformed

def split_data(df):
    y = df['loan_status']
    # Drop loan_id and target column loan_status from features
    X = df.drop(columns=['loan_id', 'loan_status'], errors='ignore')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

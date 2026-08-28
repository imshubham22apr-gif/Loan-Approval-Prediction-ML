import os
import sys
import pandas as pd
import joblib
import data_cleaner as dc
import model as ml

def predict_single_instance(features_dict, models_dir="models"):
    # Load model and preprocessor
    try:
        model, preprocessors = ml.load_pipeline(models_dir)
    except Exception as e:
        print(f"Error loading model from {models_dir}: {e}")
        print("Please ensure you have run main.py to train and save the model.")
        return None, None
    
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([features_dict])
    
    # Define features
    num_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
                'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
                'luxury_assets_value', 'bank_asset_value']
    cat_cols = ['education', 'self_employed']
    
    # Process the dataframe using the same data cleaner logic
    df = dc.clean_data(df)
    df_processed = dc.transform_data(df, preprocessors, categorical_cols=cat_cols, numerical_cols=num_cols)
    
    # The models are trained on the preprocessed training set, which does not contain 'loan_id' or 'loan_status'
    # Drop them if they somehow got in
    df_processed = df_processed.drop(columns=['loan_id', 'loan_status'], errors='ignore')
    
    # Make prediction
    prediction = model.predict(df_processed)[0]
    
    # Get probability if possible
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_processed)[0]
        
    status = "Approved" if prediction == 1 else "Rejected"
    confidence = prob[prediction] if prob is not None else None
    
    return status, confidence

if __name__ == "__main__":
    # If arguments are passed, parse them. Otherwise use sample data.
    if len(sys.argv) > 1:
        # Example command line usage:
        # python predict.py 2 Graduate No 9600000 29900000 12 778 2400000 17600000 22700000 8000000
        try:
            sample_features = {
                'no_of_dependents': int(sys.argv[1]),
                'education': sys.argv[2],
                'self_employed': sys.argv[3],
                'income_annum': float(sys.argv[4]),
                'loan_amount': float(sys.argv[5]),
                'loan_term': int(sys.argv[6]),
                'cibil_score': int(sys.argv[7]),
                'residential_assets_value': float(sys.argv[8]),
                'commercial_assets_value': float(sys.argv[9]),
                'luxury_assets_value': float(sys.argv[10]),
                'bank_asset_value': float(sys.argv[11])
            }
        except IndexError:
            print("Insufficient arguments! Usage:")
            print("python predict.py <dependents> <education> <self_employed> <income> <loan_amount> <term> <cibil> <residential_val> <commercial_val> <luxury_val> <bank_val>")
            sys.exit(1)
    else:
        # Use a default test sample that should typically get approved (high CIBIL score)
        print("No arguments provided. Running inference with a sample approved-case profile...")
        sample_features = {
            'no_of_dependents': 2,
            'education': 'Graduate',
            'self_employed': 'No',
            'income_annum': 9600000,
            'loan_amount': 29900000,
            'loan_term': 12,
            'cibil_score': 778,
            'residential_assets_value': 2400000,
            'commercial_assets_value': 17600000,
            'luxury_assets_value': 22700000,
            'bank_asset_value': 8000000
        }
        
    print("\nApplicant Features:")
    for k, v in sample_features.items():
        print(f"  {k}: {v}")
        
    status, confidence = predict_single_instance(sample_features)
    if status:
        print(f"\nPrediction Results:")
        print(f"  Loan Status: {status}")
        if confidence is not None:
            print(f"  Confidence: {confidence:.2%}")

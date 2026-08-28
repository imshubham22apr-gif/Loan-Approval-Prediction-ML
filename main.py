import os
import pandas as pd
import data_cleaner as dc
import model as ml

def main():
    file_path = "loan_approval_dataset.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run download_data.py first.")
        return
        
    print("Loading dataset...")
    df = dc.load_data(file_path)
    print(f"Dataset shape: {df.shape}")
    
    print("\nCleaning dataset...")
    df = dc.clean_data(df)
    
    # Map target column explicitly: Approved -> 1, Rejected -> 0
    print("Mapping target status (Approved -> 1, Rejected -> 0)...")
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    # Target distribution
    print("\nTarget distribution:")
    print(df['loan_status'].value_counts())
    
    # Define columns
    num_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
                'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
                'luxury_assets_value', 'bank_asset_value']
    cat_cols = ['education', 'self_employed']
    
    print("\nSplitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = dc.split_data(df)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    print("\nFitting preprocessing on training data...")
    preprocessors = dc.fit_preprocessing(X_train, categorical_cols=cat_cols, numerical_cols=num_cols)
    
    print("Transforming train and test data...")
    X_train_processed = dc.transform_data(X_train, preprocessors, categorical_cols=cat_cols, numerical_cols=num_cols)
    X_test_processed = dc.transform_data(X_test, preprocessors, categorical_cols=cat_cols, numerical_cols=num_cols)
    
    print("\nTraining models...")
    trained_models = ml.train_models(X_train_processed, y_train)
    
    print("\nEvaluating models...")
    evaluation_results = ml.evaluate_models(trained_models, X_test_processed, y_test)
    
    best_model_name = None
    best_accuracy = -1
    
    print("\n=== Model Comparison ===")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 65)
    for name, metrics in evaluation_results.items():
        auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
        print(f"{name:<25} | {metrics['accuracy']:.4f}     | {metrics['f1_score']:.4f}     | {auc_str}")
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    # Save the best model and preprocessors
    best_model_obj = evaluation_results[best_model_name]['model_object']
    ml.save_pipeline(best_model_obj, preprocessors, directory="models")
    
    # Print classification report and confusion matrix for best model
    print(f"\nEvaluation details for the best model ({best_model_name}):")
    print("\nConfusion Matrix:")
    print(evaluation_results[best_model_name]['confusion_matrix'])
    print("\nClassification Report:")
    print(evaluation_results[best_model_name]['classification_report'])

if __name__ == "__main__":
    main()

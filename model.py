import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate ROC-AUC if probability estimates are available
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        results[name] = {
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'classification_report': cr,
            'model_object': model
        }
    return results

def save_pipeline(model, preprocessors, directory="models"):
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, os.path.join(directory, "best_model.joblib"))
    joblib.dump(preprocessors, os.path.join(directory, "preprocessors.joblib"))
    print(f"Pipeline saved successfully in '{directory}/' directory.")

def load_pipeline(directory="models"):
    model = joblib.load(os.path.join(directory, "best_model.joblib"))
    preprocessors = joblib.load(os.path.join(directory, "preprocessors.joblib"))
    return model, preprocessors

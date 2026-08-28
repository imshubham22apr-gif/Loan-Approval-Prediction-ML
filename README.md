# Loan Approval Prediction Machine Learning Project

This repository contains a complete, end-to-end Machine Learning pipeline to predict loan approval status (Approved or Rejected) based on applicant details like CIBIL score, income, assets, and loan terms.

## 📊 Project Overview

Predicting loan default or approval is a critical task for financial institutions. This project automates the workflow by:
1. **Data Preprocessing**: Handling leading/trailing whitespaces in string values and column headers, imputing missing values, and scaling features.
2. **Exploratory Data Analysis (EDA)**: Generating and saving visualizations to understand feature distributions and correlations.
3. **Model Selection**: Training and comparing multiple classification models: **Logistic Regression**, **Decision Tree**, and **Random Forest**.
4. **Pipeline Serialization**: Storing the best-performing model along with the pre-fitted preprocessing pipeline for future inference.
5. **Interactive Prediction**: Providing a command-line interface to classify new loan applications.

---

## 📈 Model Performance & Comparison

The models were evaluated on an 80/20 train-test split stratified by the target label (`loan_status`). The comparison is as follows:

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 91.33% | 0.9315 | 0.9734 |
| Decision Tree | 98.13% | 0.9849 | 0.9801 |
| **Random Forest (Best)** | **98.24%** | **0.9859** | **0.9990** |

### Best Model Details (Random Forest)
- **Accuracy**: `98.24%`
- **F1-Score**: `0.9859`
- **ROC-AUC**: `0.9990`
- **Confusion Matrix**:
  ```text
  [[314   9]    <- [True Rejected, False Approved]
   [  6 525]]   <- [False Rejected, True Approved]
  ```

---

## 🎨 Exploratory Data Analysis Visualizations

The pipeline generates three analytical plots saved in the `plots/` directory:

1. **CIBIL Score vs Loan Status (`plots/cibil_score_vs_status.png`)**: Demonstrates the strong correlation between high CIBIL scores (generally > 600) and successful loan approvals.
2. **Annual Income vs Loan Amount (`plots/loan_amount_vs_income.png`)**: Scatter plot showing the relationship between earnings and requested loan amounts, color-coded by approval status.
3. **Correlation Matrix Heatmap (`plots/correlation_matrix.png`)**: Illustrates linear correlations between all numerical features.

---

## 📁 Repository Structure

```text
├── loan_approval_dataset.csv     # The complete dataset (approx. 4,269 records)
├── download_data.py              # Script to download dataset from source
├── data_cleaner.py               # Preprocessing pipelines (fit/transform)
├── model.py                      # Model training, evaluation, and serialization logic
├── visualize.py                  # Script to generate and save EDA plots
├── main.py                       # Main pipeline execution script
├── predict.py                    # Script to perform prediction on new profiles
├── models/                       # Directory containing saved model artifacts
│   ├── best_model.joblib         # Serialized Random Forest model
│   └── preprocessors.joblib      # Serialized Scaler & Label Encoders
├── plots/                        # Directory containing EDA plots
│   ├── cibil_score_vs_status.png
│   ├── correlation_matrix.png
│   └── loan_amount_vs_income.png
└── README.md                     # This file
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
Make sure you have Python 3.8+ installed.

### 2. Install Dependencies
Install the required packages using `pip`:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## 🚀 How to Run the Pipeline

### 1. Download Data
If you don't have the dataset locally, download the complete 4,200+ row dataset:
```bash
python download_data.py
```

### 2. Generate EDA Visualizations
Create the exploratory plots:
```bash
python visualize.py
```

### 3. Train Models
Run the complete training, evaluation, and serialization pipeline:
```bash
python main.py
```

### 4. Run Prediction / Inference
To run a prediction on a sample applicant profile (with a high CIBIL score):
```bash
python predict.py
```

To run a prediction with custom applicant parameters:
```bash
python predict.py <dependents> <education> <self_employed> <income> <loan_amount> <term> <cibil> <residential_val> <commercial_val> <luxury_val> <bank_val>
```
**Example (High Risk - Rejected)**:
```bash
python predict.py 0 "Not Graduate" No 3000000 10000000 20 300 2000000 1000000 3000000 1000000
```

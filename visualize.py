import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_cleaner as dc

def generate_plots():
    # 1. Load and clean the data
    df = dc.load_data("loan_approval_dataset.csv")
    df = dc.clean_data(df)
    
    # Create plots directory if not exists
    os.makedirs("plots", exist_ok=True)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # ---- PLOT 1: CIBIL Score Distribution by Loan Status ----
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="loan_status", y="cibil_score", hue="loan_status", palette={"Approved": "green", "Rejected": "red"}, legend=False)
    plt.title("CIBIL Score Distribution by Loan Status", fontsize=14, fontweight='bold')
    plt.xlabel("Loan Status", fontsize=12)
    plt.ylabel("CIBIL Score", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/cibil_score_vs_status.png", dpi=300)
    plt.close()
    print("Generated plots/cibil_score_vs_status.png")
    
    # ---- PLOT 2: Correlation Heatmap ----
    plt.figure(figsize=(10, 8))
    # Select numerical columns
    num_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
                'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
                'luxury_assets_value', 'bank_asset_value']
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png", dpi=300)
    plt.close()
    print("Generated plots/correlation_matrix.png")
    
    # ---- PLOT 3: Loan Amount vs. Income colored by Loan Status ----
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df, x="income_annum", y="loan_amount", hue="loan_status", 
                    palette={"Approved": "green", "Rejected": "red"}, alpha=0.7)
    plt.title("Loan Amount vs. Annual Income", fontsize=14, fontweight='bold')
    plt.xlabel("Annual Income (in Millions)", fontsize=12)
    plt.ylabel("Loan Amount (in Millions)", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/loan_amount_vs_income.png", dpi=300)
    plt.close()
    print("Generated plots/loan_amount_vs_income.png")

if __name__ == "__main__":
    generate_plots()

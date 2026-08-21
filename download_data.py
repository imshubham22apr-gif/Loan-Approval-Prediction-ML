import urllib.request
import os

def download_dataset():
    url = "https://raw.githubusercontent.com/AbhishekBiswas-github/AI-Engineer-Projects/refs/heads/main/Machine-Learning/Loan-Approval-Prediction/loan_approval_dataset.csv"
    target_path = "loan_approval_dataset.csv"
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print("Download completed successfully!")
        
        # Verify the file exists and check lines
        if os.path.exists(target_path):
            with open(target_path, 'r') as f:
                lines = f.readlines()
            print(f"Dataset has {len(lines)} lines (including header).")
            print("First few lines:")
            for line in lines[:5]:
                print(line.strip())
        else:
            print("Error: Target file was not saved.")
    except Exception as e:
        print(f"An error occurred while downloading: {e}")

if __name__ == "__main__":
    download_dataset()

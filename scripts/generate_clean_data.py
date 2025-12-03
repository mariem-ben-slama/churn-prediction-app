# scripts/generate_clean_data.py

import sys
import os

# Make sure Python can find the utils folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess_telco_data

# Generate clean dataset
input_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
output_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn-clean.csv'

clean_df = preprocess_telco_data(input_path)
clean_df.to_csv(output_path, index=False)

print("Clean dataset saved to:", output_path)

# Customer Churn Prediction App

A machine learning project for predicting customer churn, including:

- A **Streamlit web application** for running predictions  
- A **training pipeline** (`train_model.py`) for model training and export  
- A **Jupyter Notebook** with full exploratory data analysis (EDA) and model evaluation  

🔗 **Live Streamlit App:**  
https://churn-prediction-app-fzd2whg28r6g5nyajrfmk2.streamlit.app/

---

## Streamlit App (`app.py`)

The deployed app is **prediction-only**:

- Loads the final trained model (`best_model.pkl`)  
- Processes input features using the same preprocessing pipeline as training  
- Generates churn predictions for given customers  
- Displays results through a clean and responsive Streamlit interface  

> Note: The app does **not** perform EDA or model training — these steps are handled separately in `train_model.py` and the notebook.

---

## Machine Learning Workflow

This project follows a clean separation between **training** and **inference**:

### 1. Training Pipeline (`train_model.py`)

- Loads and preprocesses the dataset (`data/`)  
- Performs feature engineering  
- Trains multiple ML models (Logistic Regression, Random Forest, XGBoost)  
- Evaluates model performance  
- Selects the best-performing model (`best_model.pkl`)  
- Saves preprocessing objects (`scaler.pkl`, `training_columns.pkl`)  
- Stores trained models for reference (`models/experiments/`)  

### 2. Streamlit App (`app.py`)

- Loads `best_model.pkl` and preprocessing objects  
- Accepts customer data input  
- Applies preprocessing  
- Outputs churn prediction  

This separation ensures **fast predictions** without retraining.

---

## Analysis Notebook (`churn_analysis.ipynb`)

- Performs **EDA**  
- Cleans and preprocesses the dataset  
- Trains multiple models and evaluates them  
- Helps select the best model for deployment in the app  

---
## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd customer-churn-prediction

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app/app.py
---
## Directory Structure 
```text
churn-prediction-app/
│
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv # dataset(s)
│
├── models/
│ ├── experiments/ # trained models from experiments
│ │ ├── notebook_logreg_model.pkl
│ │ ├── notebook_rf_model.pkl
│ │ └── notebook_XGB_model.pkl
│ ├── best_model.pkl # final deployed model
│ ├── scaler.pkl # preprocessing scaler
│ └── training_columns.pkl # column info for preprocessing
│
├── scripts/
│ └── generate_clean_data.py # preprocessing / cleaning scripts
│
├── notebooks/
│ └── churn_analysis.ipynb # full EDA & model selection
│
├── streamlit_app/
│ └── app.py # prediction app
│
├── utils/
│ ├── pycache/
│ └── preprocessing.py # helper functions
│
├── train_model.py # training pipeline
├── requirements.txt
└── README.md
```

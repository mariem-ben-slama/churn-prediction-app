# models/train_model.py

import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the clean data (already preprocessed and scaled in preprocessing.py)
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn-clean.csv')

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save training columns
training_columns = X_train.columns.tolist()
joblib.dump(training_columns, 'models/training_columns.pkl')

#  MODEL SELECTION: Logistic Regression
# Why we chose Logistic Regression over Random Forest and XGBoost:
# Based on comprehensive notebook experiments, Logistic Regression demonstrated:
# - Highest Accuracy (79.8% vs 78.4% for RF and 78.6% for XGBoost)
# - Highest ROC-AUC (83.9% vs 82.1% for RF and 82.4% for XGBoost)  
# - Better balance of precision and recall
# - Faster training and inference times
# - Better interpretability for business stakeholders

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/best_model.pkl')

# Evaluate model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Model coefficients (feature importance for logistic regression)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_[0],
    'abs_coefficient': abs(model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("\n Top 10 Most Influential Features (by coefficient magnitude):")
print(feature_importance.head(10))

print("\n Model and training columns saved successfully!")
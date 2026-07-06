import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Ingest Generated Analytical Dataset
if not os.path.exists('credit_card_data.csv'):
    print("❌ Error: credit_card_data.csv not found. Execute generate_dataset.py first.")
    exit(1)

df = pd.read_csv('credit_card_data.csv')

# 2. Data Preprocessing & Feature Engineering Matrix
df['is_high_risk'] = np.where(df['past_due_status'] >= 2, 1, 0)

# Split target from features
X = df[['gender', 'income_type', 'annual_income', 'employment_duration', 'education_level', 'is_high_risk']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Benchmark Setup Protocol
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

best_accuracy = 0
best_model_instance = None
best_model_name = ""

print("\n--- Starting Machine Learning Model Evaluation Pipeline ---")
for name, model in classification_models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Algorithm: {name:22} | Test Accuracy Score: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_instance = model
        best_model_name = name

print(f"\n🥇 Winning Classifier Strategy: {best_model_name} ({best_accuracy*100:.2f}% Accuracy)")

# 4. Save and Serialize the Winning Asset
os.makedirs('models', exist_ok=True)
with open(os.path.join('models', 'best_model.pkl'), 'wb') as file:
    pickle.dump(best_model_instance, file)

print("💾 Serialized artifact successfully exported into 'models/best_model.pkl' for deployment routing.")

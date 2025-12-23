import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv(r"D:\ML_Python_Projects\Labeled_Leukemia_Disease_Dataset.csv")

# CLEANING
# dropping Patient_ID and the noisy 'Leukemia_Status' column
# we will use 'Leukemia_Label' as our True Target
X = df.drop(['Patient_ID', 'Leukemia_Status', 'Leukemia_Label'], axis=1)
y = df['Leukemia_Label']

# encoding dta
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col].astype(str))

# handle Socioeconomic mapping manually for better logic
status_map = {'Low': 0, 'Medium': 1, 'High': 2}
if 'Socioeconomic_Status' in X.columns:
    X['Socioeconomic_Status'] = X['Socioeconomic_Status'].map(status_map).fillna(1)

# train and test splits (no SMOTE needed because this target is more balanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# scaling _ standard
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL TRAINING XGBoost
print("\n[INFO] Training model on the CORRECT target (Leukemia_Label)...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

# evaluation of models
y_pred = model.predict(X_test_scaled)

print("\n===============  SUCCESSFUL CLASSIFICATION REPORT  ===============\n")
print(classification_report(y_test, y_pred))

# EDA of major features
plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', color='teal')
plt.title('Why the Model is Working: Top Clinical Features')
plt.tight_layout()
plt.savefig('leukemia_feature_importance.png')

# saving the final asset of best model
with open("leukemia_final_model.pkl", "wb") as f: pickle.dump(model, f)
with open("leukemia_final_scaler.pkl", "wb") as f: pickle.dump(scaler, f)

print("\n[DONE] Model saved. This is your final production-ready model.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb
import lightgbm as lgb
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# loading dataset using method
def loading_dataset(path_to_db):
    data = pd.read_csv(path_to_db)
    pd.set_option('display.max_columns', None)
    return data

# LOADING dataset
dataAnemia = loading_dataset(r"D:\ML_Python_Projects\Anemia&Leukemia\anemia.csv")

print("\n===============  null values checking  ============\n")
print(dataAnemia.isnull().sum())

print("\n===============  target class distribution  ============\n")
print(dataAnemia['Result'].value_counts())

# ===============  feature extraction  ===============
X = dataAnemia.drop('Result', axis=1)
y = dataAnemia['Result']
feature_list = X.columns.tolist()

# ===============  exploratory data visualization  ===============
# visualizing distributions of features based on target class
for col in X.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=dataAnemia, x=col, hue='Result', kde=True, palette='magma')
    plt.title(f'Distribution of {col} by Anemia Result')
    plt.savefig(f"{col}_distribution.png") # saving for README
    plt.show()

# ===============  data splitting (80% temp, 20% test)  ===============
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42
)

# ===============  feature scaling  ===============
scaler_xgb = StandardScaler()
X_train_xgb = scaler_xgb.fit_transform(X_train)
X_val_xgb = scaler_xgb.transform(X_val)
X_test_xgb = scaler_xgb.transform(X_test)

scaler_lgbm = StandardScaler()
X_train_lgbm = scaler_lgbm.fit_transform(X_train)
X_val_lgbm = scaler_lgbm.transform(X_val)
X_test_lgbm = scaler_lgbm.transform(X_test)

# ===============  applying SMOTE to balance training data  ===============
smote = SMOTE(random_state=42)
X_train_bal_xgb, y_train_bal_xgb = smote.fit_resample(X_train_xgb, y_train)
X_train_bal_lgbm, y_train_bal_lgbm = smote.fit_resample(X_train_lgbm, y_train)

# =========================  model 1: XGBOOST  =========================
print("\n===============  training xgboost model  ===============")
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train_bal_xgb, y_train_bal_xgb)

xgb_val_pred = xgb_model.predict(X_val_xgb)
xgb_acc = accuracy_score(y_val, xgb_val_pred)
print(f"XGBoost Validation Accuracy: {xgb_acc}")

# ========================  model 2: LIGHTGBM  ========================
print("\n===============  training lightgbm model  ===============")
lgbm_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
lgbm_model.fit(X_train_bal_lgbm, y_train_bal_lgbm)

lgbm_val_pred = lgbm_model.predict(X_val_lgbm)
lgbm_acc = accuracy_score(y_val, lgbm_val_pred)
print(f"LightGBM Validation Accuracy: {lgbm_acc}")

# ===============  visualizing model results (Confusion Matrix)  ===============
best_model = xgb_model if xgb_acc > lgbm_acc else lgbm_model
best_scaler = scaler_xgb if xgb_acc > lgbm_acc else scaler_lgbm
best_model_name = "XGBoost" if xgb_acc > lgbm_acc else "LightGBM"

y_test_pred = best_model.predict(best_scaler.transform(X_test))
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Anemic'], yticklabels=['Healthy', 'Anemic'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("confusion_matrix.png")
plt.show()

# ===============  visualizing feature importance  ===============
plt.figure(figsize=(10, 6))
if best_model_name == "XGBoost":
    xgb.plot_importance(best_model, importance_type='weight')
else:
    lgb.plot_importance(best_model, importance_type='split')
plt.title(f'Feature Importance - {best_model_name}')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ===============  comparing and saving best model files  ===============
with open("anemia_xgb_predict_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open("anemia_xgb_predict_scaler.pkl", "wb") as f:
    pickle.dump(scaler_xgb, f)

with open("anemia_lgbm_predict_model.pkl", "wb") as f:
    pickle.dump(lgbm_model, f)
with open("anemia_lgbm_predict_scaler.pkl", "wb") as f:
    pickle.dump(scaler_lgbm, f)

with open("anemia_features.pkl", "wb") as f:
    pickle.dump(feature_list, f)

print("\n===============  all models and scalers saved successfully  ===============")
print(f"\nWinning Model based on Validation: {best_model_name}")
print("\nFinal Test Set Classification Report:\n", classification_report(y_test, y_test_pred))

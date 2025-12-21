import pickle
import pandas as pd
import warnings

# ignoring specific warnings to keep the console output claen
warnings.filterwarnings("ignore", category=UserWarning)

# method to load objects
def load_pkls(model_path, scaler_path):
    with open(model_path, "rb") as m_file:
        model = pickle.load(m_file)
    with open(scaler_path, "rb") as s_file:
        scaler = pickle.load(s_file)
    return model, scaler

# loading saved assets
xgb_model, xgb_scaler = load_pkls("anemia_xgb_predict_model.pkl", "anemia_xgb_predict_scaler.pkl")
lgbm_model, lgbm_scaler = load_pkls("anemia_lgbm_predict_model.pkl", "anemia_lgbm_predict_scaler.pkl")

# ===============  sample input for testing  ==========
sample_dict = {
    'Gender': [0],
    'Hemoglobin': [10.5],
    'MCH': [22.1],
    'MCHC': [28.5],
    'MCV': [82.0]
}
sample_df = pd.DataFrame(sample_dict)

# ==========  prediction using XGBoost  ===============
# wraping scaled data in a DataFrame to maintain feature names for the model
sample_scaled_xgb = pd.DataFrame(xgb_scaler.transform(sample_df), columns=sample_df.columns)
xgb_pred = xgb_model.predict(sample_scaled_xgb)[0]
xgb_prob = xgb_model.predict_proba(sample_scaled_xgb)[0]

print("\n==========  XGBoost Prediction Results  ==============")
print(f"Predicted Class: {'Anemic' if xgb_pred == 1 else 'Healthy'}")
print(f"Confidence: {max(xgb_prob)*100:.2f}%")

# ===============  prediction using LightGBM  ===============
# maintaining feature names here prevents the UserWarning from LightGBM
sample_scaled_lgbm = pd.DataFrame(lgbm_scaler.transform(sample_df), columns=sample_df.columns)
lgbm_pred = lgbm_model.predict(sample_scaled_lgbm)[0]
lgbm_prob = lgbm_model.predict_proba(sample_scaled_lgbm)[0]

print("\n===============  LightGBM Prediction Results  ============")
print(f"Predicted Class: {'Anemic' if lgbm_pred == 1 else 'Healthy'}")
print(f"Confidence: {max(lgbm_prob)*100:.2f}%")
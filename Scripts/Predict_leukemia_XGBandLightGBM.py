from fontTools.misc.classifyTools import Classifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from datasetLeukemia import datasetForTraining
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# initialize dataset of leukemia from dataset of leukemia
leukemia_df = datasetForTraining()

leukemia_df = leukemia_df[leukemia_df['WBC_Count'] >= 0] # data cleaned


def sendDFtoVisualize():
    return leukemia_df

# separate dependent and independent variables
X = leukemia_df.drop('Leukemia_Status', axis=1)
y = leukemia_df['Leukemia_Status']

# split dataset for the training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# features standarization
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train) # fit only on training data
X_scaled_test = scaler.transform(X_test) # transform testing using training stats

# training and testing dataframe shapes
print("\nTraining Data Sample:\n", X_train.shape[0])
print("\n\nTesting Data Sample:\n", X_test.shape[0])

columns_to_check = ['WBC_Count', 'RBC_Count', 'Platelet_Count', 'Hemoglobin_Level']
negative_values = (leukemia_df[columns_to_check] < 0).sum()
print("\nNegative values in the Balanced Dataset:\n", )
print("\nnegative values in x: \n", (X[columns_to_check]< 0).sum())
print("y count: \n", y.value_counts())


# Apply XG-Boost Model for Classfication of Leukemia

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# trainig xgb model on the labeld data
xgb_model.fit(X_scaled_train, y_train)
# model predict on trained data
y_predict_xgb = xgb_model.predict(X_scaled_test)

# evaluate model performance
print("classification report of the xhb model:\n", classification_report(y_test, y_predict_xgb))
print("\n===================================\n")


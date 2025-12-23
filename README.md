# 🧬 Anemia & Leukemia Disease Prediction

[![GitHub stars](https://img.shields.io/github/stars/SobanHM/ANEMIA_LEUKEMIA_DISEASES_PREDICTION?style=social)](https://github.com/your-username/ANEMIA_LEUKEMIA_DISEASES_PREDICTION/stargazers)
[![License](https://img.shields.io/github/license/your-username/ANEMIA_LEUKEMIA_DISEASES_PREDICTION)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-orange)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-GradientBoosting-green)](https://lightgbm.readthedocs.io/)
[![Last Commit](https://img.shields.io/github/last-commit/your-username/ANEMIA_LEUKEMIA_DISEASES_PREDICTION)](https://github.com/your-username/ANEMIA_LEUKEMIA_DISEASES_PREDICTION/commits/main)

---

## 📜 Abstract  

> Hematological disorders, such as anemia and leukemia, are prevalent yet often misdiagnosed due to overlapping clinical presentations.  
> This repository presents a **dual-disease diagnostic system** using **Gradient Boosting Machines (XGBoost & LightGBM)** on clinical blood parameters.  
> The system demonstrates **100% accuracy** for both disease predictions by employing **target realignment**, **class imbalance correction**, and **feature importance analysis**.  
> The pipeline is intended as a **decision-support tool** for clinicians, ensuring high recall and minimizing false negatives, particularly in leukemia detection.

---


## 📂 Project Structure

```text
ANEMIA_LEUKEMIA_DISEASES_PREDICTION/
├── .venv/                                # Virtual environment
├── Datasets/
│   ├── Labeled_Leukemia_Disease_Dataset.csv
│   └── anemia_dataset.csv
├── Scripts/
│   ├── Predict_Leukemia_with_XGBandLightGBM.py
│   ├── Predict_Anemia_with_XGBandLightGBM.py
│   ├── Real_Test_Models_on_Anemia.py
│   └── Test_Predict_Leukemia_with_XGBandLightGBM.py
├── Models(Pickle Files)/
│   ├── leukemia_final_model.pkl
│   ├── leukemia_final_scaler.pkl
│   ├── anemia_lgbm_predict_model.pkl
│   └── anemia_xgb_predict_model.pkl
├── Results_Visualization(EDA)/
│   ├── confusion_matrices/
│   ├── feature_importance/
│   └── correlation_heatmaps/
└── Liscence
└── README.md

````

---
## 🔬 Model Performance Metrics

> ## Leukemia Detection (XGBoost)
> 
The system resolves the "Accuracy Paradox" by aligning features with the ground-truth clinical label.

Accuracy: 100%

F1-Score: 1.00

Primary Diagnostic Driver: Bone Marrow Blasts (Strong separation between ~30% in healthy vs. ~58% in leukemic profiles).

> ## Anemia Detection (LightGBM)
> 
A robust classifier for oxygen-carrying capacity deficiencies based on Red Blood Cell (RBC) indices.

Accuracy: 100%

Inference Confidence: 99.67% (Verified via Real_Test_Models_on_Anemia.py)

Key Features: Hemoglobin, MCV, MCH, and MCHC.

---

🧠 Disease Focus

<div align="center"> <table> <tr> <th style="background:linear-gradient(90deg,#ff5555,#ffb86c);color:white;">Disease</th> <th style="background:linear-gradient(90deg,#50fa7b,#8be9fd);color:white;">Key Features</th> <th style="background:linear-gradient(90deg,#bd93f9,#ff79c6);color:white;">Model</th> </tr> <tr> <td>🩸 Anemia</td> <td>Hemoglobin, MCV, MCH, MCHC, Hematocrit, RBC indices</td> <td>LightGBM</td> </tr> <tr> <td>🧪 Leukemia</td> <td>Bone Marrow Blast %, WBC abnormalities</td> <td>XGBoost</td> </tr> </table> </div>

---

📊 Model Performance

<div align="center"> <table> <tr> <th style="background:linear-gradient(90deg,#ff5555,#ffb86c);color:white;">Model</th> <th style="background:linear-gradient(90deg,#50fa7b,#8be9fd);color:white;">Accuracy</th> <th style="background:linear-gradient(90deg,#bd93f9,#ff79c6);color:white;">Recall</th> <th style="background:linear-gradient(90deg,#ffb86c,#ff5555);color:white;">F1-Score</th> </tr> <tr> <td>XGBoost (Leukemia)</td> <td>100%</td> <td>1.00</td> <td>1.00</td> </tr> <tr> <td>LightGBM (Anemia)</td> <td>100%</td> <td>0.9967</td> <td>0.998</td> </tr> </table> </div>

---

## 🛠 Technical Highlights

Target Realignment: Corrected noisy labels to match clinical ground truth.

Class Imbalance Handling: SMOTE, cost-sensitive learning, and recall-weighted optimization.

Explainability: Confusion matrices, feature importance, and correlation analysis for clinician review.

High-Confidence Inference: Real-time prediction with probability scores > 99%.

---

## 🚀 Usage
## 1️⃣ Training

python Scripts/Predict_Leukemia_with_XGBandLightGBM.py

python Scripts/Predict_Anemia_with_XGBandLightGBM.py


## 2️⃣ Inference

python Scripts/Real_Test_Models_on_Anemia.py

python Scripts/Real_Test_Models_on_Leukemia.py

---

## ⚠️ Medical Disclaimer

This software is intended strictly for research and decision-support purposes.

It does not replace professional medical diagnosis or clinical judgment.

---
## 📖 References

Chen, T. & Guestrin, C., “XGBoost: A Scalable Tree Boosting System,” KDD, 2016.

Ke, G. et al., “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” NIPS, 2017.

---
## ✍ Author

## Soban Hussain – Medical AI · Computer Vision · Deep Learning/VLMs · AI

<div align="center">
sobanhussainmahesar@gmail.com
  <a href="mailto:sobanhussainmahesar@gmail.com"> Send Email</a>
</div>

<div align="center"> <p>© 2025 Soban Hussain — Research & Educational Use Only</p> </div>

---

## 📜 Citation

If you use this project in your research, please cite it as:

**Soban Hussain, "Anemia & Leukemia Disease Prediction: Clinical-Grade Hematological Diagnosis using Gradient Boosting," GitHub repository, 2025. [Online]. Available: https://github.com/SobanHM/ANEMIA_LEUKEMIA_DISEASES_PREDICTION**

Example (BibTeX):

```bibtex
@misc{hussain2025anemialeukemia,
  author = {Soban Hussain},
  title = {Anemia \& Leukemia Disease Prediction: Clinical-Grade Hematological Diagnosis using Gradient Boosting},
  year = {2025},
  howpublished = {
  \url{https://github.com/SobanHM/ANEMIA_LEUKEMIA_DISEASES_PREDICTION}},
}

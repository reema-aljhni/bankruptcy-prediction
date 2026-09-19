# Polish Companies Bankruptcy Prediction

An end-to-end Machine Learning pipeline designed to predict corporate bankruptcy using historical financial indicators. The project handles severe class imbalance, high-dimensional features, and evaluates multiple classification algorithms.

---

## 📌 Dataset Overview
- **Dataset:** Polish Companies Bankruptcy Dataset
- **Features:** 64 financial ratios / features
- **Target Variable:** Binary classification (`0` = Operating / Solvent, `1` = Bankrupt)

---

## 🛠️ Pipeline & Methodology

### 1. Data Preprocessing & Cleaning
- **Missing Value Handling:** K-Nearest Neighbors (KNN) Imputation (`n_neighbors=5`) to effectively preserve feature relationships and feature distributions.
- **Scaling:** Standard Scaling (`StandardScaler`) applied across features.
- **Class Imbalance Management:** Synthetic Minority Over-sampling Technique (**SMOTE**) applied exclusively to training data to address class distribution disparity.

### 2. Feature Selection
- **Recursive Feature Elimination (RFE):** Systematically selected top predictive financial indicators.
- **SHAP Analysis:** Interpretability analysis to measure global feature importance and feature interaction values.

### 3. Machine Learning Models & Architecture
The project evaluates and compares multiple baseline and advanced classifiers:
- **Random Forest Classifier**
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Ensemble Voting Classifier** (Soft-voting combination)

---

## 📊 Evaluation & Performance
Models are evaluated using metrics suitable for imbalanced classification tasks:
- **Accuracy**
- **Precision, Recall, & F1-Score** (Primary focus on Recall for detecting minority bankrupt cases)
- **ROC-AUC Score**

---

## 📂 Project Structure
```text
├── bankruptcy_prediction.py   # Main ML pipeline (Preprocessing, Modeling, Evaluation)
├── .gitignore                # Environment, checkpoints, and data exclusion rules
└── README.md                 # Project documentation
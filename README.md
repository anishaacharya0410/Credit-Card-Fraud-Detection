# Credit-Card-Fraud-Detection

A Machine Learning project designed to detect fraudulent credit card transactions using classification techniques on highly imbalanced financial data.

---

## 📌 Project Overview

Credit card fraud is a significant issue in financial transactions. Detecting fraudulent activity is challenging due to the severe class imbalance in datasets (fraudulent transactions represent a tiny fraction of the total dataset).

This project cleans, preprocesses, and balances transaction data to train and evaluate classification models. It iterates across multiple approaches, progressing from a baseline Logistic Regression model to a high-performing Random Forest Classifier.

---

## 📊 Dataset Summary

The dataset consists of credit card transactions made by European cardholders.

- **Total Records:** 284,807 transactions
- **Cleaned Records:** 283,726 (after removing 1,081 duplicate rows)
- **Features (30):**
  - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
  - `V1` to `V28`: Principal components obtained via PCA (anonymized feature features).
  - `Amount`: Transaction amount.
- **Target Variable (`Class`):**
  - `0`: Legitimate transaction (283,253 samples)
  - `1`: Fraudulent transaction (473 samples)

---

## 🛠️ Project Workflow

### 1. Data Cleaning & Exploratory Data Analysis (EDA)
- **Missing Values:** Checked for missing values (none found).
- **Duplicates:** Identified and removed 1,081 duplicate records.
- **Correlation & Distributions:** Analyzed summary statistics, feature correlations, and distribution histograms of transaction amounts.

### 2. Preprocessing & Feature Scaling
- **Train-Test Split:** Split data into training (75%) and testing (25%) sets using stratified sampling to preserve the target class ratio.
- **Feature Scaling:** Applied `StandardScaler` on features to normalize data scales across numerical attributes.

### 3. Handling Class Imbalance
Applied `RandomUnderSampler` from `imblearn` to balance the class distribution during model training:
- **Version 1 (1:1 Ratio):** Balanced majority class to equal minority class size (355 legitimate vs. 355 fraudulent samples).
- **Version 2 & 3 (10:1 Ratio):** Balanced majority to minority ratio using `sampling_strategy=0.1` (3,550 legitimate vs. 355 fraudulent samples).

---

## 🤖 Models & Performance

### Comparison Matrix

| Model Version | Algorithm | Sampling Strategy | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Version 1** | Logistic Regression | Undersampling (1:1) | 95.56% | 3.30% | 90.68% | 0.0637 | 0.9743 |
| **Version 3** | Random Forest | Undersampling (10:1) | **99.87%** | **58.28%** | **80.51%** | **0.6762** | **0.9775** |

### Key Observations
- **Logistic Regression (V1)** achieves high recall (~90.68%) but suffers from very low precision (~3.30%) due to excessive false positives resulting from aggressive 1:1 undersampling.
- **Random Forest Classifier (V3)** with a 10:1 sampling ratio significantly improves precision (58.28%) and overall F1 Score (0.6762) while maintaining high Recall (80.51%) and strong ROC-AUC performance (0.9775).

---

## 🚀 Requirements & Installation

### Prerequisites
Make sure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Place the dataset `creditcard.csv` in your project folder or update the file path inside the notebook.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## 💡 Future Improvements

- Experiment with SMOTE (Synthetic Minority Over-sampling Technique) for oversampling instead of undersampling.
- Test gradient boosting algorithms such as XGBoost, LightGBM, or CatBoost.
- Perform hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for the Random Forest model.
- Adjust decision thresholds to optimize the Precision-Recall tradeoff based on business priorities.

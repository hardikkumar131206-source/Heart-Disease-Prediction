# Heart Disease Data Science Project

A complete end-to-end data science project using clinical patient data to analyze risk factors and predict the presence of heart disease. Covers exploratory data analysis, feature engineering, and multi-model machine learning with visual reporting.

---

## Overview

This project works with the UCI Heart Disease dataset schema (918 patient records, 13 clinical features) to answer two questions:

1. What patient characteristics and clinical measurements are most associated with heart disease?
2. Can we accurately predict whether a patient has heart disease from these features?

---

## Dataset

The dataset follows the UCI Heart Disease (Cleveland) format and includes the following features:

| Column | Description |
|---|---|
| age | Age in years |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type (0 = Typical Angina, 1 = Atypical, 2 = Non-Anginal, 3 = Asymptomatic) |
| trestbps | Resting blood pressure in mmHg |
| chol | Serum cholesterol in mg/dl |
| fbs | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| restecg | Resting ECG results (0, 1, 2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = Yes, 0 = No) |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | Slope of the peak exercise ST segment (0, 1, 2) |
| ca | Number of major vessels coloured by fluoroscopy (0-3) |
| thal | Thalassemia type (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect) |
| target | Heart disease present (1 = Yes, 0 = No) |

918 rows, 14 columns, no missing values.

---

## Project Structure

```
heart_disease_project/
|-- heart_disease_project.py    # Main analysis and modelling script
|-- heart_disease.csv           # Dataset
|-- fig1_eda_dashboard.png      # Exploratory data analysis output
|-- fig2_model_results.png      # Model evaluation output
|-- README.md                   # This file
```

---

## Requirements

Install dependencies with pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Tested with Python 3.10+.

---

## How to Run

```bash
python heart_disease_project.py
```

The script will print progress and a summary to the terminal, and save two figures to the working directory.

---

## What the Script Does

### 1. Data Loading and Inspection

Loads the CSV, prints shape, missing value counts, target distribution, and descriptive statistics for key numerical columns.

### 2. Exploratory Data Analysis

Generates a dashboard of 11 plots saved as `fig1_eda_dashboard.png`:

- Target distribution (pie chart)
- Age and cholesterol distributions by outcome
- Sex, chest pain type, thalassemia type, and exercise angina breakdowns
- Age vs maximum heart rate scatter plot
- Full feature correlation heatmap
- Resting blood pressure and ST depression boxplots by outcome

### 3. Preprocessing

- Categorical features (cp, restecg, slope, thal) are one-hot encoded
- Data is split 80/20 into training and test sets, stratified on the target
- Features are standardised with StandardScaler before being passed to linear models and SVM

### 4. Model Training and Comparison

Five classifiers are trained and evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- XGBoost

Each model is evaluated with 5-fold stratified cross-validation (AUC) and on the held-out test set.

### 5. Results and Visualisation

Generates `fig2_model_results.png` containing:

- Side-by-side bar chart of CV AUC and test AUC across models
- Performance metrics table (accuracy, precision, recall, F1, AUC)
- Overlaid ROC curves for all five models
- Confusion matrix for the best-performing model
- Feature importance or coefficient chart for the best model
- Boxplot of 5-fold AUC distribution per model

---

## Key Findings

### EDA Findings

- Males in this dataset have a higher rate of heart disease than females
- Asymptomatic chest pain (cp = 0) is the type most associated with disease presence
- A reversible thalassemia defect is a strong positive indicator
- Higher ST depression (oldpeak) is associated with greater disease risk
- Exercise-induced angina is highly correlated with a positive outcome
- Maximum heart rate achieved is inversely associated with disease risk

### Model Results

| Model | Test AUC | Accuracy | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.733 | 0.739 | 0.400 |
| Random Forest | 0.724 | 0.761 | 0.267 |
| SVM | 0.726 | 0.739 | 0.294 |
| XGBoost | 0.709 | 0.701 | 0.382 |
| Gradient Boosting | 0.691 | 0.728 | 0.405 |

Logistic Regression achieved the highest test AUC. The top predictive features across tree-based models were: number of major vessels coloured (ca), thalassemia type, exercise-induced angina, ST depression (oldpeak), and chest pain type.

---

## Extending the Project

Some directions for further work:

- Hyperparameter tuning with GridSearchCV or Optuna
- SHAP values for model explainability
- Handling class imbalance with SMOTE or class weighting
- Merging with other UCI regional datasets (Hungarian, Switzerland, VA) for a larger sample
- Deploying the best model as a REST API with Flask or FastAPI

---

## Data Source

Based on the UCI Heart Disease dataset originally collected from the Cleveland Clinic Foundation.

Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5), 304-310.

---

##  Author

**Hardik Kumar**
📧 hardikumar131206@gmail.com
🔗 [linkedin.com/in/yourprofile](https://www.linkedin.com/in/hardik-kumar-7631a832b)
🐙 [github.com/your-username](https://github.com/hardikkumar131206-source)

---

# Bank Customer Churn Prediction - Random Forest Classifier

## Objective
Predict which bank customers are likely to churn based on their demographic and financial data.

## Dataset
- **File:** BankChurners.csv
- **Source:** Kaggle

## Steps
1. Data cleaning & preprocessing
2. Train/Test split
3. Model training
4. Model evaluation
5. Improve the model with class weighting
6. Tune hyperparameters for the best model
7. Feature importance analysis
8. Compare with Logistic Regression & XGBoost
9. Save model for deployment

## Libraries Used
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
```

## Data Cleaning & Preprocessing
- Removed duplicates and irrelevant columns (`Surname`, `RowNumber`, `CustomerId`).
- Filled missing numeric values with the median.
- Filled missing categorical values with mode.
- Encoded `Gender` (Female=0, Male=1) and one-hot encoded `Geography`.
- Ensured `Exited` target column is integer.
- Final shape: 10,000 rows × 12 columns, no missing values.

## Train/Test Split
- Features: 11 columns (excluding `Exited`).
- Target: `Exited`.
- Split: 80% train, 20% test, stratified.
- Training set: 8000 samples, Test set: 2000 samples.

## Baseline Model
- **Random Forest Classifier** with 200 trees.
- Test Accuracy: 86.15%
- Confusion Matrix:
```
[[1537   56]
 [ 221  186]]
```
- Recall for churners (class 1): 45.7%
- ROC-AUC: 0.856

## Balanced Class Weight Model
- **Random Forest with class_weight='balanced'**
- ROC-AUC: 0.853
- Confusion Matrix:
```
[[1543   50]
 [ 227  180]]
```
- Recall for churners decreased to 44.2%.

## Hyperparameter Tuning
- **RandomizedSearchCV** with pipeline (`StandardScaler + RandomForestClassifier`)
- Best hyperparameters:
```
{'rf__n_estimators': 200, 'rf__min_samples_split': 10, 'rf__min_samples_leaf': 2, 'rf__max_features': 0.6, 'rf__max_depth': 10}
```

### Tuned Model Performance
- ROC-AUC: 0.860
- Confusion Matrix:
```
[[1396  197]
 [ 131  276]]
```
- Recall (churners): 67.8%
- Precision (churners): 58.4%
- F1-score (churners): 0.627
- Accuracy: 83.6%

### Business Insights
- **The Big Win:** Our tuned model correctly identifies 276 out of 407 at-risk customers, a huge improvement over the baseline.
- **Trade-Off (Accuracy):** Accuracy dropped to 83.6% because the model produces more false positives (197 loyal customers flagged).
- **Key Question:** Missing a churner is costlier than false alarms. Sacrificing accuracy to increase recall is justified.

**Conclusion:** Tuned Random Forest is the most effective model for identifying customers at risk, with strong Recall, F1-Score, and ROC-AUC.

## Feature Importance
Top 10 features (importance descending):
1. Age (0.322)
2. NumOfProducts (0.214)
3. Balance (0.139)
4. EstimatedSalary (0.081)
5. CreditScore (0.078)
6. IsActiveMember (0.051)
7. Geography_Germany (0.045)
8. Tenure (0.038)
9. Gender (0.016)
10. Geography_Spain (0.009)

## Model Comparison
| Model | Accuracy | Recall (Churn) | Precision (Churn) | F1-Score (Churn) | ROC-AUC |
|-------|----------|----------------|------------------|-----------------|---------|
| Logistic Regression | 80.6% | 19.0% | 57.0% | 0.29 | 0.58 |
| Random Forest (Tuned) | 83.6% | 67.8% | 58.4% | 0.627 | 0.86 |
| XGBoost | 86.8% | 49.0% | 78.0% | 0.60 | 0.73 |

**Recommendation:** Tuned Random Forest is preferred due to highest recall for churn, allowing the bank to target maximum at-risk customers.

## Deployment
```python
joblib.dump(best_rf, "RFC_churn_pipeline.pkl")
df.to_csv("BankChurners_cleaned.csv", index=False)
```
- Saved cleaned dataset and model for future use or deployment.


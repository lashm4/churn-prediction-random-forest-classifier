🏦 Customer Churn Prediction (Random Forest)

## Project Overview
This project predicts customer churn (whether a customer leaves the bank) using a **Tuned Random Forest Classifier**.
The model is trained on customer demographic and financial data and outputs churn risk.

## 🔑 Key Steps:
- Data cleaning & preprocessing  
- Train/Test split  
- Model training  
- Model evaluation  
- Hyperparameter tuning for best Random Forest model  
- Feature importance analysis  
- Comparison with Logistic Regression & XGBoost  
- Save model for deployment  

## 📊 Dataset
- **Source:** Bank Customer Churn Dataset (Kaggle)  
- **Target:** `Exited` → (1 = Churned, 0 = Stayed)  
- **Features:** Demographics, account information, and financial data.  

## ⚙️ Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction
pip install -r requirements.txt
```

## 🚀 Usage
Run the Jupyter Notebook:

```bash
jupyter notebook notebooks/random_forest_churn_model.ipynb
```

Or load the pre-trained model for predictions:

```python
import joblib
import pandas as pd

# Load Random Forest model
rf_model = joblib.load("RFC_churn_pipeline.pkl")

# Example usage
X_new = pd.DataFrame([[619, 42, 2, 0, 1, 1, 101348.88, 0, 0, 1]],
                     columns=["CreditScore","Age","Tenure","Balance","NumOfProducts",
                              "HasCrCard","EstimatedSalary","Gender","IsActiveMember","Geography_Germany","Geography_Spain"])
rf_pred = rf_model.predict(X_new)
print("Random Forest Churn Prediction:", rf_pred)
```

## 📈 Model Performance
**Tuned Random Forest (Test Set):**
- Test Accuracy: 83.6%  
- ROC-AUC: 0.860 → good separation of churners  
- Confusion Matrix:  
  ```
  [[1396  197]
   [ 131  276]]
  ```  
- Classification Report:  
  ```
  Class 0 (Non-churners): Precision = 0.914, Recall = 0.876, F1-score = 0.895
  Class 1 (Churners): Precision = 0.584, Recall = 0.678, F1-score = 0.627
  ```

### Key Observations:
- Recall for churners improved to 67.8%, identifying 276 out of 407 at-risk customers.  
- Accuracy dropped slightly due to more false positives (197 loyal customers flagged incorrectly).  
- From a business standpoint, catching more churners outweighs the cost of false positives.

## 🔑 Key Features & Insights
Top features contributing to churn (Random Forest Feature Importance):

- Age → 32.2%  
- NumOfProducts → 21.4%  
- Balance → 13.9%  
- EstimatedSalary → 8.1%  
- CreditScore → 7.8%  
- IsActiveMember → 5.1%  
- Geography_Germany → 4.5%  
- Tenure → 3.8%  
- Gender → 1.6%  
- Geography_Spain → 0.9%  
- HasCrCard → 0.7%  

**Insights:**  
- Older customers with fewer products and inactive status are more likely to churn.  
- German customers have a higher risk of leaving.  
- Financial attributes like Balance and EstimatedSalary are important predictors.  

## Next Steps
- Compare Random Forest with Logistic Regression & XGBoost  
- Monitor model performance on new data  
- Deploy model via Streamlit or Flask for real-time predictions  
- Further tune hyperparameters or test ensemble methods  

## Requirements
Dependencies are listed in `requirements.txt`, including:

- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  
- joblib  
- jupyter  

## 👩‍💻 Author
Created by [Your Name], feel free to reach out!


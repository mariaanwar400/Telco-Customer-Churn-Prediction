# 📊 Customer Churn Prediction

Predicting telecom customer churn using machine learning to help businesses identify at-risk customers before they leave.

## 🎯 Problem Statement

Customer churn (customers leaving a service) is costly for subscription-based businesses. This project analyzes telecom customer data to identify **which customers are likely to churn** and **why**, enabling proactive retention strategies.

## 📁 Dataset

- **Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Sample Dataset)
- **Rows:** 7,043 customers
- **Features:** 21 columns including demographics, account info, and services subscribed

## 🔍 Key Insights from EDA

- **Contract type is a major churn driver:** Month-to-month customers churn at a much higher rate than customers on 1-year or 2-year contracts.
- **New customers are highest risk:** Churn is heavily concentrated in the first 10 months of tenure.
- **Pricing matters:** Customers with higher monthly charges (~$70-100) churn more frequently than low-cost customers.
- **Class imbalance:** ~73.5% of customers stayed, ~26.5% churned — addressed using `class_weight='balanced'`.

## 🛠️ Tools & Libraries

`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Scikit-learn` `Joblib`

## 🧹 Data Cleaning

- Converted `TotalCharges` from `object` to `float64` (handled 11 hidden blank-space values)
- Encoded categorical variables using one-hot encoding
- Scaled numeric features using `StandardScaler`

## 🤖 Models Trained & Compared

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|---|---|---|---|---|
| Logistic Regression | 80.70% | 0.66 | 0.57 | 0.61 |
| Random Forest | 78.64% | 0.62 | 0.49 | 0.55 |
| **Logistic Regression (Balanced)** | 73.88% | 0.51 | **0.78** | 0.61 |

**Selected Model:** Logistic Regression with `class_weight='balanced'` — prioritizes catching actual churners (78% recall) over raw accuracy, since missing a real churner is more costly to a business than a false alarm.

## 📈 Top Features Driving Churn

1. **TotalCharges** (19.2%)
2. **Tenure** (17.5%)
3. **MonthlyCharges** (16.8%)
4. Payment Method (Electronic Check)
5. Internet Service (Fiber Optic)

![Feature Importance](images/feature_importance.png)

## 💡 Business Recommendation

Target retention efforts (discounts, loyalty programs) at:
- New customers within their first 10 months
- Month-to-month contract holders
- Customers paying via electronic check with fiber optic service

## 🚀 How to Run This Project

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `churn_analysis.ipynb` in Jupyter/VS Code
4. Run all cells

## 🔮 Future Improvements

- Deploy the model as an interactive web app using Streamlit
- Perform hyperparameter tuning (GridSearchCV) to further optimize Random Forest
- Experiment with additional models (XGBoost, SVM) for comparison
- Add SHAP values for deeper model interpretability

## 📂 Project Structure

```
Telco-Customer-Churn-Prediction/
├── churn_analysis.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_prediction_model.pkl
├── scaler.pkl
├── requirements.txt
├── images/
└── README.md
```
## 👤 Author
**Maria Anwar**
[LinkedIn](https://www.linkedin.com/in/maria-anwar88/) | [GitHub](https://github.com/mariaanwar400)
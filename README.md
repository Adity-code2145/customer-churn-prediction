# Customer Churn Prediction Using Machine Learning

## 📋 Project Overview

This is a Machine Learning project that predicts whether a telecom customer is likely to **churn** (leave the company) based on their profile information such as tenure, contract type, monthly charges, internet service, payment method, and more.

Customer churn is a critical problem for telecom companies. Identifying at-risk customers early allows businesses to take proactive retention measures, reducing revenue loss.

## 🎯 Problem Statement

Given customer demographics, account information, and subscribed services, predict whether a customer will churn (leave the company) or stay.

## 🎯 Objective

- Build and compare multiple Machine Learning models to predict customer churn.
- Identify the most important factors that contribute to customer churn.
- Select the best-performing model based on evaluation metrics.
- Provide a prediction function for new customer data.

## 📊 Dataset

- **Name**: IBM Telco Customer Churn Dataset
- **Source**: [IBM Sample Data Sets](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)
- **Records**: 7,043 customers
- **Features**: 21 columns
- **Target Variable**: `Churn` (Yes / No)

### Key Features:

| Feature | Description |
|---------|-------------|
| gender | Male / Female |
| SeniorCitizen | Whether the customer is a senior citizen (0/1) |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Number of months the customer has stayed |
| PhoneService | Whether the customer has phone service |
| MultipleLines | Whether the customer has multiple lines |
| InternetService | Type of internet service (DSL, Fiber optic, No) |
| OnlineSecurity | Whether the customer has online security |
| OnlineBackup | Whether the customer has online backup |
| DeviceProtection | Whether the customer has device protection |
| TechSupport | Whether the customer has tech support |
| StreamingTV | Whether the customer has streaming TV |
| StreamingMovies | Whether the customer has streaming movies |
| Contract | Contract type (Month-to-month, One year, Two year) |
| PaperlessBilling | Whether the customer has paperless billing |
| PaymentMethod | Payment method |
| MonthlyCharges | Monthly charges amount |
| TotalCharges | Total charges amount |
| Churn | Whether the customer churned (Yes/No) — **Target** |

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.x | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model Serialization |
| Jupyter Notebook | Development Environment |

## 🤖 Machine Learning Algorithms

1. **Logistic Regression** — A linear model for binary classification.
2. **Decision Tree Classifier** — A tree-based model that makes decisions based on feature thresholds.
3. **Random Forest Classifier** — An ensemble of decision trees for improved accuracy and robustness.

## 📈 Project Workflow

```
Dataset Loading → Data Understanding → Data Cleaning → Preprocessing →
EDA (Visualization) → Feature Engineering → Train-Test Split →
Feature Scaling → Model Training → Model Evaluation → Model Comparison →
Best Model Selection → Feature Importance → Prediction → Save Model
```

## 📊 Exploratory Data Analysis

The following visualizations were created:

1. **Churn Distribution** — Shows the imbalance between churned and non-churned customers.
2. **Churn vs Contract Type** — Month-to-month contracts have the highest churn rate.
3. **Churn vs Internet Service** — Fiber optic users tend to churn more.
4. **Churn vs Payment Method** — Electronic check users show higher churn.
5. **Churn vs Tenure** — Newer customers are more likely to churn.
6. **Monthly Charges Distribution** — Higher charges correlate with higher churn.
7. **Correlation Heatmap** — Shows relationships between numerical features and churn.

## 📈 Model Evaluation

Each model was evaluated using:
- **Accuracy** — Overall correctness of predictions.
- **Precision** — Of all predicted churns, how many were actual churns.
- **Recall** — Of all actual churns, how many were correctly predicted.
- **F1 Score** — Harmonic mean of Precision and Recall (balanced metric).
- **Confusion Matrix** — Visual representation of correct/incorrect predictions.
- **Classification Report** — Detailed per-class metrics.

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | **80.70%** | **65.84%** | **56.68%** | **60.92%** |
| Random Forest | 80.70% | 67.11% | 53.48% | 59.52% |
| Decision Tree | 79.42% | 62.96% | 54.55% | 58.45% |

## 🏆 Best Model

The **Logistic Regression** model is selected as the best-performing model based on the highest **F1 Score (60.92%)** and overall balanced performance:

- **Best Model**: Logistic Regression
- **Accuracy**: 80.70%
- **Precision**: 65.84%
- **Recall**: 56.68%
- **F1 Score**: 60.92%

*Why Logistic Regression is suitable for this problem:*
In customer churn prediction, the dataset is moderately imbalanced (approx. 26.5% churn rate). While Random Forest achieved slightly higher precision (67.11%), Logistic Regression achieved a significantly higher recall (56.68% vs 53.48%) and the highest F1-score (60.92%). In a business setting, catching more churning customers (higher recall) while maintaining strong precision (65.84%) and 80.70% accuracy makes Logistic Regression the optimal model.

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb           # Main Jupyter Notebook
│
├── src/
│   └── churn_prediction.py                       # Python source code
│
├── models/
│   └── churn_model.pkl                           # Saved trained model
│
├── README.md                                      # Project documentation
├── requirements.txt                               # Python dependencies
└── .gitignore                                     # Git ignore rules
```

## ▶️ How to Run

### Option 1: Jupyter Notebook (Recommended)

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   cd notebooks
   jupyter notebook Customer_Churn_Prediction.ipynb
   ```
4. Run all cells from top to bottom.

### Option 2: Google Colab

1. Upload the notebook (`Customer_Churn_Prediction.ipynb`) to Google Colab.
2. Upload the dataset CSV to Colab or modify the data path.
3. Run all cells.

### Option 3: Python Script

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   cd src
   python churn_prediction.py
   ```

## 🔮 Future Scope

- **Advanced Algorithms**: Implement Gradient Boosting, XGBoost, or SVM.
- **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters.
- **Feature Engineering**: Create derived features like charge-per-month-of-tenure.
- **Real-time Predictions**: Build an API for live churn prediction.
- **Customer Segmentation**: Apply clustering to group customers by behavior.
- **Deep Learning**: Explore neural networks for complex patterns.
- **Cross-validation**: Implement k-fold cross-validation for robust evaluation.

## 📝 Conclusion

This project demonstrates a complete machine learning workflow for predicting customer churn. By analyzing customer data and building predictive models, telecom companies can identify at-risk customers and implement targeted retention strategies. The project covers data preprocessing, exploratory data analysis, model training, evaluation, and deployment-ready prediction functions.

---

**Project Type**: Minor Project (5th Semester)  
**Subject**: Machine Learning  
**Tools**: Python, Scikit-learn, Pandas, Matplotlib, Seaborn  
**Environment**: Jupyter Notebook / Google Colab

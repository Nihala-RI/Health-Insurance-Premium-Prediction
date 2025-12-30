# Health Insurance Premium Prediction 

## Project Overview
This project predicts annual health insurance premium amounts using customer demographic, lifestyle, income, and medical information.  
The solution follows a complete machine learning lifecycle including data cleaning, feature engineering, multicollinearity handling, error-driven customer segmentation, model training, and deployment using Streamlit.

The final system uses **two different models** based on customer age to ensure fair and accurate pricing.

---

[Try the App on Streamlit](https://health-insurance-prem-prediction.streamlit.app/)


## Dataset Description

| Feature | Description |
|------|------------|
| Age | Age of the customer |
| Gender | Male / Female |
| Region | Residential region |
| Marital_status | Married / Unmarried |
| Number_of_dependants | Number of dependents |
| BMI_Category | BMI classification |
| Smoking_Status | Smoking behavior |
| Employment_Status | Employment type |
| Income_Level | Income category |
| Income_Lakhs | Income in lakhs |
| Medical_History | Existing medical conditions |
| Insurance_Plan | Bronze / Silver / Gold |
| Annual_Premium_Amount | Target variable |

---

## Data Cleaning & Preprocessing

- Dropped rows with missing values
- Converted negative values in number of dependents to absolute values
- Age capped using a business threshold of 100
- Income_Lakhs capped at the 99th percentile after business discussion (IQR avoided to prevent removal of valid records)

---

## Exploratory Data Analysis (EDA)

- Performed bivariate analysis using scatter plots:
  - Age vs Annual Premium
  - Income vs Annual Premium
  - Dependents vs Annual Premium
- Observed a positive correlation between age and premium for the overall dataset

---

## Categorical Data Standardization

Standardized inconsistent smoking status labels into a single category.

---

## Medical Risk Feature Engineering

- Split medical history into individual diseases
- Assigned risk scores to each disease
- Summed and normalized risk scores to a 0–1 range
- Created `normalized_risk_score` as a numeric health risk indicator

---

## Encoding & Feature Preparation

### Ordinal Encoding
- Insurance Plan: Bronze → 1, Silver → 2, Gold → 3
- Income Level encoded in increasing order

### One-Hot Encoding
Applied to:
- Gender
- Region
- Marital Status
- BMI Category
- Smoking Status
- Employment Status

### Feature Selection
Dropped intermediate and redundant columns after feature engineering.

---

## Correlation & Multicollinearity Analysis

- Generated correlation heatmap to understand linear relationships
- Calculated Variance Inflation Factor (VIF)
- Dropped `income_level` due to high multicollinearity
- Retained features with acceptable VIF values

---

## Feature Scaling

Scaled numerical features using MinMaxScaler:
- Age
- Number of Dependents
- Income_Lakhs
- Insurance_Plan

---

## Model Training

Trained and evaluated:
- Linear Regression
- Ridge Regression
- XGBoost Regressor

Hyperparameter tuning for XGBoost was performed using RandomizedSearchCV.

---

## Residual & Error Analysis

- Calculated percentage prediction errors
- Defined extreme error as ±10%
- Identified that ~30% of customers fell under extreme error category

Business implication:
Overcharging or undercharging premiums for a significant customer segment is not acceptable.

---

## Root Cause Analysis of Errors

- Compared feature distributions between overall test data and extreme-error customers
- Found significant distribution differences in the age feature
- Majority of extreme errors were from customers aged below 25

---

## Customer Segmentation Strategy

Segmented customers into:
- Young customers (Age ≤ 25)
- Rest customers (Age > 25)

Re-ran the full pipeline separately for both segments.

---

## Young Customer Model (Age ≤ 25)

- Introduced an additional feature: `genetical_risk`
- Age did not show a strong linear relationship within this segment
- All models showed similar accuracy
- Selected Linear Regression for simplicity, stability, and interpretability

---

## Rest Customer Model (Age > 25)

- Observed non-linear relationships
- Selected XGBoost Regressor as the final model
- Set genetical_risk = 0 for feature consistency

---

## Final Model Architecture

| Customer Segment | Model Used |
|-----------------|------------|
| Age ≤ 25 | Linear Regression |
| Age > 25 | XGBoost Regressor |

---

## Model & Scaler Artifacts

Saved using joblib:

artifacts/
├── model_young.joblib
├── model_rest.joblib
├── scaler_young.joblib
├── scaler_rest.joblib


Each scaler includes column metadata to ensure consistent inference.

---

## Streamlit Deployment

### Application Files

├── main.py
├── prediction_helper.py


### prediction_helper.py
- Loads trained models and scalers
- Encodes user input
- Calculates normalized medical risk
- Applies correct scaler
- Routes input to the appropriate model based on age

### main.py
- Builds Streamlit UI
- Collects user input
- Displays predicted insurance premium

---

## End-to-End Prediction Flow

1. User enters details in the UI
2. Input is preprocessed and scaled
3. Age-based routing is applied
4. Correct model is selected
5. Predicted premium is displayed

---

## How to Run the Application

```bash
streamlit run main.py


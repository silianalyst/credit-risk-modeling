# credit-risk-modeling
 Machine learning project for predicting loan default risk using Logistic Regression.
# Credit Risk Modeling Project

A machine learning project for predicting loan default risk using Logistic Regression on historical loan data from 2007-2014.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)

## Overview

This project develops a predictive model to assess credit risk by identifying borrowers who are likely to default on their loans. The model uses historical lending data to classify loans as either default or non-default, helping financial institutions make informed lending decisions.

**Target Variable**: Binary classification of loan default status
- **Default (1)**: Charged Off, Default, Late (31-120 days), Late (16-30 days), or Does not meet credit policy
- **Non-Default (0)**: All other loan statuses

## Dataset

**Source**: Lending Club loan data (2007-2014)  
**Original Size**: ~466,000+ loan records  
**Features**: 74 initial features (reduced through feature engineering)

The dataset includes information about:
- Loan characteristics (amount, term, interest rate, grade)
- Borrower demographics (employment, income, home ownership)
- Credit history (delinquencies, inquiries, revolving balance)
- Payment behavior and loan performance

## Features

### Data Preprocessing
- Removal of columns with excessive missing values (>95% null)
- Removal of low-variance features (>95% same value)
- Removal of highly correlated features (>0.95 correlation)
- Missing value imputation using median (numeric) and mode (categorical)
- One-hot encoding for categorical variables
- Feature standardization using StandardScaler

### Key Engineered Features
- **loan_age**: Days between loan issuance and last payment
- **tot_coll_amt_binary**: Binary indicator for collection amounts
- **grade_final**: Combined grade classification (A-G with sub-grades for A-E)

### Final Feature Set
**Numeric Features** (after filtering):
- Loan characteristics: int_rate, installment, annual_inc, dti
- Credit utilization: revol_bal, revol_util
- Credit history: delinq_2yrs, inq_last_6mths, open_acc, pub_rec, total_acc
- Payment history: total_pymnt, total_rec_int, total_rec_prncp
- And others (see notebook for complete list)

**Categorical Features**:
- term, grade, grade_final, emp_length
- home_ownership, verification_status
- pymnt_plan, purpose, addr_state, initial_list_status

## Installation

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Dependencies
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter notebook

## Usage

1. **Clone the repository** (or download the notebook)
```bash
git clone <repository-url>
cd credit-risk-modeling
```

2. **Prepare the data**
   - Ensure `loan_data_2007_2014.csv` is in the project directory

3. **Run the notebook**
```bash
jupyter notebook credit_risk_modeling.ipynb
```

4. **Execute cells sequentially**
   - Data loading and exploration
   - Feature engineering and preprocessing
   - Model training
   - Evaluation and visualization

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of numeric features
- Default rate analysis across different feature bins
- Correlation analysis and heatmap visualization
- Category-wise default rate comparison

### 2. Feature Engineering
- Created binary default label from loan status
- Log transformation for highly skewed features
- Binning continuous variables for trend analysis
- Date feature extraction (loan age calculation)

### 3. Model Development
**Algorithm**: Logistic Regression
- Class balancing using `class_weight='balanced'`
- Train-test split: 80-20 with stratification
- Maximum iterations: 1000
- Random state: 42 for reproducibility

### 4. Model Evaluation
Metrics used:
- Accuracy Score
- Confusion Matrix (TN, FP, FN, TP)
- Sensitivity (Recall) and Specificity
- Precision
- AUC-ROC Score
- Classification Report
- Feature Importance Analysis

## Results

### Model Performance
*(Actual results depend on model execution - update with your specific results)*

**Key Metrics**:
- Accuracy: ~XX%
- AUC-ROC: ~0.XX
- Sensitivity: ~XX%
- Specificity: ~XX%
- Precision: ~XX%

### Top Predictive Features
The model identifies the most influential features in predicting default risk:
1. Interest rate (int_rate)
2. Loan grade classifications
3. Debt-to-income ratio (dti)
4. Delinquency history
5. And others (see feature importance plot in notebook)

Positive coefficients indicate features associated with higher default risk, while negative coefficients suggest lower default risk.

## Key Findings

### Default Rate Insights
- **Overall default rate**: ~XX% of loans in the dataset
- **Interest rate**: Strong positive correlation with default risk
- **Loan grade**: Lower grades (F, G) show significantly higher default rates
- **Debt-to-income ratio**: Higher DTI correlates with increased default probability
- **Collection amounts**: Borrowers with prior collections show elevated default risk
- **Loan age**: Default patterns vary across different loan age periods
- **Geographic variation**: Default rates differ across states

### Risk Factors
**High Risk Indicators**:
- Higher interest rates
- Lower loan grades (F, G)
- Higher debt-to-income ratios
- Recent delinquencies
- Multiple credit inquiries
- Payment plan enrollment

**Protective Factors**:
- Higher annual income
- Better credit grades (A, B)
- Verified income status
- Longer employment history

## Future Improvements

### Model Enhancements
- [ ] Test additional algorithms (Random Forest, XGBoost, Neural Networks)
- [ ] Implement ensemble methods for improved performance
- [ ] Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Cross-validation for more robust performance estimates
- [ ] Handle class imbalance with SMOTE or other techniques

### Feature Engineering
- [ ] Create interaction features between important variables
- [ ] Implement more sophisticated time-based features
- [ ] Include external economic indicators (unemployment rate, GDP)
- [ ] Develop custom risk scores or composite features

### Analysis
- [ ] Perform SHAP or LIME analysis for model interpretability
- [ ] Segment analysis by loan purpose or borrower characteristics
- [ ] Time-series analysis of default trends
- [ ] Cost-benefit analysis for different decision thresholds

### Deployment
- [ ] Create a prediction pipeline for new loan applications
- [ ] Develop API endpoint for real-time risk scoring
- [ ] Build dashboard for model monitoring and performance tracking
- [ ] Implement automated retraining pipeline

## Project Structure
```
credit-risk-modeling/
│
├── credit_risk_modeling.ipynb    # Main analysis notebook
├── loan_data_2007_2014.csv       # Dataset (not included in repo)
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies (optional)
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is available for educational and research purposes.

## Acknowledgments
- Dataset source: Lending Club
- Built with scikit-learn, pandas, and matplotlib

---

**Note**: This project is for educational purposes only. Real-world credit risk modeling requires additional regulatory compliance, fairness considerations, and rigorous validation before deployment.

# Bank Customer Churn Prediction: Risk Scoring and Retention Prioritisation

## Abstract

Customer churn can weaken revenue stability, customer lifetime value, and long-term relationships in retail banking. This project develops a reproducible machine-learning workflow to estimate customer churn probability and group customers into risk bands for retention prioritisation. Using a 10,000-row European bank customer dataset, four classification models were compared on a stratified 80/20 train-test split. Gradient Boosting achieved the strongest overall test performance among the evaluated models, with an ROC-AUC of 0.871, accuracy of 0.869, precision of 0.782, recall of 0.494, and F1-score of 0.605. The model generated probability-based Low, Medium, and High risk categories for the 2,000-customer test set. The results demonstrate how predictive scores can support targeted review, while also showing that the current decision threshold misses approximately half of actual churners. The system is a portfolio prototype and requires further validation before any operational use.

## 1. Background

Customer churn directly affects customer lifetime value, revenue stability, cross-sell and upsell opportunities, and the competitiveness of retail banks. Traditional churn analysis describes patterns among customers who have already left. Predictive modelling can complement this analysis by estimating which existing customers may be at greater risk, allowing retention teams to prioritise investigation and engagement.

This project focuses on churn prediction and risk scoring using customer-level banking attributes. The output is intended to support retention prioritisation, not to make automatic decisions about customers.

## 2. Problem Statement

Banks may have customer-level information but lack a consistent way to convert it into churn probabilities and actionable risk groups. Without such estimates, retention activity can remain broad or reactive. This project addresses the analytical task of estimating churn risk from available customer attributes and comparing several classification approaches.

## 3. Objectives

The project objectives are to:
- Predict the target variable `Exited`, where 1 indicates churn and 0 indicates retention.
- Compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Generate churn probability scores for held-out customers.
- Assign Low, Medium, and High risk categories.
- Present customer and segment-level results through a Streamlit dashboard.
- Identify model limitations relevant to retention use.

## 4. Dataset

The project uses `Data/European_Bank.csv`, containing 10,000 customer records and 14 original columns. The target variable is `Exited`. The dataset contains 2,037 churned customers, corresponding to an overall churn rate of 20.37%.

The fields include CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, and Exited, along with a Year field in the supplied file. CustomerId, Surname, and Year are excluded from model predictors. The dataset is treated as a supplied portfolio dataset; it is not represented as live or confidential bank data.

## 5. Methodology

### 5.1 Data preparation

The workflow reads the source CSV and checks for the target column. CustomerId, Surname, and Year are removed from the predictor set. Missing numerical values are imputed using the median; missing categorical values are imputed using the most frequent category. Categorical variables are one-hot encoded, with unseen categories handled safely.

### 5.2 Feature engineering

Two derived variables are created:
- `BalanceZero`: 1 when the customer's balance is zero, otherwise 0.
- `ManyProducts`: 1 when the customer has three or more products, otherwise 0.

### 5.3 Train-test strategy

The data is split into training and test subsets using an 80/20 stratified split with random_state=42. This preserves the target class proportions across the two subsets. The held-out test set contains 2,000 customers.

### 5.4 Models

Four classifiers are trained and evaluated:
- Logistic Regression, used as a linear baseline.
- Decision Tree.
- Random Forest with 300 trees.
- Gradient Boosting.

All models are placed in preprocessing pipelines so that imputation and encoding are fitted on the training data.

### 5.5 Evaluation metrics

Accuracy measures overall classification correctness. Precision measures the proportion of predicted churners who actually churned. Recall measures the proportion of actual churners identified by the model. F1-score balances precision and recall. ROC-AUC measures the model's ability to rank churners above non-churners across classification thresholds.

## 6. Results

The recorded held-out test results are:

| Model | Accuracy | Churn precision | Churn recall | Churn F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.841 | 0.739 | 0.334 | 0.460 | 0.823 |
| Decision Tree | 0.781 | 0.466 | 0.523 | 0.493 | 0.685 |
| Random Forest | 0.861 | 0.754 | 0.467 | 0.577 | 0.854 |
| Gradient Boosting | 0.869 | 0.782 | 0.494 | 0.605 | 0.871 |

Gradient Boosting has the strongest ROC-AUC, accuracy, precision, and F1-score among these four recorded runs. Its recall is 0.494. At the default 0.50 classification threshold, its confusion matrix is TN=1,537, FP=56, FN=206, and TP=201. Thus, it identifies 201 of the 407 actual churners in the test set and misses 206.

Logistic Regression emitted a convergence warning at the configured maximum of 1,000 iterations. Its reported metrics are included for comparison but should be treated as provisional until feature scaling and convergence are addressed.

## 7. Risk Scoring and Dashboard

The Gradient Boosting model's predicted probabilities are exported as `ChurnRisk`. Risk bands are defined as:
- Low: probability less than or equal to 0.30.
- Medium: probability greater than 0.30 and less than or equal to 0.60.
- High: probability greater than 0.60.

In the 2,000-customer test set, the recorded distribution is 1,580 Low-risk customers, 234 Medium-risk customers, and 186 High-risk customers. The Streamlit dashboard displays the risk distribution, a table of high-risk test customers, and descriptive churn rates by geography, product count, and age band.

The dashboard is an analytics interface over the exported test-set scores. It is not a live connection to a bank's production systems.

## 8. Exploratory Findings and Business Interpretation

The project's exploratory analysis recorded an overall churn rate of 20.37%. Geography-level churn rates were 32.44% for Germany, 16.67% for Spain, and 16.15% for France. The recorded gender-level rates were 25.07% for female customers and 16.46% for male customers. These are descriptive associations in this dataset and should not be interpreted as causal explanations.

The analysis also highlighted customers with three or four products as a high-churn segment, with a recorded churn rate of 85.89%. This segment merits further investigation, but the relationship should be checked for sample size, data quality, and business context before designing interventions. The dataset's average account balance was 76,485.89 and its 75th percentile was 127,644.24.

## 9. Recommendations

1. Use risk bands to prioritise human review and targeted retention outreach rather than treating model scores as automatic decisions.
2. Investigate the high churn observed in the Germany segment and among customers with three or four products; validate the patterns with additional periods and operational context.
3. Review false negatives because the current Gradient Boosting threshold misses approximately half of actual churners.
4. Select a decision threshold based on the relative cost of missed churners and unnecessary retention actions, rather than accuracy alone.
5. Scale numerical predictors and retrain Logistic Regression to address its convergence warning.
6. Validate the model on newer, temporally separate data before operational deployment.
7. Assess probability calibration, segment stability, fairness, privacy, and the effectiveness of retention actions before using scores in customer-facing workflows.

## 10. Limitations

This project uses a single supplied dataset and a random stratified split rather than a time-based validation design. The results therefore do not establish future-period performance. The Gradient Boosting recall is approximately 49%, so many actual churners are not identified at the 0.50 threshold. The Logistic Regression run did not converge within 1,000 iterations. The current workflow does not include SHAP explanations, probability calibration, cross-validation, or external validation. The dashboard presents test-set scoring outputs and descriptive segment analysis; it is not a production deployment or a live bank integration.

## 11. Conclusion

This project demonstrates an end-to-end baseline workflow for bank customer churn prediction: data preparation, feature engineering, model comparison, probability scoring, risk categorisation, and dashboard presentation. Gradient Boosting performed best overall among the four evaluated models, but its recall and the Logistic Regression convergence warning show that further model validation and refinement are necessary. The outputs can support a prototype retention-prioritisation process, provided that scores are treated as decision-support signals and not as definitive predictions about individual customers.

## Reproducibility

The repository includes the source dataset, training script, model comparison output, customer risk-score output, reconstructed notebook, dashboard code, and dependency list. Run the commands in the repository README to reproduce the workflow. The notebook is explicitly labelled reconstructed because the original notebook was not available.

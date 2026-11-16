# Executive Summary
## Bank Customer Churn Prediction and Retention Prioritisation

### Purpose

This project demonstrates how customer data and machine learning can help a retail bank estimate churn risk before customers leave. The goal is to help retention teams prioritise review and engagement, rather than relying only on retrospective churn reporting.

### Dataset and approach

The analysis uses a supplied European bank customer dataset containing 10,000 records. There are 2,037 customers marked as churned, giving an overall churn rate of 20.37%.

Four models were compared: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. The workflow uses a stratified 80/20 train-test split. Customer identifiers and surnames are excluded from modelling. Missing values are imputed, categorical fields are one-hot encoded, and two additional indicators identify zero-balance customers and customers holding at least three products.

### Key results

Gradient Boosting produced the strongest overall results among the four recorded model runs:

- Accuracy: 86.9%
- Churn precision: 78.2%
- Churn recall: 49.4%
- F1-score: 60.5%
- ROC-AUC: 87.1%

The model scored 2,000 held-out customers:
- Low risk: 1,580
- Medium risk: 234
- High risk: 186

At the default 0.50 classification threshold, the model identified 201 of 407 actual churners and missed 206. This means the current model should not be treated as a complete churn-detection solution.

### Findings for further review

The exploratory analysis recorded churn rates of 32.44% in Germany, 16.67% in Spain, and 16.15% in France. The recorded rates were 25.07% for female customers and 16.46% for male customers. Customers with three or four products formed a high-churn segment in the analysis, with a recorded churn rate of 85.89%.

These are associations in the supplied dataset, not proof that geography, gender, or product count causes churn. They should guide further investigation, not determine eligibility or treatment.

### Recommended actions

1. Use the risk bands to prioritise human review and carefully designed retention outreach.
2. Investigate the Germany and high-product-count segments to understand the service, product, and customer-experience factors behind the observed patterns.
3. Review missed churners and test alternative decision thresholds based on the cost of missed churn and unnecessary outreach.
4. Validate the model on newer data and monitor performance over time before operational use.
5. Resolve the Logistic Regression convergence warning and assess probability calibration, fairness, privacy, and stability.

### Decision and limitations

The project is a portfolio prototype, not a production banking system. It uses one supplied dataset and a random train-test split; it has not been validated on future periods or integrated with live banking systems. The model's scores should be used only as decision-support signals, with human oversight.

### Conclusion

The project shows that churn probabilities and risk categories can provide a structured starting point for retention prioritisation. Gradient Boosting is the strongest model in the current comparison, but its 49.4% recall makes further validation and threshold review essential before any real-world use.

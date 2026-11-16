# Bank Customer Churn Prediction

Machine-learning project to analyse bank customer churn, compare classification models, assign churn-risk bands, and support retention prioritisation.

## Business objective
Identify customer segments associated with churn and estimate churn probability so a retention team can prioritise investigation and outreach.

## Dataset
`Data/European_Bank.csv` — 10,000 rows, 14 original columns, target `Exited`. The dataset contains 2,037 churned customers (20.37%). No missing values or duplicate rows were found during validation.

## Reproduced model results
The training workflow was rerun on the supplied dataset using a stratified 80/20 split and random_state=42.

| Model | Accuracy | Churn precision | Churn recall | Churn F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.841 | 0.739 | 0.334 | 0.460 | 0.823 |
| Decision Tree | 0.781 | 0.466 | 0.523 | 0.493 | 0.685 |
| Random Forest | 0.861 | 0.754 | 0.467 | 0.577 | 0.854 |
| Gradient Boosting | 0.869 | 0.782 | 0.494 | 0.605 | 0.871 |

Gradient Boosting confusion matrix at threshold 0.50: TN=1,537, FP=56, FN=206, TP=201. Logistic Regression emitted a convergence warning during training; its result should be treated as provisional until feature scaling and convergence are addressed.

## Risk bands
- Low: probability <= 0.30
- Medium: 0.30 < probability <= 0.60
- High: probability > 0.60

The held-out test set contains 2,000 customers: 1,580 Low, 234 Medium and 186 High. There are 407 actual churners (20.35%).

## Project structure
- `src/train_model.py`: train and compare four classifiers; export metrics and test-set risk scores.
- `dashboard/churn_dashboard.py`: Streamlit dashboard with risk and segment views.
- `notebooks/churn_model_reconstructed.ipynb`: reconstructed, reproducible notebook (not the original notebook).
- `Data/European_Bank.csv`: source dataset.
- `outputs/`: model comparison and risk-score exports.

## Run locally
```bash
pip install -r requirements.txt
python src/train_model.py --data Data/European_Bank.csv --output outputs
streamlit run dashboard/churn_dashboard.py
```

## Limitations
Gradient Boosting recall is about 49%, so it misses roughly half of actual churners at the default 0.50 threshold. Risk scores are prioritisation signals, not automatic decisions. Segment patterns are associations, not causes. Validate on newer data, check calibration, and tune thresholds against the cost of missed churners and unnecessary retention actions.

## Reproducibility note
The supplied dataset was used to rerun the modelling workflow. Results above reproduce the documented metrics to rounding. The notebook is explicitly labelled reconstructed because the original notebook was not available.

## Author
Aishwarya GM

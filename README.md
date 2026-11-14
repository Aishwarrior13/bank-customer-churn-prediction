# Bank Customer Churn Prediction

Machine-learning project to analyse bank customer churn, compare classification models, assign churn-risk bands, and support retention prioritisation.

## Business objective
Identify customer segments associated with churn and estimate churn probability so a retention team can prioritise investigation and outreach.

## Dataset
Designed around `European_Bank.csv` (10,000 rows; 14 original columns; target: `Exited`). The raw dataset is not available in the current workspace and is not included yet. Add it under `Data/European_Bank.csv` before running the scripts.

## Reported model results
The project report records these held-out test results:

| Model | Accuracy | Churn precision | Churn recall | Churn F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.84 | 0.72 | 0.37 | 0.49 |
| Decision Tree | 0.77 | 0.46 | 0.77 | 0.58 |
| Random Forest | 0.86 | 0.75 | 0.45 | 0.56 |
| Gradient Boosting | 0.87 | 0.78 | 0.49 | 0.61 |

Gradient Boosting had ROC-AUC 0.871. Its test confusion matrix was TN=1,537, FP=56, FN=206, TP=201.

## Risk bands
- Low: probability <= 0.30
- Medium: 0.30 < probability <= 0.60
- High: probability > 0.60

The reported risk-score export covers the 2,000-row held-out test set only (1,580 Low, 234 Medium, 186 High). It is not a full-population operational scorecard.

## Project structure
- `src/train_model.py`: train and evaluate classifiers; export test-set risk scores.
- `dashboard/churn_dashboard.py`: Streamlit dashboard using generated scores and the original dataset.
- `Data/`: place the source CSV here.
- `outputs/`: generated model comparison and risk-score files.

## Run locally
```bash
pip install -r requirements.txt
python src/train_model.py --data Data/European_Bank.csv --output outputs
streamlit run dashboard/churn_dashboard.py
```

## Limitations
The selected Gradient Boosting model's churn recall is 49%, so it misses about half of actual churners at the default threshold. Scores should guide review, not automatic customer decisions. Segment patterns and feature importance are associations, not causes. Validate on newer data, check calibration, and tune thresholds against the cost of missed churners and unnecessary retention actions before operational use.

## Author
Aishwarya GM

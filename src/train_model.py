"""Train churn classifiers and export held-out test-set risk scores."""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Data/European_Bank.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    if "Exited" not in df.columns:
        raise ValueError("Dataset must contain target column 'Exited'.")
    y = df["Exited"].astype(int)
    X = df.drop(columns=["Exited"] + [c for c in ["CustomerId", "Surname", "Year"] if c in df])
    if "Balance" in X: X["BalanceZero"] = (X["Balance"] == 0).astype(int)
    if "NumOfProducts" in X: X["ManyProducts"] = (X["NumOfProducts"] >= 3).astype(int)
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    metrics, best_pipe, best_prob = [], None, None
    for name, model in models.items():
        pipe = Pipeline([("preprocess", prep), ("model", model)])
        pipe.fit(Xtr, ytr); pred = pipe.predict(Xte); prob = pipe.predict_proba(Xte)[:, 1]
        metrics.append({"Model":name,"Accuracy":accuracy_score(yte,pred),
          "Churn_Precision":precision_score(yte,pred,zero_division=0),
          "Churn_Recall":recall_score(yte,pred,zero_division=0),
          "Churn_F1":f1_score(yte,pred,zero_division=0),"ROC_AUC":roc_auc_score(yte,prob)})
        if name == "Gradient Boosting": best_pipe, best_prob = pipe, prob
    pd.DataFrame(metrics).to_csv(out/"model_comparison.csv", index=False)
    scored = df.loc[Xte.index].copy()
    scored["ChurnProbability"] = best_prob
    scored["RiskCategory"] = pd.cut(best_prob,[-.000001,.30,.60,1.0],
      labels=["Low","Medium","High"],include_lowest=True).astype(str)
    scored["ActualChurn"] = yte.values
    scored.to_csv(out/"customer_churn_risk_scores.csv",index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print("Saved test-set scores:", out/"customer_churn_risk_scores.csv")

if __name__ == "__main__": main()

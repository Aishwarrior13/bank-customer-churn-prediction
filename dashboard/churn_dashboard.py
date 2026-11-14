from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bank Customer Churn Risk", layout="wide")
st.title("Bank Customer Churn Risk Dashboard")
st.caption("Risk scores support prioritisation; they should not drive automatic customer decisions.")
root = Path(__file__).resolve().parents[1]
scores_path = root/"outputs"/"customer_churn_risk_scores.csv"
data_path = root/"Data"/"European_Bank.csv"
if not scores_path.exists() or not data_path.exists():
    st.error("Missing input/output files. Add Data/European_Bank.csv and run src/train_model.py first.")
    st.stop()
scores, original = pd.read_csv(scores_path), pd.read_csv(data_path)
a,b,c,d = st.columns(4)
a.metric("Customers scored", f"{len(scores):,}")
b.metric("High risk", f"{(scores.RiskCategory=='High').sum():,}")
c.metric("Medium risk", f"{(scores.RiskCategory=='Medium').sum():,}")
d.metric("Actual churn rate", f"{scores.ActualChurn.mean():.1%}")
st.subheader("Risk category distribution")
st.bar_chart(scores.RiskCategory.value_counts().reindex(["High","Medium","Low"]).fillna(0))
st.subheader("High-risk customers")
st.dataframe(scores[scores.RiskCategory=="High"].head(100),use_container_width=True)
left,right=st.columns(2)
with left:
    st.subheader("Churn by geography")
    st.bar_chart(original.groupby("Geography").Exited.mean().mul(100))
with right:
    st.subheader("Churn by product count")
    st.bar_chart(original.groupby("NumOfProducts").Exited.mean().mul(100))
st.subheader("Churn by age band")
age=original.assign(AgeBand=pd.cut(original.Age,[0,30,40,50,60,100]))
st.bar_chart(age.groupby("AgeBand",observed=False).Exited.mean().mul(100))
st.info("The reported Gradient Boosting model achieved 49% churn recall on the held-out test set. Review false negatives and tune thresholds before operational use.")

from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Bank Customer Churn Risk",
    layout="wide"
)

st.title("Bank Customer Churn Risk Dashboard")
st.caption(
    "Risk scores support prioritisation; "
    "they should not drive automatic customer decisions."
)

root = Path(__file__).resolve().parents[1]

scores_path = root / "outputs" / "customer_churn_risk_scores.csv"
data_path = root / "Data" / "European_Bank.csv"

if not scores_path.exists() or not data_path.exists():
    st.error(
        "Missing files. Check Data/European_Bank.csv "
        "and outputs/customer_churn_risk_scores.csv."
    )
    st.stop()

scores = pd.read_csv(scores_path)
original = pd.read_csv(data_path)

# Key metrics
a, b, c, d = st.columns(4)

a.metric("Customers scored", f"{len(scores):,}")
b.metric("High risk", f"{(scores.RiskCategory == 'High').sum():,}")
c.metric("Medium risk", f"{(scores.RiskCategory == 'Medium').sum():,}")
d.metric("Actual churn rate", f"{scores.ActualChurn.mean():.1%}")

# Risk distribution
st.subheader("Risk Category Distribution")

risk_counts = (
    scores["RiskCategory"]
    .value_counts()
    .reindex(["High", "Medium", "Low"])
    .fillna(0)
)

st.bar_chart(risk_counts)

# High-risk customers
st.subheader("High-Risk Customers")

st.dataframe(
    scores[scores.RiskCategory == "High"].head(100),
    use_container_width=True
)

# Churn analysis
left, right = st.columns(2)

with left:
    st.subheader("Churn by Geography")
    geography = original.groupby("Geography")["Exited"].mean() * 100
    st.bar_chart(geography)

with right:
    st.subheader("Churn by Product Count")
    products = original.groupby("NumOfProducts")["Exited"].mean() * 100
    st.bar_chart(products)

# Churn by age band - Matplotlib avoids the Altair interval error
st.subheader("Churn by Age Band")

age = original.copy()

age["AgeBand"] = pd.cut(
    age["Age"],
    bins=[0, 30, 40, 50, 60, 100],
    labels=["0-30", "31-40", "41-50", "51-60", "61+"],
    include_lowest=True
)

age_churn = (
    age.groupby("AgeBand", observed=False)["Exited"]
    .mean()
    .mul(100)
)

fig, ax = plt.subplots()

age_churn.plot(kind="bar", ax=ax)

ax.set_xlabel("Age Band")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Customer Churn by Age Band")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.info(
    "The reported Gradient Boosting model achieved 49% churn recall "
    "on the held-out test set. Review false negatives and tune "
    "thresholds before operational use."
)

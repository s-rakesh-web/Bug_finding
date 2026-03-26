"""Optional Streamlit UI for the defect prediction prototype."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from defect_prediction_prototype import run_pipeline


st.set_page_config(page_title="Defect Risk Predictor", layout="wide")
st.title("Predictive Software Defect Detection (Prototype)")
st.caption(
    "This dashboard mines Git history, builds synthetic labels, trains an ML model, "
    "and ranks files by defect risk."
)

repo_url = st.text_input(
    "Public GitHub repository URL",
    value="https://github.com/pandas-dev/pandas",
)

if st.button("Analyze Repository"):
    with st.spinner("Mining commits and training model. This may take time..."):
        try:
            scored_df = run_pipeline(repo_url, top_n=10)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
        else:
            display_cols = ["file_name", "commits_count", "churn", "risk_score", "risk_label"]
            top_df = scored_df.sort_values("risk_score", ascending=False).head(10)[display_cols].copy()
            top_df["risk_score"] = top_df["risk_score"].round(4)

            st.subheader("Top 10 Risky Files")
            st.dataframe(top_df, use_container_width=True)

            st.subheader("Risk Score Bar Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            chart_df = top_df.sort_values("risk_score", ascending=True)
            ax.barh(chart_df["file_name"], chart_df["risk_score"])
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("File")
            ax.set_title("Top 10 Files by Predicted Defect Risk")
            st.pyplot(fig)

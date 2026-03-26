# Predictive Software Defect Detection Prototype

A complete Python prototype for **predictive software defect detection using GitHub repository mining and machine learning**.

## What it does

This project does **not** detect exact bugs. Instead, it predicts which files are likely to be defect-prone based on historical repository patterns:

- Commit frequency (`commits_count`)
- Code churn (`churn = added + deleted lines`)
- Number of contributors
- Recency (`time since last modification`)

Why these indicators matter:

- Files changed often are usually more complex and less stable.
- High churn can indicate refactoring pressure or unstable requirements.
- More contributors can increase coordination overhead and merge risk.
- Recent, frequent changes often concentrate current development risk.

## Install

```bash
pip install -r requirements.txt
```

## CLI usage

```bash
python defect_prediction_prototype.py https://github.com/<owner>/<repo> --top-n 10
```

Outputs:

- Accuracy, Precision, Recall (on synthetic labels)
- Top risky files table:
  - `file_name | commits | churn | risk_score | risk_label`

## Streamlit dashboard (optional)

```bash
streamlit run streamlit_dashboard.py
```

Dashboard features:

- Repository URL input
- Risk table for top files
- Bar chart of top risk scores

## Labeling note

Real defect prediction datasets use bug-linked commits/issues. This prototype uses a **synthetic heuristic label**:

- `label = 1` if `churn > median(churn)` and `commits_count > median(commits_count)`
- else `label = 0`

This simulates defect-prone vs non-defect-prone categories so the full pipeline can be demonstrated end-to-end.

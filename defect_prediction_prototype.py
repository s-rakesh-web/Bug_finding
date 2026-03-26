"""
Predictive Software Defect Detection Prototype
=============================================

This script mines a GitHub repository with PyDriller and builds a *prototype*
model to estimate which files are likely to be defect-prone.

Important: this is not exact bug detection. It is a risk-ranking prototype that
uses repository history patterns (commit activity, churn, contributors, recency)
as proxies for defect-proneness.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from pydriller import Repository
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


@dataclass
class FileStats:
    """Container for aggregated file-level metrics mined from commit history."""

    commits_count: int = 0
    churn: int = 0
    contributors: set = None
    last_modified: datetime | None = None

    def __post_init__(self) -> None:
        if self.contributors is None:
            self.contributors = set()


def _is_valid_code_file(path: str | None) -> bool:
    """Filter out non-file entries and common generated/binary artifacts."""
    if not path:
        return False

    # Keep the prototype broad, but ignore some obvious non-source files.
    ignored_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".pdf",
        ".zip",
        ".gz",
        ".jar",
        ".class",
        ".exe",
        ".dll",
        ".so",
    }
    lower = path.lower()
    return not any(lower.endswith(ext) for ext in ignored_extensions)


def extract_repo_data(repo_url: str) -> List[Dict]:
    """
    Mine a GitHub repository and aggregate file-level history statistics.

    Extracted metrics per file:
    - total commits affecting file
    - churn (lines added + lines deleted across history)
    - number of unique contributors
    - last modified timestamp
    """
    stats_by_file: Dict[str, FileStats] = defaultdict(FileStats)

    for commit in Repository(repo_url).traverse_commits():
        author = commit.author.name if commit.author else "unknown"

        for modified_file in commit.modified_files:
            # `new_path` handles renames/moves; fallback to old_path.
            file_path = modified_file.new_path or modified_file.old_path
            if not _is_valid_code_file(file_path):
                continue

            file_stats = stats_by_file[file_path]
            file_stats.commits_count += 1
            file_stats.churn += abs(modified_file.added_lines) + abs(modified_file.deleted_lines)
            file_stats.contributors.add(author)

            commit_date = commit.author_date
            if commit_date and (
                file_stats.last_modified is None or commit_date > file_stats.last_modified
            ):
                file_stats.last_modified = commit_date

    result = []
    for file_name, fs in stats_by_file.items():
        result.append(
            {
                "file_name": file_name,
                "commits_count": fs.commits_count,
                "churn": fs.churn,
                "contributors_count": len(fs.contributors),
                "last_modified": fs.last_modified,
            }
        )

    return result


def build_features(data: Iterable[Dict]) -> pd.DataFrame:
    """
    Convert mined file-level stats to a feature DataFrame for ML.

    Features:
    - commits_count
    - churn
    - contributors_count
    - avg_changes_per_commit
    - recency (days since last change)
    """
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("No file-level history found. Try a repository with commit history.")

    df["last_modified"] = pd.to_datetime(df["last_modified"], utc=True, errors="coerce")

    now = datetime.now(timezone.utc)
    df["avg_changes_per_commit"] = df["churn"] / np.maximum(df["commits_count"], 1)
    df["recency"] = (now - df["last_modified"]).dt.total_seconds() / 86400.0

    # For files where timestamp parsing failed, assign conservative median recency.
    median_recency = df["recency"].median()
    df["recency"] = df["recency"].fillna(median_recency)

    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic defect labels.

    Since true bug labels are unavailable in this prototype, we simulate defect
    propensity using a common heuristic inspired by defect prediction research:
    files with both high churn and high commit activity are more likely to be
    defect-prone.

    Label rule:
        IF churn > median(churn) AND commits_count > median(commits_count):
            label = 1
        ELSE:
            label = 0
    """
    churn_median = df["churn"].median()
    commits_median = df["commits_count"].median()

    df = df.copy()
    df["label"] = ((df["churn"] > churn_median) & (df["commits_count"] > commits_median)).astype(int)
    return df


def train_model(df: pd.DataFrame):
    """Train a Random Forest classifier and print quality metrics."""
    feature_cols = [
        "commits_count",
        "churn",
        "contributors_count",
        "avg_changes_per_commit",
        "recency",
    ]

    X = df[feature_cols]
    y = df["label"]

    # In very small repos label can collapse to one class; handle gracefully.
    if y.nunique() < 2:
        raise ValueError(
            "Synthetic labels contain only one class. "
            "Try a larger repository with richer commit history."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("\nModel Evaluation (Prototype)")
    print("-" * 32)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    return model


def predict_risk(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each file with defect probability and risk label.

    risk_score = P(defect-prone) in [0, 1]
    risk_label:
      > 0.6  -> HIGH RISK
      <= 0.6 -> LOW RISK
    """
    feature_cols = [
        "commits_count",
        "churn",
        "contributors_count",
        "avg_changes_per_commit",
        "recency",
    ]

    scored = df.copy()
    scored["risk_score"] = model.predict_proba(scored[feature_cols])[:, 1]
    scored["risk_label"] = np.where(scored["risk_score"] > 0.6, "HIGH RISK", "LOW RISK")
    return scored


def display_results(df: pd.DataFrame, top_n: int = 10) -> None:
    """Display top-N risky files in a clean table."""
    cols = ["file_name", "commits_count", "churn", "risk_score", "risk_label"]
    top = df.sort_values("risk_score", ascending=False).head(top_n)[cols].copy()

    top = top.rename(columns={"commits_count": "commits"})
    top["risk_score"] = top["risk_score"].round(4)

    print("\nTop Risky Files")
    print("-" * 80)
    print(top.to_string(index=False))


def run_pipeline(repo_url: str, top_n: int = 10) -> pd.DataFrame:
    """Run the complete end-to-end defect risk prediction pipeline."""
    print(f"Mining repository: {repo_url}")
    raw_data = extract_repo_data(repo_url)

    df = build_features(raw_data)
    df = create_labels(df)
    model = train_model(df)

    scored_df = predict_risk(model, df)
    display_results(scored_df, top_n=top_n)

    return scored_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict defect-prone files from GitHub repository history "
            "using PyDriller + Machine Learning (prototype)."
        )
    )
    parser.add_argument("repo_url", help="Public GitHub repository URL")
    parser.add_argument("--top-n", type=int, default=10, help="Number of risky files to display")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.repo_url, top_n=args.top_n)

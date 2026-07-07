"""
eda.py
------
Quick exploratory data analysis on the raw dataset. Saves a handful of
plots (PNG) into outputs/eda/ so they can be reviewed without needing a
notebook viewer.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_raw_data, clean_data, TARGET_COL, NUMERIC_COLS

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def run_eda():
    df = clean_data(load_raw_data())

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nTarget class balance:\n", df[TARGET_COL].value_counts())

    # 1. Target class distribution
    plt.figure(figsize=(10, 5))
    order = df[TARGET_COL].value_counts().index
    sns.countplot(y=TARGET_COL, data=df, order=order, hue=TARGET_COL,
                  palette="viridis", legend=False)
    plt.title("Suggested Job Role - class distribution")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "target_distribution.png"), dpi=120)
    plt.close()

    # 2. Numeric feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "numeric_distributions.png"), dpi=120)
    plt.close()

    # 3. Correlation heatmap of numeric features
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[NUMERIC_COLS].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation between numeric features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    # 4. Categorical vs target example: Management or Technical
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y=TARGET_COL, hue="Management or Technical",
                  order=order, palette="Set2")
    plt.title("Job role by Management/Technical preference")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "role_by_mgmt_technical.png"), dpi=120)
    plt.close()

    print(f"\nEDA plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    run_eda()

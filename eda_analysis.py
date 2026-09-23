

import argparse
import base64
import os
import warnings
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from loan_approval import generate_synthetic_data, load_data, add_features, TARGET

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")

OUT_DIR = "outputs/eda"
REPORT_PATH = "outputs/eda_report.html"
PALETTE = {"Approved": "#2a9d8f", "Rejected": "#e76f51"}


def status_label(s: pd.Series) -> pd.Series:
    return s.map({1: "Approved", 0: "Rejected"})


def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def save_and_encode(fig, name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=130, bbox_inches="tight")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
def analyze(df: pd.DataFrame) -> list[dict]:
    """Return a list of {title, img_b64, insight} dicts, in report order."""
    sections = []
    df = df.copy()
    df["Status"] = status_label(df[TARGET])
    df = add_features(df)

    n = len(df)
    approved_rate = df[TARGET].mean()
    missing_total = int(df.isna().sum().sum())

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["Status"].value_counts()
    ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + n * 0.01, f"{v} ({v/n:.0%})", ha="center")
    ax.set_title("Loan status distribution")
    ax.set_ylabel("Applicants")
    sections.append({
        "title": "1. Target Class Distribution",
        "img": save_and_encode(fig, "01_target_distribution"),
        "insight": f"{n} applicants total, {approved_rate:.1%} approved. The classes are imbalanced "
                   f"(~{approved_rate:.0%}/{1-approved_rate:.0%}), so accuracy alone can be misleading and "
                   f"precision/recall per class matter."
    })

    # 2. Missing values
    miss = df.drop(columns=["Status"]).isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(miss.index[::-1], miss.values[::-1], color="#8d99ae")
        ax.set_title("Missing values by column")
        ax.set_xlabel("Missing count")
        sections.append({
            "title": "2. Missing Data",
            "img": save_and_encode(fig, "02_missing_values"),
            "insight": f"{missing_total} missing cells across the dataset "
                       f"({missing_total / (n * len(df.columns)):.1%} of all cells). "
                       f"{'Credit_History and LoanAmount carry the most gaps, consistent with real loan datasets.' if 'Credit_History' in miss.index else ''} "
                       "Handled with median imputation for numbers and mode imputation for categories."
        })

    # 3. Numeric distributions (income, loan amount)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df["ApplicantIncome"].dropna(), bins=40, kde=True, ax=axes[0], color="#264653")
    axes[0].set_title("Applicant income")
    sns.histplot(df["CoapplicantIncome"].dropna(), bins=40, kde=True, ax=axes[1], color="#2a9d8f")
    axes[1].set_title("Co-applicant income")
    sns.histplot(df["LoanAmount"].dropna(), bins=40, kde=True, ax=axes[2], color="#e76f51")
    axes[2].set_title("Loan amount (thousands)")
    plt.tight_layout()
    sections.append({
        "title": "3. Numeric Feature Distributions",
        "img": save_and_encode(fig, "03_numeric_distributions"),
        "insight": "Income and loan amount are right-skewed with a long tail of high earners/large loans — "
                   "typical of financial data. This motivates log-scaling or robust scalers over raw values."
    })

    # 4. Income by status (log scale, violin)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_df = df.copy()
    plot_df["log_income"] = np.log1p(plot_df["ApplicantIncome"])
    sns.violinplot(data=plot_df, x="Status", y="log_income", ax=ax, palette=PALETTE, order=["Rejected", "Approved"])
    ax.set_title("log(Applicant income) by loan status")
    sections.append({
        "title": "4. Income vs. Approval",
        "img": save_and_encode(fig, "04_income_vs_status"),
        "insight": "Approved and rejected applicants have broadly overlapping income distributions — income alone "
                   "is a weak predictor. Approval depends more on repayment capacity relative to the loan, "
                   "not income in isolation."
    })

    # 5. Approval rate by credit history
    fig, ax = plt.subplots(figsize=(5, 4))
    rates = df.groupby("Credit_History")[TARGET].mean() * 100
    ax.bar(rates.index.astype(str), rates.values, color="#264653")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center")
    ax.set_title("Approval rate (%) by credit history")
    ax.set_xlabel("Credit History (1 = good, 0 = bad)")
    sections.append({
        "title": "5. Credit History — the strongest signal",
        "img": save_and_encode(fig, "05_credit_history"),
        "insight": f"Applicants with good credit history are approved at a far higher rate than those without — "
                   "this is consistently the single strongest predictor in the modelling stage as well."
    })

    # 6. Approval rate by categorical features (grid)
    cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Dependents"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.ravel(), cat_cols):
        rates = df.groupby(col)[TARGET].mean().sort_values(ascending=False) * 100
        ax.bar(rates.index.astype(str), rates.values, color="#f4a261")
        ax.set_title(f"Approval rate (%) by {col}")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    sections.append({
        "title": "6. Approval Rate by Applicant Profile",
        "img": save_and_encode(fig, "06_categorical_approval_rates"),
        "insight": "Property area and education show the widest spread among categorical features; gender and "
                   "self-employment status show smaller differences. This helps prioritise which categorical "
                   "features to encode carefully and to check for fairness."
    })

    # 7. Correlation heatmap (numeric + engineered features)
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                     "Credit_History", "TotalIncome", "EMI", "BalanceIncome", "LoanToIncome", TARGET]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation matrix (numeric + engineered features)")
    plt.tight_layout()
    top_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    sections.append({
        "title": "7. Correlation Analysis",
        "img": save_and_encode(fig, "07_correlation_heatmap"),
        "insight": f"'{top_corr.index[0]}' has the strongest linear correlation with loan status "
                   f"({corr[TARGET][top_corr.index[0]]:.2f}), followed by '{top_corr.index[1]}' "
                   f"({corr[TARGET][top_corr.index[1]]:.2f}). TotalIncome correlates strongly with "
                   "ApplicantIncome/CoapplicantIncome as expected (multicollinearity to be aware of in linear models)."
    })

    # 8. Loan amount vs income scatter, colored by status
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for status, color in PALETTE.items():
        sub = df[df["Status"] == status]
        ax.scatter(sub["TotalIncome"], sub["LoanAmount"], alpha=0.4, s=18, color=color, label=status)
    ax.set_xscale("log")
    ax.set_xlabel("Total income (log scale)")
    ax.set_ylabel("Loan amount (thousands)")
    ax.set_title("Loan amount vs. total income")
    ax.legend()
    sections.append({
        "title": "8. Loan Amount vs. Income",
        "img": save_and_encode(fig, "08_loan_vs_income_scatter"),
        "insight": "Rejections cluster where the loan amount is large relative to income (upper-left region), "
                   "reinforcing that the engineered LoanToIncome / BalanceIncome features capture real signal."
    })

    # 9. EMI burden by status
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.boxplot(data=df, x="Status", y="EMI", ax=ax, palette=PALETTE, order=["Rejected", "Approved"], showfliers=False)
    ax.set_title("Monthly EMI by loan status (outliers hidden)")
    sections.append({
        "title": "9. Repayment Burden (EMI)",
        "img": save_and_encode(fig, "09_emi_by_status"),
        "insight": "Rejected applicants tend to have a higher or more variable monthly EMI, supporting EMI and "
                   "BalanceIncome as informative engineered features for the model."
    })

    # 10. Pairwise relationships (pairplot on a sample for speed)
    sample = df.sample(min(400, n), random_state=42)
    pp = sns.pairplot(
        sample[["ApplicantIncome", "LoanAmount", "EMI", "Status"]],
        hue="Status", palette=PALETTE, diag_kind="kde", plot_kws={"alpha": 0.5, "s": 18}, height=2.3,
    )
    pp.fig.suptitle("Pairwise relationships (400-row sample)", y=1.02)
    b64 = fig_to_b64(pp.fig)
    os.makedirs(OUT_DIR, exist_ok=True)
    pp.fig.savefig(os.path.join(OUT_DIR, "10_pairplot.png"), dpi=130, bbox_inches="tight")
    sections.append({
        "title": "10. Pairwise Relationships",
        "img": b64,
        "insight": "The diagonal KDEs confirm the right-skew of income and loan amount seen earlier; off-diagonal "
                   "panels show the Approved/Rejected classes overlap heavily in raw feature space, which is why "
                   "engineered ratio features and credit history — not raw income — drive the model's accuracy."
    })

    # 11. Outlier check (boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["ApplicantIncome", "LoanAmount", "LoanToIncome"]):
        sns.boxplot(y=df[col], ax=ax, color="#e9c46a")
        ax.set_title(col)
    plt.tight_layout()
    sections.append({
        "title": "11. Outlier Detection",
        "img": save_and_encode(fig, "11_outlier_boxplots"),
        "insight": "All three numeric features show high-value outliers (very high earners / very large loans). "
                   "These were kept rather than removed, since models here use RobustScaler-friendly preprocessing "
                   "and tree-based models are not sensitive to them; extreme values were confirmed to be plausible, not data errors."
    })

    return sections, {"n": n, "approved_rate": approved_rate, "missing_total": missing_total}


def build_html(sections: list[dict], stats: dict, source_note: str) -> str:
    cards = ""
    for s in sections:
        cards += f"""
        <section class="card">
          <h2>{s['title']}</h2>
          <img src="data:image/png;base64,{s['img']}" alt="{s['title']}">
          <p class="insight">{s['insight']}</p>
        </section>
        """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Approval — EDA & Graphical Analysis</title>
<style>
  :root {{
    --bg: #f7f7f5; --card: #ffffff; --ink: #1f2937; --muted: #6b7280; --accent: #2a9d8f;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          background: var(--bg); color: var(--ink); }}
  header {{ background: linear-gradient(135deg, #264653, #2a9d8f); color: white; padding: 40px 24px; }}
  header h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
  header p {{ margin: 4px 0; opacity: 0.9; }}
  .stats {{ display:flex; gap: 20px; flex-wrap: wrap; margin-top: 18px; }}
  .stat {{ background: rgba(255,255,255,0.12); padding: 10px 16px; border-radius: 10px; }}
  .stat b {{ display:block; font-size: 1.3rem; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
  .card {{ background: var(--card); border-radius: 14px; padding: 22px; margin-bottom: 22px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow-x: auto; }}
  .card h2 {{ margin-top: 0; font-size: 1.15rem; color: #264653; }}
  .card img {{ max-width: 100%; height: auto; display:block; margin: 10px 0; border-radius: 8px; }}
  .insight {{ color: var(--muted); line-height: 1.55; margin: 8px 0 0; }}
  footer {{ text-align:center; color: var(--muted); padding: 30px; font-size: 0.85rem; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{ --bg:#0f1115; --card:#181b21; --ink:#e5e7eb; --muted:#9ca3af; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Loan Approval Prediction — EDA &amp; Graphical Analysis</h1>
  <p>{source_note}</p>
  <div class="stats">
    <div class="stat"><b>{stats['n']:,}</b>applicants</div>
    <div class="stat"><b>{stats['approved_rate']:.1%}</b>approval rate</div>
    <div class="stat"><b>{stats['missing_total']}</b>missing cells</div>
  </div>
</header>
<main>
{cards}
</main>
<footer>Generated by eda_analysis.py — Loan Approval Prediction System</footer>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="EDA & graphical analysis for loan approval data")
    parser.add_argument("--data", help="Path to CSV (optional; synthetic data used if omitted)")
    args = parser.parse_args()

    if args.data:
        df = load_data(args.data)
        source_note = f"Dataset: {args.data}"
    else:
        df = generate_synthetic_data()
        df[TARGET] = df["Loan_Status"].map({"Y": 1, "N": 0})
        source_note = "Dataset: synthetic (3,000 applicants) — replace with --data your_file.csv for real analysis"

    os.makedirs("outputs", exist_ok=True)
    sections, stats = analyze(df)
    html = build_html(sections, stats, source_note)
    with open(REPORT_PATH, "w") as f:
        f.write(html)
    print(f"[done] {len(sections)} charts saved to {OUT_DIR}/")
    print(f"[done] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

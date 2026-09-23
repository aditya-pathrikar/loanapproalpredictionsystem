"""
Loan Approval Prediction System
================================
End-to-end machine learning pipeline:
  data loading / synthetic generation -> EDA plots -> feature engineering
  -> preprocessing -> model comparison -> evaluation -> model saving -> prediction

Usage
-----
  python loan_approval.py train                      # uses synthetic data
  python loan_approval.py train --data loan_data.csv # uses your own CSV
  python loan_approval.py predict --json '{"Gender": "Male", ...}'
  python loan_approval.py predict --interactive
  python loan_approval.py generate-data --rows 3000  # writes loan_data.csv

Expected CSV columns (same as the popular Kaggle "Loan Prediction" dataset):
  Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
  CoapplicantIncome, LoanAmount (in thousands), Loan_Amount_Term (months),
  Credit_History (1/0), Property_Area, Loan_Status (Y/N)
  (a Loan_ID column, if present, is ignored)
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend so plots save without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "loan_model.joblib")

CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "TotalIncome",
    "EMI",
    "BalanceIncome",
    "LoanToIncome",
]
RAW_FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area",
]
TARGET = "Loan_Status"


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def generate_synthetic_data(n_rows: int = 3000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a realistic synthetic loan dataset (used when no CSV is supplied)."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n_rows, p=[0.8, 0.2])
    married = rng.choice(["Yes", "No"], n_rows, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], n_rows, p=[0.57, 0.17, 0.16, 0.10])
    education = rng.choice(["Graduate", "Not Graduate"], n_rows, p=[0.78, 0.22])
    self_emp = rng.choice(["Yes", "No"], n_rows, p=[0.14, 0.86])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n_rows, p=[0.33, 0.38, 0.29])

    applicant_income = rng.lognormal(mean=8.5, sigma=0.6, size=n_rows).round(0)
    applicant_income += np.where(education == "Graduate", 1200, 0)
    coapplicant_income = np.where(
        rng.random(n_rows) < 0.45, 0, rng.lognormal(mean=7.6, sigma=0.7, size=n_rows)
    ).round(0)

    total_income = applicant_income + coapplicant_income
    loan_amount = (total_income * rng.uniform(0.02, 0.06, n_rows)).round(0)  # in thousands
    loan_term = rng.choice([360, 240, 180, 120, 84, 60], n_rows, p=[0.78, 0.06, 0.07, 0.03, 0.03, 0.03])
    credit_history = rng.choice([1, 0], n_rows, p=[0.82, 0.18])

    # Hidden approval logic (with noise) so the ML models have a real signal to learn
    emi = loan_amount * 1000 / loan_term
    burden = emi / (total_income + 1)
    score = (
        3.0 * (credit_history - 0.5)
        - 6.0 * burden
        + 0.35 * (education == "Graduate")
        + 0.30 * (married == "Yes")
        + 0.35 * (property_area == "Semiurban")
        - 0.15 * (dependents == "3+")
        - 0.20 * (self_emp == "Yes")
        + 0.00006 * total_income
        + rng.normal(0, 0.7, n_rows)
    )
    threshold = np.quantile(score, 0.30)  # ~70% approval rate
    status = np.where(score > threshold, "Y", "N")

    df = pd.DataFrame(
        {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_emp,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term.astype(float),
            "Credit_History": credit_history.astype(float),
            "Property_Area": property_area,
            "Loan_Status": status,
        }
    )

    # Inject realistic missing values
    for col, frac in [("Gender", 0.02), ("Married", 0.01), ("Dependents", 0.025),
                      ("Self_Employed", 0.05), ("LoanAmount", 0.035),
                      ("Loan_Amount_Term", 0.02), ("Credit_History", 0.08)]:
        df.loc[rng.random(n_rows) < frac, col] = np.nan
    return df


def load_data(path: str | None) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
        print(f"[data] Loaded {len(df)} rows from {path}")
    else:
        df = generate_synthetic_data()
        print(f"[data] No CSV given -> generated {len(df)} synthetic rows")
    df = df.drop(columns=[c for c in ["Loan_ID"] if c in df.columns])
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        sys.exit(f"[error] CSV is missing required columns: {missing}")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0, "1": 1, "0": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df


# --------------------------------------------------------------------------- #
# 2. Feature engineering & preprocessing
# --------------------------------------------------------------------------- #
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features. Safe to call on missing values (pipeline imputes later)."""
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    term = df["Loan_Amount_Term"].replace(0, np.nan)
    df["EMI"] = df["LoanAmount"] * 1000 / term                      # monthly instalment
    df["BalanceIncome"] = df["TotalIncome"] - df["EMI"]             # income left after EMI
    df["LoanToIncome"] = df["LoanAmount"] * 1000 / (df["TotalIncome"] + 1)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


# --------------------------------------------------------------------------- #
# 3. Exploratory plots
# --------------------------------------------------------------------------- #
def save_eda_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    df[TARGET].map({1: "Approved", 0: "Rejected"}).value_counts().plot.bar(ax=axes[0, 0], color=["#2a9d8f", "#e76f51"])
    axes[0, 0].set_title("Loan status distribution")

    (df.groupby("Credit_History")[TARGET].mean() * 100).plot.bar(ax=axes[0, 1], color="#264653")
    axes[0, 1].set_title("Approval rate (%) by Credit History")

    (df.groupby("Property_Area")[TARGET].mean() * 100).plot.bar(ax=axes[0, 2], color="#e9c46a")
    axes[0, 2].set_title("Approval rate (%) by Property Area")

    (df.groupby("Education")[TARGET].mean() * 100).plot.bar(ax=axes[1, 0], color="#f4a261")
    axes[1, 0].set_title("Approval rate (%) by Education")

    for status, label, color in [(1, "Approved", "#2a9d8f"), (0, "Rejected", "#e76f51")]:
        axes[1, 1].hist(np.log1p(df.loc[df[TARGET] == status, "ApplicantIncome"]), bins=30, alpha=0.6, label=label, color=color)
    axes[1, 1].set_title("log(Applicant Income) by status")
    axes[1, 1].legend()

    df.isna().sum().sort_values(ascending=False).head(8).plot.barh(ax=axes[1, 2], color="#8d99ae")
    axes[1, 2].set_title("Missing values (top 8)")

    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda.png"), dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# 4. Training & evaluation
# --------------------------------------------------------------------------- #
def train(data_path: str | None) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(data_path)
    print(f"[data] Approval rate: {df[TARGET].mean():.1%} | missing cells: {int(df.isna().sum().sum())}")
    save_eda_plots(df)

    X = add_features(df[RAW_FEATURES])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\n[cv] 5-fold cross-validation on the training set (ROC-AUC)")
    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(make_pipeline(model), X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = scores.mean()
        print(f"     {name:<20s} {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\n[tune] Best baseline: {best_name} -> running grid search")
    grids = {
        "Logistic Regression": {"model__C": [0.1, 1, 10]},
        "Random Forest": {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5]},
        "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
    }
    search = GridSearchCV(make_pipeline(candidates[best_name]), grids[best_name], cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"       Best params: {search.best_params_}")

    # ---- Final evaluation on the untouched test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": best_name,
        "best_params": search.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cv_roc_auc": cv_scores,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    print("\n[test] Hold-out results")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
    print(f"       ROC-AUC: {metrics['roc_auc']:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Rejected", "Approved"], ax=ax[0], cmap="Blues")
    ax[0].set_title("Confusion matrix")
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax[1])
    ax[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[1].set_title("ROC curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "evaluation.png"), dpi=130)
    plt.close()

    # ---- Feature importance
    importance = extract_importance(best_model)
    if importance is not None:
        top = importance.head(12)[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(top.index, top.values, color="#2a9d8f")
        plt.title(f"Top features ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=130)
        plt.close()
        metrics["top_features"] = importance.head(10).round(4).to_dict()
        print("\n[importance] Top features:")
        print(importance.head(8).round(4).to_string())

    joblib.dump(best_model, MODEL_PATH)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\n[done] Model saved to {MODEL_PATH}; plots + metrics.json in ./{OUTPUT_DIR}/")


def extract_importance(pipeline: Pipeline):
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return None
    return pd.Series(values, index=names).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 5. Prediction
# --------------------------------------------------------------------------- #
def predict_one(applicant: dict) -> dict:
    if not os.path.exists(MODEL_PATH):
        sys.exit("[error] No trained model found. Run: python loan_approval.py train")
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{col: applicant.get(col, np.nan) for col in RAW_FEATURES}])
    row = add_features(row)
    prob = float(model.predict_proba(row)[0, 1])
    return {
        "decision": "APPROVED" if prob >= 0.5 else "REJECTED",
        "approval_probability": round(prob, 4),
    }


def interactive_input() -> dict:
    print("Enter applicant details (press Enter to leave a field blank):")
    prompts = {
        "Gender": "Gender (Male/Female)",
        "Married": "Married (Yes/No)",
        "Dependents": "Dependents (0/1/2/3+)",
        "Education": "Education (Graduate/Not Graduate)",
        "Self_Employed": "Self employed (Yes/No)",
        "ApplicantIncome": "Applicant monthly income",
        "CoapplicantIncome": "Co-applicant monthly income",
        "LoanAmount": "Loan amount (in thousands)",
        "Loan_Amount_Term": "Loan term (months, e.g. 360)",
        "Credit_History": "Credit history (1 = good, 0 = bad)",
        "Property_Area": "Property area (Urban/Semiurban/Rural)",
    }
    data = {}
    for key, text in prompts.items():
        val = input(f"  {text}: ").strip()
        if val == "":
            continue
        data[key] = float(val) if key in NUMERIC else val
    return data


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Loan Approval Prediction System")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train and evaluate models")
    p_train.add_argument("--data", help="Path to CSV (optional; synthetic data used if omitted)")

    p_pred = sub.add_parser("predict", help="Predict for one applicant")
    p_pred.add_argument("--json", help="Applicant details as a JSON string")
    p_pred.add_argument("--interactive", action="store_true", help="Enter details via prompts")

    p_gen = sub.add_parser("generate-data", help="Write a synthetic dataset to CSV")
    p_gen.add_argument("--rows", type=int, default=3000)
    p_gen.add_argument("--out", default="loan_data.csv")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data)
    elif args.command == "predict":
        if args.json:
            applicant = json.loads(args.json)
        elif args.interactive:
            applicant = interactive_input()
        else:
            sys.exit("[error] Provide --json '{...}' or --interactive")
        print(json.dumps(predict_one(applicant), indent=2))
    elif args.command == "generate-data":
        generate_synthetic_data(args.rows).to_csv(args.out, index=False)
        print(f"Wrote {args.rows} rows to {args.out}")


if __name__ == "__main__":
    main()

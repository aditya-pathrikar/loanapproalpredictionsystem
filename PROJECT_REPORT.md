# Loan Approval Prediction System: Project Report

## 1. Abstract

Banks and lenders must decide quickly and consistently whether to approve a loan. Manual review is slow and can vary between reviewers. This project builds a machine learning system that predicts loan approval from applicant and loan attributes. Three classifiers (Logistic Regression, Random Forest and Gradient Boosting) were compared using cross-validation, the best was tuned with grid search, and it was evaluated on a held-out test set. On a 3,000-row dataset the final model reached **86.3% accuracy** and a **ROC-AUC of 0.896**. The trained model can be used through a command line tool to score a new applicant.

## 2. Introduction

### 2.1 Problem statement
Given an applicant's demographic, financial and loan details, predict whether the loan will be approved (`Y`) or rejected (`N`). This is a binary classification problem.

### 2.2 Objectives
1. Build a complete, reproducible ML pipeline for loan approval prediction.
2. Compare several algorithms and choose the best using cross-validation.
3. Evaluate the model honestly on unseen data.
4. Provide a simple interface to predict for a new applicant.

### 2.3 Scope
The project covers data preparation, modelling, evaluation and command line prediction. It does not include deployment, monitoring or regulatory compliance.

## 3. Dataset

The pipeline accepts any CSV in the format of the public Kaggle *Loan Prediction* dataset. When no file is given, it generates a **synthetic dataset of 3,000 applicants** with the same columns, so the project is runnable without downloads.

| Feature | Type | Description |
|---|---|---|
| Gender, Married, Dependents, Education, Self_Employed, Property_Area | Categorical | Applicant profile |
| ApplicantIncome, CoapplicantIncome | Numeric | Monthly income |
| LoanAmount | Numeric | Amount in thousands |
| Loan_Amount_Term | Numeric | Term in months |
| Credit_History | Binary | 1 if credit guidelines are met |
| Loan_Status | Target | Y / N |

**Synthetic data details.** Approval was generated from a hidden score combining credit history, EMI-to-income burden, education, marital status, property area, dependents, self-employment and income, plus random noise. The top 70% of scores were labelled approved (70% approval rate). About 729 cells (roughly 2% of feature cells) were blanked to mimic real missing data.

## 4. Methodology

### 4.1 Pipeline overview
Data loading → exploratory analysis → feature engineering → preprocessing → model comparison → tuning → evaluation → saving → prediction.

### 4.2 Feature engineering
| New feature | Formula | Rationale |
|---|---|---|
| TotalIncome | Applicant + Coapplicant income | Household repayment capacity |
| EMI | LoanAmount × 1000 / Term | Monthly instalment |
| BalanceIncome | TotalIncome − EMI | Income left after repayment |
| LoanToIncome | LoanAmount × 1000 / TotalIncome | Loan size relative to income |

### 4.3 Preprocessing
- Numeric columns: median imputation, then standardisation.
- Categorical columns: most-frequent imputation, then one-hot encoding (unknown categories ignored at prediction time).
- All steps live inside a scikit-learn `Pipeline`, so imputation and scaling are learned only from training folds. This prevents data leakage.

### 4.4 Models
| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline, interpretable, balanced class weights |
| Random Forest | 300 trees, balanced class weights |
| Gradient Boosting | Sequential boosted trees |

### 4.5 Training and validation
- 80/20 stratified train/test split (2,400 train, 600 test).
- 5-fold stratified cross-validation on the training set, scored by ROC-AUC.
- Grid search on the best model.
- The test set was used only once, for the final evaluation.

## 5. Results

### 5.1 Cross-validation (ROC-AUC, training set)
| Model | Mean ROC-AUC |
|---|---|
| **Logistic Regression** | **0.9249** |
| Gradient Boosting | 0.9188 |
| Random Forest | 0.9177 |

Logistic Regression was selected. Grid search chose `C = 1`.

### 5.2 Hold-out test performance (600 applicants)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Rejected | 0.78 | 0.76 | 0.77 | 180 |
| Approved | 0.90 | 0.91 | 0.90 | 420 |

| Overall metric | Value |
|---|---|
| Accuracy | 0.863 |
| ROC-AUC | 0.896 |

**Confusion matrix (approximate, derived from the reported metrics):** 382 approved correctly, 137 rejected correctly, 43 rejected applicants wrongly approved, 38 approved applicants wrongly rejected. The exact matrix is in `outputs/evaluation.png`.

### 5.3 Feature importance (absolute coefficients)
| Rank | Feature | Weight |
|---|---|---|
| 1 | Credit_History | 2.61 |
| 2 | EMI | 0.86 |
| 3 | BalanceIncome | 0.65 |
| 4 | Loan_Amount_Term | 0.58 |
| 5 | Education | 0.56 / 0.42 |
| 6 | Self_Employed | 0.40 |

Credit history dominates, followed by repayment burden (EMI and remaining income). The engineered features ranked highly, supporting the value of the feature engineering step.

### 5.4 Example predictions
| Applicant | Result |
|---|---|
| Graduate, married, income 6,000 + 2,000, loan 150k over 360 months, good credit, semiurban | Approved (97.4%) |
| Non-graduate, income 2,500, loan 200k, bad credit, rural | Rejected (0.2% approval probability) |

## 6. Discussion

- **Why Logistic Regression won.** The synthetic approval rule is close to linear in the engineered features, so a simple model fits it well. On real data, tree ensembles often do better, and the pipeline will pick whichever scores highest.
- **Class imbalance.** Rejected applicants are the minority (30%), and recall on that class (0.76) is lower than on approved applicants. A lender who fears defaults more than lost customers might raise the decision threshold or optimise for rejected-class recall.
- **Error costs.** Approving a bad loan (false positive) usually costs more than rejecting a good one. The 0.5 threshold can be tuned to reflect this.

## 7. Limitations

1. The default dataset is synthetic, so metrics demonstrate the pipeline and do not measure real-world accuracy.
2. Only a single train/test split is used for final evaluation.
3. No fairness analysis was performed. Attributes like gender and marital status can lead to discriminatory outcomes and are legally restricted in many jurisdictions.
4. The model gives probabilities but no per-applicant explanation.
5. No monitoring for data drift after deployment.

## 8. Future work

- Train and validate on a real, larger dataset.
- Add SHAP explanations so applicants can be told why they were rejected.
- Run fairness audits and consider removing sensitive attributes.
- Handle imbalance with SMOTE or cost-sensitive thresholds.
- Build a web interface (Streamlit or Flask) and deploy via an API.
- Add model monitoring and scheduled retraining.

## 9. Conclusion

The project delivers a working end-to-end loan approval predictor: a leak-free preprocessing pipeline, systematic model comparison, tuning, honest evaluation and a prediction interface. The final Logistic Regression model reached 86.3% accuracy and 0.896 ROC-AUC on held-out data, with credit history and repayment burden as the main drivers. With a real dataset, fairness checks and explainability, the same structure could form the basis of a decision-support tool.

## 10. Technologies used

Python 3.10+, pandas, NumPy, scikit-learn, matplotlib, joblib.

## 11. How to run

See `README.md`: `pip install -r requirements.txt`, then `python loan_approval.py train`.

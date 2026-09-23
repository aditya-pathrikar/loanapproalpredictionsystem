# Loan Approval Prediction System

A machine learning project that predicts whether a loan application will be **approved or rejected** from applicant details such as income, credit history, loan amount and property area.

The full pipeline is in one file, `loan_approval.py`: data loading, exploratory plots, feature engineering, model comparison, hyperparameter tuning, evaluation, model saving and single-applicant prediction.

## Project structure

```
loan_approval_project/
├── loan_approval.py      # complete pipeline + command line interface
├── requirements.txt      # Python dependencies
├── README.md             # this file
├── PROJECT_REPORT.md     # full project report
└── outputs/              # created after training
    ├── loan_model.joblib       # saved model pipeline
    ├── metrics.json            # evaluation metrics
    ├── eda.png                 # exploratory plots
    ├── evaluation.png          # confusion matrix + ROC curve
    └── feature_importance.png  # top features
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Train
```bash
python loan_approval.py train
```
With no data file, the script generates a synthetic dataset of 3,000 applicants so the project runs out of the box.

To use a real dataset (for example the Kaggle *Loan Prediction* dataset):
```bash
python loan_approval.py train --data train.csv
```

### 2. Predict for one applicant
```bash
python loan_approval.py predict --json '{"Gender":"Male","Married":"Yes","Dependents":"1","Education":"Graduate","Self_Employed":"No","ApplicantIncome":6000,"CoapplicantIncome":2000,"LoanAmount":150,"Loan_Amount_Term":360,"Credit_History":1,"Property_Area":"Semiurban"}'
```
Output:
```json
{ "decision": "APPROVED", "approval_probability": 0.9736 }
```
Or answer prompts one by one:
```bash
python loan_approval.py predict --interactive
```
Fields you leave out are treated as missing and filled in automatically by the pipeline.

### 3. Export the synthetic dataset
```bash
python loan_approval.py generate-data --rows 5000 --out loan_data.csv
```

## Input columns

| Column | Description |
|---|---|
| Gender | Male / Female |
| Married | Yes / No |
| Dependents | 0, 1, 2, 3+ |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Monthly income of applicant |
| CoapplicantIncome | Monthly income of co-applicant |
| LoanAmount | Loan amount in thousands |
| Loan_Amount_Term | Term in months (e.g. 360) |
| Credit_History | 1 = meets guidelines, 0 = does not |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target: Y (approved) / N (rejected) |

## How it works

1. **Feature engineering**: adds `TotalIncome`, `EMI`, `BalanceIncome` and `LoanToIncome`.
2. **Preprocessing**: median imputation and scaling for numbers; mode imputation and one-hot encoding for categories.
3. **Model comparison**: Logistic Regression, Random Forest and Gradient Boosting, compared by 5-fold cross-validated ROC-AUC.
4. **Tuning**: grid search on the best model.
5. **Evaluation**: accuracy, precision, recall, F1 and ROC-AUC on a held-out 20% test set.

## Results (synthetic data, 3,000 rows)

| Metric | Score |
|---|---|
| Accuracy | 0.863 |
| Precision (approved) | 0.897 |
| Recall (approved) | 0.910 |
| F1 (approved) | 0.903 |
| ROC-AUC | 0.896 |

Because the default data is synthetic, these numbers show the pipeline works; they are not a claim about real-world lending performance. Train on real data for real results.

## Limitations and responsible use

- This is an educational project, not a production credit-scoring system.
- Real lending decisions are regulated. Models must be audited for bias (e.g. on gender, marital status, location) and be explainable to applicants.
- Sensitive attributes are included here to match common public datasets; a real system should evaluate whether to use them at all.

## Possible extensions

- Web app with Streamlit or Flask
- SHAP explanations per applicant
- Handling class imbalance with SMOTE
- Fairness metrics across demographic groups

# XGBoost — Bank Customer Churn Prediction

Binary classification on the **Churn_Modelling** dataset using XGBoost, implemented in both Python and R with k-Fold Cross Validation.

## What It Does

Predicts whether a bank customer will leave (churn) based on features like credit score, geography, gender, age, tenure, balance, and more. The model is evaluated with a confusion matrix and 10-fold cross-validation accuracy.

## Dataset

| Field | Description |
|---|---|
| Source | `Churn_Modelling.csv` (10 000 rows, 14 columns) |
| Target | `Exited` — 1 if the customer left, 0 otherwise |
| Features used | Columns 4–13 (CreditScore … EstimatedSalary) |

## 🛠 Tech Stack

| | Tool | Role |
|---|---|---|
| 🐍 | Python 3.10+ | Main implementation |
| 📊 | scikit-learn | Preprocessing, cross-validation |
| 🚀 | XGBoost | Gradient-boosted tree classifier |
| 📈 | pandas / NumPy | Data wrangling |
| 🇷 | R | Alternative implementation |
| 📦 | caret / caTools | R model evaluation utilities |

## Getting Started

### Python

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
python xgboost.py
```

### R

```r
install.packages(c("caTools", "xgboost", "caret"))
source("xgboost.R")
```

## Bugs Fixed in This Revision

| File | Bug | Fix |
|---|---|---|
| `xgboost.py` | `OneHotEncoder(categorical_features=...)` removed in sklearn ≥ 1.0 | Replaced with `ColumnTransformer` + `OneHotEncoder(drop="first")` |
| `xgboost.py` | Imports scattered throughout file | Moved all imports to top (PEP 8) |
| `xgboost.py` | No entry-point guard | Added `if __name__ == "__main__"` |
| `xgboost.py` | Hardcoded CSV path | Uses `os.path` relative to script |
| `xgboost.py` | CV accuracy computed but never printed | Added formatted print output |
| `xgboost.R` | CV loop trained on full `training_set` instead of `training_fold` | Fixed to use `training_fold` |
| `xgboost.R` | Hardcoded CSV path | Uses `file.path()` relative to script |

## ⚠️ Known Issues

- The R script uses `sys.frame(1)$ofile` for path resolution, which only works when the script is `source()`'d — running line-by-line in RStudio will fall back to the working directory.
- The dataset file (`Churn_Modelling.csv`) must be in the same directory as the scripts.

## License

[MIT](LICENSE)

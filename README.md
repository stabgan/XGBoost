# 🚀 XGBoost Churn Prediction

Binary classification on a bank customer churn dataset using XGBoost, implemented in both Python and R. The model predicts whether a customer will leave the bank based on demographic and account features, validated with 10-fold cross-validation.

## 📋 Methodology

1. Load the `Churn_Modelling.csv` dataset (10,000 bank customers, 11 features)
2. Encode categorical variables (Geography, Gender) using label encoding + one-hot encoding
3. Split data 80/20 into training and test sets
4. Train an XGBoost classifier with binary logistic objective
5. Evaluate with confusion matrix and 10-fold cross-validation

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Primary implementation |
| 📊 R 4.0+ | Alternative implementation |
| 🌲 XGBoost | Gradient boosting classifier |
| 🔬 scikit-learn | Preprocessing, encoding, cross-validation |
| 🐼 pandas / NumPy | Data manipulation |
| 📈 matplotlib | Plotting (available for extensions) |
| 📦 caret / caTools | R-side splitting and CV |

## 📦 Dependencies

### Python

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

### R

```r
install.packages(c("caTools", "xgboost", "caret"))
```

## ▶️ How to Run

### Python

```bash
python xgboost.py
```

### R

```bash
Rscript xgboost.R
```

Both scripts expect `Churn_Modelling.csv` in the same directory.

## ⚠️ Known Issues

- The dataset path is hardcoded — the CSV must be in the working directory.
- No hyperparameter tuning is performed; default XGBoost settings are used with `nrounds = 10`.
- One-hot encoding via `ColumnTransformer` returns a dense array, which may be memory-heavy on larger datasets.
- No feature scaling is applied (XGBoost is tree-based and generally doesn't require it, but stacking with other models may).

## 📄 License

See [LICENSE](LICENSE) for details.

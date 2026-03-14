# XGBoost — Churn Prediction (R)
# Trains an XGBoost classifier with k-Fold Cross Validation.

# ── Load dataset ────────────────────────────────────────────────────────
script_dir <- dirname(sys.frame(1)$ofile %||% ".")
dataset <- read.csv(file.path(script_dir, "Churn_Modelling.csv"))
dataset <- dataset[4:14]

# ── Encode categorical variables as numeric factors ─────────────────────
dataset$Geography <- as.numeric(factor(dataset$Geography,
                                       levels = c("France", "Spain", "Germany"),
                                       labels = c(1, 2, 3)))
dataset$Gender <- as.numeric(factor(dataset$Gender,
                                    levels = c("Female", "Male"),
                                    labels = c(1, 2)))

# ── Train / Test split ─────────────────────────────────────────────────
library(caTools)
set.seed(123)
split <- sample.split(dataset$Exited, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set     <- subset(dataset, split == FALSE)

# ── Fit XGBoost ─────────────────────────────────────────────────────────
library(xgboost)
classifier <- xgboost(data    = as.matrix(training_set[-11]),
                      label   = training_set$Exited,
                      nrounds = 10)

# ── Predict & evaluate ─────────────────────────────────────────────────
y_pred <- predict(classifier, newdata = as.matrix(test_set[-11]))
y_pred <- (y_pred >= 0.5)
cm <- table(test_set[, 11], y_pred)
cat("Confusion Matrix:\n")
print(cm)

# ── k-Fold Cross Validation ────────────────────────────────────────────
library(caret)
folds <- createFolds(training_set$Exited, k = 10)
cv <- lapply(folds, function(x) {
  training_fold <- training_set[-x, ]
  test_fold     <- training_set[x, ]
  # FIX: train on training_fold, NOT the full training_set
  fold_model <- xgboost(data    = as.matrix(training_fold[-11]),
                        label   = training_fold$Exited,
                        nrounds = 10,
                        verbose = 0)
  y_pred <- predict(fold_model, newdata = as.matrix(test_fold[-11]))
  y_pred <- (y_pred >= 0.5)
  fold_cm <- table(test_fold[, 11], y_pred)
  accuracy <- (fold_cm[1, 1] + fold_cm[2, 2]) /
              (fold_cm[1, 1] + fold_cm[2, 2] + fold_cm[1, 2] + fold_cm[2, 1])
  return(accuracy)
})
accuracy <- mean(as.numeric(cv))
cat(sprintf("\nk-Fold CV Accuracy: %.4f\n", accuracy))

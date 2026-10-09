# NumPy House Price Regression

An end-to-end house-price regressor written in **pure NumPy**, with no scikit-learn or pandas. Starting from a raw numeric feature matrix (with missing values and outliers), the pipeline cleans the data, engineers features, splits it reproducibly, standardizes it without leakage, fits an ordinary least squares (OLS) model, and evaluates it on held-out homes.

Every step is a small, testable function. The goal is to understand what libraries like scikit-learn do under the hood.

## Pipeline

```
raw X, y
  │
  ├─ 1. Clean        impute NaNs with column means → clip outliers with the IQR rule
  ├─ 2. Features     add a ratio feature (e.g. rooms / households) → optional one-hot categories
  ├─ 3. Split        reproducible shuffle → train / validation / test
  ├─ 4. Scale        fit mean/std on TRAIN only → apply to all splits → prepend bias column
  ├─ 5. Fit          OLS on the training set (least-squares solver)
  └─ 6. Evaluate     MAE, RMSE, R², residual summary on validation and test
```

## Project structure

| File | Contents |
|------|----------|
| `model.py` | All 24 pipeline functions, assembled from the step-by-step solutions |
| `scaffold.py` | The functions plus a runnable demo on synthetic housing data |
| `docs/` | Supporting documentation |

## Quick start

Requirements: Python 3.9+ and NumPy.

```powershell
# Windows (PowerShell)
git clone https://github.com/sadokaminbenyahia/numpy-house-price-regression.git
cd numpy-house-price-regression
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy
python scaffold.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

The demo generates 200 synthetic houses (rooms, households, age, income, plus a district label A/B/C) with injected NaNs and outliers. It then runs the full pipeline and prints the test metrics, a residual summary, and the first few predictions next to the true prices.

## Usage

```python
import numpy as np
from model import house_price_pipeline

result = house_price_pipeline(
    X, y,
    ratio_num_idx=0,        # column used as the ratio's numerator
    ratio_den_idx=1,        # column used as the ratio's denominator
    cat_labels=districts,   # optional 1-D array of categories, or None
    train_ratio=0.7,
    val_ratio=0.15,         # the rest (0.15) goes to the test set
    seed=42,
    iqr_k=1.5,              # IQR multiplier for outlier clipping
)

result["theta"]          # fitted weights, shape (D,), bias first
result["y_test"]         # held-out targets
result["y_test_pred"]    # held-out predictions
result["test_metrics"]   # {'mae', 'rmse', 'r2', 'residual_summary'}
result["val_metrics"]    # same keys, on the validation split
```

## Functions

| # | Function | Purpose |
|---|----------|---------|
| 1 | `impute_nan_with_mean` | Replace NaNs with each column's mean (all-NaN columns become 0) |
| 2 | `compute_iqr_bounds` | Per-column bounds `[Q1 − k·IQR, Q3 + k·IQR]` |
| 3 | `clip_columns` | Clip each column to its bounds |
| 4 | `make_ratio_feature` | Safe division `num / (den + eps)` |
| 5 | `append_column` | Append a feature column to the matrix |
| 6 | `one_hot_encode` | Labels to a dense binary matrix (vectorized with `np.unique`) |
| 7 | `fit_standardizer` | Per-column mean and std (zero std replaced by 1) |
| 8 | `apply_standardizer` | `(X − mean) / std` via broadcasting |
| 9 | `add_bias_column` | Prepend a column of ones |
| 10 | `make_shuffled_indices` | Seeded permutation of row indices |
| 11 | `partition_indices` | Split indices into train / val / test |
| 12 | `subset_xy` | Select matching rows of `X` and `y` |
| 13 | `ols_fit` | Least-squares weights (`np.linalg.lstsq`) |
| 14 | `ols_predict` | `X @ theta` |
| 15 | `mean_absolute_error` | MAE |
| 16 | `root_mean_squared_error` | RMSE |
| 17 | `r_squared` | R² (0.0 when the target has zero variance) |
| 18 | `residual_summary` | Mean, std, and median absolute residual |
| 19 | `prepare_cleaned_features` | Impute, then clip |
| 20 | `assemble_feature_matrix` | Numeric features + ratio + optional one-hots |
| 21 | `make_train_val_test` | Shuffle and materialize the three splits |
| 22 | `standardize_and_add_bias` | Train-only standardization + bias column |
| 23 | `evaluate_predictions` | All metrics in one dict |
| 24 | `house_price_pipeline` | End-to-end entry point |

## Design notes and lessons learned

- **No data leakage in scaling.** The standardization mean and std are computed on the training split only, then reused for validation and test, exactly as they would be for new houses in production. (Imputation and IQR clipping run before the split, as the exercise specified; a stricter version would fit those on train only too.)
- **`lstsq` instead of the normal equation.** Solving `XᵀX θ = Xᵀy` with `np.linalg.solve` fails when features are collinear: duplicated or shifted columns, or a full one-hot block plus a bias column (the "dummy variable trap"), make `XᵀX` singular. `np.linalg.lstsq` returns the minimum-norm least-squares solution, which still fits perfectly when the target is in the column space.
- **Reproducible splits.** Shuffling uses `np.random.RandomState(seed)`. The newer `np.random.default_rng(seed)` produces a *different* permutation for the same seed, so the generator must match whatever produced the reference results.
- **Small splits make metrics unreliable.** With a single validation sample, R² is undefined (zero variance) and is reported as 0.0.

## Possible extensions

- Fit imputation and clipping statistics on the training split only
- Batch gradient descent as an alternative to the closed-form solver, with a loss curve
- Ridge regularization: `θ = (XᵀX + λI)⁻¹ Xᵀy`
- Run the pipeline on the real California Housing dataset and compare with scikit-learn's `LinearRegression`.

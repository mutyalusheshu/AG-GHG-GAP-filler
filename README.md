# GHG Gap Filler — Universal ML Gap-Filling Shiny App

A point-and-click R Shiny application for filling gaps in any daily flux or
environmental measurement dataset using 7 machine learning models.

---

## Quick start

```r
# 1. Install packages (once)
source("GHGGapFiller/install_packages.R")

# 2. Launch
shiny::runApp("GHGGapFiller/app.R")
```

---

## What it does

| Step | Action |
|------|--------|
| Upload | Upload any CSV file |
| Configure | Pick the target column (the one with gaps), predictor columns, and optional date column |
| Select models | Choose from 7 ML models |
| Run | 5-fold cross-validation → metrics + gap-filling of all missing rows |
| Download | Gap-filled CSV or Excel report |

---

## Models included

| Key | Model | Package | Notes |
|-----|-------|---------|-------|
| RF | Random Forest | `ranger` | 500 trees, permutation importance |
| XGB | XGBoost | `xgboost` | 300 rounds, depth=5, lr=0.05 |
| GBM | Gradient Boosting | `gbm` | 200 trees, depth=4 |
| KNN | k-NN | `kknn` | k=7, weighted |
| ENET | Elastic Net | `glmnet` | alpha=0.5, CV lambda |
| SVM | SVM Radial | `e1071` | RBF kernel, scaled |
| CUBIST | Cubist | `Cubist` | 10 committees |
| TABPFN | TabPFN | Python | Optional — see below |

---

## Using TabPFN (optional)

TabPFN requires Python + the `tabpfn-client` library.

```bash
# Install Python package
pip install tabpfn-client
```

```r
# Install reticulate
install.packages("reticulate")

# Authenticate (get token at https://priorlabs.ai)
tabpfn_client$set_access_token("YOUR_TOKEN")
```

Then check the **TabPFN (Python)** box in the app.

---

## Output

### Gap-filled CSV
Original data + one new column per model named `{target}_{MODEL}`.
Observed rows keep their original value; gap rows get the model prediction.

### Excel report (4 sheets)
1. `Summary` — CV metrics for all models (RMSE, R², MAE, Bias, PBIAS, ΔASum, Rank)
2. `Gap_Filled_Data` — full dataset with filled columns
3. `Variable_Importance` — importance (%) per feature per model
4. `Settings` — run parameters

---

## Metrics explained

| Metric | Meaning |
|--------|---------|
| RMSE | Root mean squared error (original units) |
| MAE | Mean absolute error |
| Bias | Mean systematic over/under-prediction |
| R² | Fraction of variance explained (1 = perfect) |
| NSE | Nash-Sutcliffe Efficiency (= R² here) |
| PBIAS | % systematic bias in annual total |
| ΔASum | Annual budget offset = Bias × N_gaps (Taki 2019 Eq.6) |
| Rank | Equal-weight mean rank of RMSE, \|Bias\|, R², NSE |

---

## Data requirements

- **Any CSV file** — no fixed column names required
- Target column can be any numeric column with NA values
- Predictor columns can be numeric or categorical (categoricals are auto-encoded)
- Date column is optional — used for the time series plot only
- Rows with missing predictor values are dropped before training

---

## Troubleshooting

**"Need at least 10 observed rows"** — The target column has fewer than 10
non-missing values. Models cannot train on so little data.

**Model fails / NA predictions** — Some models fail on certain data
configurations (e.g. SVM with constant features). The app reports NA for
that model and continues with the others.

**Slow runtime** — GBM and kNN are slowest. Uncheck them for a quick first run.
XGBoost and RF are typically the fastest and best-performing.

---

## Citation

If you use this tool in a publication, please cite:

> Machine Learning-Based GHG Gap-Filling. BCSE_KBS Agricultural N₂O Study.
> April 2026. Built with R Shiny, ranger, xgboost, gbm, kknn, glmnet, e1071, Cubist.

---

## File structure

```
GHGGapFiller/
├── app.R               ← Main Shiny application (run this)
├── helpers.R           ← Model training, CV, importance functions
├── install_packages.R  ← One-time package installer
└── README.md           ← This file
```

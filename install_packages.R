# install_packages.R — Run this once before launching the app
# install.packages("pak") # faster installer (optional)

pkgs <- c(
  # Shiny UI
  "shiny", "shinydashboard", "shinyWidgets",
  "DT", "plotly", "writexl",

  # Data
  "dplyr", "tidyr",

  # ML models
  "ranger",    # Random Forest
  "xgboost",   # XGBoost
  "gbm",       # Gradient Boosting
  "kknn",      # k-NN
  "glmnet",    # Elastic Net / Ridge
  "e1071",     # SVM
  "Cubist",     # Cubist
  "reticulate"  #python
)

missing <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(missing)) {
  message("Installing ", length(missing), " package(s): ", paste(missing, collapse=", "))
  install.packages(missing, dependencies = TRUE)
} else {
  message("All packages already installed.")
}

# Optional: TabPFN via Python
# Requires Python 3.9+ and tabpfn-client:
#   pip install tabpfn-client
# Then in R:
install.packages("reticulate")
reticulate::py_install("tabpfn-client")

message("Done. Run: shiny::runApp('GHGGapFiller/app.R')")

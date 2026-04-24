# =============================================================================
# helpers.R — Universal GHG Gap-Filling Engine
# Models: RF, XGBoost, GBM, kNN, Elastic Net, SVM, Cubist + optional TabPFN
# =============================================================================

suppressPackageStartupMessages({
  library(ranger);  library(xgboost); library(gbm)
  library(kknn);    library(glmnet);  library(e1071)
  library(Cubist);  library(dplyr)
})

# ── Model catalogue ────────────────────────────────────────────────────────────
MODEL_LABELS <- c(
  RF     = "Random Forest",
  XGB    = "XGBoost",
  GBM    = "Gradient Boosting",
  KNN    = "k-NN",
  ENET   = "Elastic Net",
  SVM    = "SVM (Radial)",
  CUBIST = "Cubist",
  TABPFN = "TabPFN (Python)"
)

MODEL_COLORS <- c(
  RF     = "#1B4332", XGB    = "#D62728", GBM    = "#2D6A4F",
  KNN    = "#854F0B", ENET   = "#9467BD", SVM    = "#185FA5",
  CUBIST = "#FF7F0E", TABPFN = "#E377C2"
)

# ── 1. Metrics ─────────────────────────────────────────────────────────────────
compute_metrics <- function(obs, pred) {
  ok <- !is.na(obs) & !is.na(pred)
  o  <- obs[ok]; p <- pred[ok]; n <- length(o)
  if (n < 3)
    return(data.frame(N=n, RMSE=NA_real_, MAE=NA_real_, Bias=NA_real_,
                      R2=NA_real_, NSE=NA_real_, PBIAS=NA_real_,
                      DeltaASum=NA_real_, stringsAsFactors=FALSE))
  res    <- p - o
  ss_res <- sum(res^2); ss_tot <- sum((o - mean(o))^2)
  r2     <- if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_
  pbias  <- if (abs(sum(o)) > 0) 100 * sum(res) / sum(o) else NA_real_
  data.frame(
    N         = n,
    RMSE      = round(sqrt(ss_res / n), 4),
    MAE       = round(mean(abs(res)), 4),
    Bias      = round(mean(res), 4),
    R2        = round(r2, 4),
    NSE       = round(r2, 4),
    PBIAS     = round(pbias, 2),
    DeltaASum = round(mean(res) * n, 2),
    stringsAsFactors = FALSE
  )
}

# ── 2. Permutation importance (for SVM, kNN, Cubist) ──────────────────────────
perm_importance <- function(predict_fn, X, y, n_repeats = 5) {
  n_samp   <- min(300L, nrow(X))
  idx      <- sample(nrow(X), n_samp)
  Xs <- X[idx, , drop = FALSE]; ys <- y[idx]
  base_r   <- sqrt(mean((predict_fn(Xs) - ys)^2, na.rm = TRUE))
  scores   <- numeric(ncol(X))
  for (fi in seq_len(ncol(X))) {
    d <- vapply(seq_len(n_repeats), function(r) {
      Xp <- Xs; Xp[, fi] <- sample(Xp[, fi])
      sqrt(mean((predict_fn(Xp) - ys)^2, na.rm = TRUE)) - base_r
    }, numeric(1))
    scores[fi] <- max(0, mean(d, na.rm = TRUE))
  }
  scores
}

# ── 3. Train a single model ────────────────────────────────────────────────────
train_model <- function(X, y, model_key) {
  X  <- as.data.frame(X); df <- cbind(X, .y. = y)
  switch(model_key,
    RF = ranger::ranger(.y. ~ ., data = df, num.trees = 500,
                        importance = "permutation", seed = 42, num.threads = 1),
    XGB = {
      d <- xgboost::xgb.DMatrix(as.matrix(X), label = y)
      xgboost::xgb.train(
        params  = list(objective = "reg:squarederror", max_depth = 5,
                       eta = 0.05, subsample = 0.8, colsample_bytree = 0.8,
                       nthread = 1),
        data    = d, nrounds = 300, verbose = 0, seed = 42
      )
    },
    GBM = gbm::gbm(.y. ~ ., data = df, distribution = "gaussian",
                   n.trees = 200, interaction.depth = 4,
                   shrinkage = 0.05, n.minobsinnode = 5,
                   verbose = FALSE, n.cores = 1),
    KNN = list(X_tr = X, y_tr = y, k = 7L),   # lazy model: store data
    ENET = {
      nf <- min(5L, max(3L, floor(nrow(X) / 10)))
      glmnet::cv.glmnet(as.matrix(X), y, alpha = 0.5, nfolds = nf)
    },
    SVM = e1071::svm(.y. ~ ., data = df, type = "eps-regression",
                     kernel = "radial", scale = TRUE, cost = 1),
    CUBIST = Cubist::cubist(X, y, committees = 10),
    TABPFN = run_tabpfn_fit(X, y),
    stop("Unknown model key: ", model_key)
  )
}

# ── 4. Predict with a trained model ───────────────────────────────────────────
predict_model <- function(fit, X_new, model_key) {
  X_new <- as.data.frame(X_new)
  tryCatch(switch(model_key,
    RF     = predict(fit, data = X_new)$predictions,
    XGB    = predict(fit, xgboost::xgb.DMatrix(as.matrix(X_new))),
    GBM    = predict(fit, newdata = X_new, n.trees = 200),
    KNN    = kknn::kknn(.y. ~ .,
                        train = cbind(fit$X_tr, .y. = fit$y_tr),
                        test  = X_new, k = fit$k)$fitted.values,
    ENET   = as.numeric(predict(fit, newx = as.matrix(X_new), s = "lambda.min")),
    SVM    = as.numeric(predict(fit, newdata = X_new)),
    CUBIST = predict(fit, newdata = X_new),
    TABPFN = run_tabpfn_predict(fit, X_new),
    stop("Unknown model key: ", model_key)
  ),
  error = function(e) {
    warning("Predict failed [", model_key, "]: ", conditionMessage(e))
    rep(NA_real_, nrow(X_new))
  })
}

# ── 5. Variable importance ─────────────────────────────────────────────────────
get_importance <- function(fit, X, y, model_key, feature_names) {
  raw <- tryCatch(switch(model_key,
    RF = {
      imp <- fit$variable.importance
      imp[feature_names]
    },
    XGB = {
      imp_df <- xgboost::xgb.importance(model = fit)
      scores <- setNames(numeric(length(feature_names)), feature_names)
      for (i in seq_len(nrow(imp_df))) {
        f <- as.character(imp_df$Feature[i])
        if (f %in% feature_names) scores[f] <- imp_df$Gain[i]
      }
      scores
    },
    GBM = {
      smry <- summary(fit, plotit = FALSE)
      scores <- setNames(numeric(length(feature_names)), feature_names)
      for (i in seq_len(nrow(smry))) {
        v <- as.character(smry$var[i])
        if (v %in% feature_names) scores[v] <- smry$rel.inf[i]
      }
      scores
    },
    ENET = abs(as.numeric(coef(fit, s = "lambda.min"))[-1]),
    SVM = , KNN = , CUBIST = , TABPFN =
      perm_importance(
        function(Xp) predict_model(fit, as.data.frame(Xp), model_key),
        as.matrix(X), y
      ),
    numeric(length(feature_names))
  ),
  error = function(e) numeric(length(feature_names)))

  raw   <- as.numeric(raw)
  raw[is.na(raw)] <- 0
  total <- sum(raw)
  normed <- if (total > 0) raw / total * 100 else raw

  data.frame(Feature    = feature_names,
             Importance = round(normed, 2),
             stringsAsFactors = FALSE) |>
    dplyr::arrange(dplyr::desc(Importance))
}

# ── 6. 5-fold CV ───────────────────────────────────────────────────────────────
run_cv <- function(X, y, model_key, n_folds = 5L) {
  set.seed(42)
  n    <- length(y)
  idx  <- sample(n)
  fold <- ceiling(seq_along(idx) / n * n_folds)
  oof  <- rep(NA_real_, n)

  for (f in seq_len(n_folds)) {
    tr  <- idx[fold != f]; te <- idx[fold == f]
    fit <- tryCatch(train_model(X[tr, , drop = FALSE], y[tr], model_key),
                    error = function(e) {
                      warning("CV fold ", f, " [", model_key, "] failed: ", conditionMessage(e))
                      NULL
                    })
    if (!is.null(fit))
      oof[te] <- predict_model(fit, X[te, , drop = FALSE], model_key)
  }
  oof
}

# ── 7. TabPFN via reticulate (optional) ───────────────────────────────────────
run_tabpfn_fit <- function(X, y) {
  if (!requireNamespace("reticulate", quietly = TRUE))
    stop("TabPFN requires the 'reticulate' R package and Python tabpfn-client.")
  py <- reticulate::import("tabpfn_client")
  model <- py$TabPFNRegressor(n_estimators = 8L)
  model$fit(as.matrix(X), y)
  list(model = model, py = py)
}

run_tabpfn_predict <- function(fit_obj, X_new) {
  as.numeric(fit_obj$model$predict(as.matrix(X_new)))
}

# ── 8. Main gap-filling runner ─────────────────────────────────────────────────
#
# obs_df          — data.frame of rows where target is NOT NA
# gap_df          — data.frame of rows where target IS NA
# target_col      — name of the target column
# predictor_cols  — character vector of predictor column names
# selected_models — character vector of model keys from MODEL_LABELS
# n_folds         — number of CV folds (default 5)
# progress_fn     — optional function(value_0_to_1, message) for Shiny progress
#
run_gap_filling <- function(obs_df, gap_df, target_col, predictor_cols,
                             selected_models, n_folds = 5L,
                             progress_fn = NULL) {

  X_obs  <- as.matrix(obs_df[, predictor_cols, drop = FALSE])
  y_obs  <- as.numeric(obs_df[[target_col]])
  X_gap  <- if (nrow(gap_df) > 0)
               as.matrix(gap_df[, predictor_cols, drop = FALSE])
             else
               matrix(nrow = 0L, ncol = length(predictor_cols),
                      dimnames = list(NULL, predictor_cols))

  n_steps  <- length(selected_models) * 2L  # CV + final train each
  cur_step <- 0L

  tick <- function(label) {
    cur_step <<- cur_step + 1L
    if (!is.null(progress_fn))
      progress_fn(cur_step / n_steps, label)
    message("  [", cur_step, "/", n_steps, "] ", label)
  }

  results <- vector("list", length(selected_models))
  names(results) <- selected_models

  for (mk in selected_models) {
    lbl <- MODEL_LABELS[mk]
    if (is.na(lbl)) lbl <- mk
    t0  <- proc.time()[["elapsed"]]

    # ── CV ──────────────────────────────────────────────────────────────────
    tick(paste("CV:", lbl))
    oof <- tryCatch(
      run_cv(X_obs, y_obs, mk, n_folds),
      error = function(e) {
        warning("CV failed [", mk, "]: ", conditionMessage(e))
        rep(NA_real_, nrow(obs_df))
      }
    )
    cv_m <- compute_metrics(y_obs, oof)

    # ── Final model ──────────────────────────────────────────────────────────
    tick(paste("Fitting final model:", lbl))
    final_fit <- tryCatch(
      train_model(X_obs, y_obs, mk),
      error = function(e) {
        warning("Final train failed [", mk, "]: ", conditionMessage(e))
        NULL
      }
    )

    # ── Gap predictions ──────────────────────────────────────────────────────
    gap_preds <- if (!is.null(final_fit) && nrow(X_gap) > 0)
      tryCatch(predict_model(final_fit, X_gap, mk),
               error = function(e) rep(NA_real_, nrow(X_gap)))
    else rep(NA_real_, nrow(X_gap))

    # ── Importance ───────────────────────────────────────────────────────────
    imp <- if (!is.null(final_fit))
      tryCatch(get_importance(final_fit, X_obs, y_obs, mk, predictor_cols),
               error = function(e)
                 data.frame(Feature = predictor_cols, Importance = 0,
                            stringsAsFactors = FALSE))
    else data.frame(Feature = predictor_cols, Importance = 0,
                    stringsAsFactors = FALSE)

    elapsed <- round(proc.time()[["elapsed"]] - t0, 1)

    results[[mk]] <- list(
      label      = lbl,
      cv_preds   = oof,
      metrics    = cbind(data.frame(Model = lbl, Model_Key = mk,
                                     Time_s = elapsed,
                                     stringsAsFactors = FALSE), cv_m),
      gap_preds  = gap_preds,
      importance = imp
    )

    message(sprintf("    RMSE=%.3f  R\u00b2=%.3f  (%.1fs)", cv_m$RMSE, cv_m$R2, elapsed))
  }

  # ── Composite rank ────────────────────────────────────────────────────────
  metrics_all <- do.call(rbind, lapply(results, `[[`, "metrics"))
  rownames(metrics_all) <- NULL

  if (nrow(metrics_all) > 0 && !all(is.na(metrics_all$RMSE))) {
    metrics_all$rk_RMSE <- rank(metrics_all$RMSE,       ties.method = "average", na.last = "keep")
    metrics_all$rk_Bias <- rank(abs(metrics_all$Bias),  ties.method = "average", na.last = "keep")
    metrics_all$rk_R2   <- rank(-metrics_all$R2,        ties.method = "average", na.last = "keep")
    metrics_all$rk_NSE  <- rank(-metrics_all$NSE,       ties.method = "average", na.last = "keep")
    metrics_all$Rank <- round(
      rowMeans(metrics_all[, c("rk_RMSE","rk_Bias","rk_R2","rk_NSE")], na.rm = TRUE), 2)
    drop_cols <- c("rk_RMSE","rk_Bias","rk_R2","rk_NSE")
    metrics_all <- metrics_all[, !names(metrics_all) %in% drop_cols]
    metrics_all <- metrics_all[order(metrics_all$Rank), ]
  }

  list(
    results        = results,
    metrics_all    = metrics_all,
    obs_df         = obs_df,
    gap_df         = gap_df,
    X_obs          = X_obs,
    y_obs          = y_obs,
    target_col     = target_col,
    predictor_cols = predictor_cols
  )
}

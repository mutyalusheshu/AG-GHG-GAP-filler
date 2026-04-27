# =============================================================================
# app.R — Universal GHG Gap-Filler
# Shiny application for gap-filling any flux / soil dataset
# with 7 ML models + optional TabPFN
#
# Run:  shiny::runApp("GHGGapFiller/app.R")
# =============================================================================

# ── Auto-install missing packages ─────────────────────────────────────────────
required_pkgs <- c(
  "shiny", "shinydashboard", "shinyWidgets",
  "DT", "plotly", "writexl",
  "dplyr", "tidyr",
  "ranger", "xgboost", "gbm", "kknn",
  "glmnet", "e1071", "Cubist"
)
new_pkgs <- required_pkgs[!required_pkgs %in% rownames(installed.packages())]
if (length(new_pkgs)) {
  message("Installing: ", paste(new_pkgs, collapse = ", "))
  install.packages(new_pkgs, dependencies = TRUE, repos = "https://cloud.r-project.org")
}
invisible(lapply(required_pkgs, library, character.only = TRUE, warn.conflicts = FALSE))

source("helpers.R")  # helpers must be in same folder



# Fallback if not in RStudio
`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !is.na(a) && a != "") a else b
if (!exists("MODEL_LABELS")) source("helpers.R")

# ── Colour map for plotly ──────────────────────────────────────────────────────
CLRS <- c("#1B4332","#D62728","#2D6A4F","#854F0B",
          "#9467BD","#185FA5","#FF7F0E","#E377C2")

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
ui <- dashboardPage(
  skin = "blue",
  
  # ── Header ──────────────────────────────────────────────────────────────────
  dashboardHeader(
    title = HTML("<span style='font-weight:600;'>&#127807; GHG Gap Filler</span>"),
    titleWidth = 230
  ),
  
  # ── Sidebar ─────────────────────────────────────────────────────────────────
  dashboardSidebar(
    width = 245,
    tags$style(HTML("
      .sidebar-section-title {
        color: #ECF0F1; font-size:12px; font-weight:600; letter-spacing:.5px;
        text-transform:uppercase; padding:12px 15px 4px; margin:0;
      }
      .sidebar-note { color:#95A5A6; font-size:11px; padding:2px 15px; }
      .sidebar-divider { border-color:#34495E; margin:6px 15px; }
    ")),
    
    # ── Step 1: Upload ─────────────────────────────────────────────────────────
    tags$p("Step 1 — Upload", class = "sidebar-section-title"),
    div(style = "padding:0 10px;",
        fileInput("file_upload", NULL,
                  accept      = c(".csv", "text/csv"),
                  buttonLabel = icon("folder-open"),
                  placeholder = "Choose CSV file"),
        checkboxInput("has_header", "First row is header", value = TRUE)
    ),
    
    tags$hr(class = "sidebar-divider"),
    
    # ── Step 2: Configure ──────────────────────────────────────────────────────
    tags$p("Step 2 — Configure", class = "sidebar-section-title"),
    div(style = "padding:0 10px;",
        selectInput("target_col", "Target column (with gaps):",
                    choices = NULL, width = "100%"),
        selectInput("date_col", "Date/time column (optional):",
                    choices = NULL, width = "100%"),
        tags$p("Predictor columns:", class = "sidebar-note"),
        uiOutput("pred_checkboxes"),
        div(style = "padding:4px 5px;",
            actionLink("sel_all_preds",   "Select all"),
            tags$span(" | ", style="color:#666"),
            actionLink("desel_all_preds", "Clear all")
        )
    ),
    
    tags$hr(class = "sidebar-divider"),
    
    # ── Step 3: Models ─────────────────────────────────────────────────────────
    tags$p("Step 3 — Select Models", class = "sidebar-section-title"),
    div(style = "padding:0 10px;",
        checkboxGroupInput("sel_models", NULL,
                           choiceNames  = c("Random Forest", "XGBoost", "Gradient Boosting",
                                            "k-NN", "Elastic Net", "SVM (Radial)", "Cubist",
                                            HTML("<span style='color:#E377C2'>TabPFN (Python)</span>")),
                           choiceValues = c("RF","XGB","GBM","KNN","ENET","SVM","CUBIST","TABPFN"),
                           selected     = c("RF","XGB","GBM")
        ),
        div(style="padding:0 5px 4px;",
            sliderInput("n_folds", "CV folds:", min=3, max=10, value=5, step=1, width="100%")
        ),
        conditionalPanel("input.sel_models.indexOf('TABPFN') >= 0",
                         tags$p(HTML("&#9432; TabPFN requires Python + tabpfn-client. See README."),
                                class = "sidebar-note",
                                style = "color:#E377C2; background:#2C1B2E; padding:6px; border-radius:4px;")
        )
    ),
    
    tags$hr(class = "sidebar-divider"),
    
    # ── Step 4: Run ────────────────────────────────────────────────────────────
    div(style = "padding:8px 10px;",
        actionBttn("run_btn", " Run Gap-Filling",
                   icon  = icon("play"),
                   style = "material-flat", color = "success",
                   size  = "md", block = TRUE)
    ),
    
    div(style = "padding:2px 10px 8px; font-size:10px; color:#7F8C8D;",
        textOutput("sidebar_status", inline = TRUE)
    )
  ),
  
  # ── Body ─────────────────────────────────────────────────────────────────────
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper { background:#F4F6F8; }
        .box { border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }
        .info-box { border-radius:8px; }
        .nav-tabs-custom > .nav-tabs > li.active { border-top-color:#3498DB; }
        .tab-content { padding-top:10px; }
        table.dataTable td { font-size:12px; }
        .step-badge {
          display:inline-block; background:#3498DB; color:white;
          border-radius:50%; width:20px; height:20px; text-align:center;
          line-height:20px; font-size:11px; font-weight:700; margin-right:6px;
        }

        /* ── Scrolling ticker banner ──────────────────────────────────────────── */
        .ticker-wrap {
          width: 100%;
          background: linear-gradient(90deg, #c0392b, #e74c3c, #c0392b);
          overflow: hidden;
          white-space: nowrap;
          padding: 7px 0;
          margin-bottom: 14px;
          border-radius: 5px;
          box-shadow: 0 2px 6px rgba(192,57,43,0.35);
          position: relative;
        }
        .ticker-wrap::before, .ticker-wrap::after {
          content: \'\';
          position: absolute;
          top: 0; bottom: 0;
          width: 40px;
          z-index: 2;
        }
        .ticker-wrap::before { left:0;  background:linear-gradient(to right,#c0392b,transparent); }
        .ticker-wrap::after  { right:0; background:linear-gradient(to left, #c0392b,transparent); }
        .ticker-content {
          display: inline-block;
          animation: ticker-scroll 28s linear infinite;
          padding-left: 100%;
        }
        .ticker-content:hover { animation-play-state: paused; }
        @keyframes ticker-scroll {
          0%   { transform: translateX(0); }
          100% { transform: translateX(-100%); }
        }
        .ticker-item {
          display: inline-block;
          color: #FFFFFF;
          font-size: 13px;
          font-weight: 700;
          letter-spacing: 1.2px;
          text-transform: uppercase;
          padding: 0 50px;
        }
        .ticker-badge {
          background: rgba(255,255,255,0.25);
          border: 1px solid rgba(255,255,255,0.55);
          border-radius: 3px;
          padding: 1px 7px;
          margin-right: 6px;
          font-size: 11px;
          letter-spacing: 2px;
        }

        /* ── Watermark ───────────────────────────────────────────────────────── */
        .lab-watermark {
          text-align: center;
          margin-top: 14px;
          padding: 10px 0 4px;
          border-top: 1px dashed #D5D8DC;
        }
        .lab-watermark-logo  { font-size:24px; opacity:0.55; margin-bottom:2px; }
        .lab-watermark-text  { font-size:14px; font-weight:700; color:#2E75B6; letter-spacing:0.5px; font-style:italic; opacity:0.82; }
        .lab-watermark-sub   { font-size:11px; color:#95A5A6; margin-top:3px; letter-spacing:0.2px; }
        .lab-watermark-line  { width:120px; height:2px; background:linear-gradient(90deg,transparent,#2E75B6,transparent); margin:8px auto 0; border-radius:2px; }

        /* faint diagonal ghost watermark behind table */
        .preview-wrapper { position: relative; overflow: hidden; }
        .preview-wrapper::after {
          content: \' Saha Lab  \\00B7  University of Tennessee\';
          position: absolute;
          bottom: 55px; right: 20px;
          font-size: 13px;
          font-weight: 700;
          color: rgba(46,117,182,0.10);
          letter-spacing: 1px;
          pointer-events: none;
          transform: rotate(-8deg);
          white-space: nowrap;
          z-index: 0;
        }
      "))
    ),
    
    # ── Scrolling BETA ticker ──────────────────────────────────────────────────
    div(class = "ticker-wrap",
        div(class = "ticker-content",
            tags$span(class = "ticker-item",
                      tags$span(class = "ticker-badge", "BETA"),
                      HTML("&nbsp; Still Testing Phase &mdash; Results may vary &nbsp;"),
                      HTML("&nbsp;&#9888;&nbsp;"),
                      HTML("&nbsp; This tool is under active development &nbsp;"),
                      HTML("&nbsp;&#128300;&nbsp;"),
                      HTML("&nbsp; Not for production use without validation &nbsp;"),
                      HTML("&nbsp;&#9888;&nbsp;"),
                      HTML("&nbsp; Please report issues to Saha Lab, University of Tennessee &nbsp;")
            ),
            tags$span(class = "ticker-item",
                      tags$span(class = "ticker-badge", "BETA"),
                      HTML("&nbsp; Still Testing Phase &mdash; Results may vary &nbsp;"),
                      HTML("&nbsp;&#9888;&nbsp;"),
                      HTML("&nbsp; This tool is under active development &nbsp;"),
                      HTML("&nbsp;&#128300;&nbsp;"),
                      HTML("&nbsp; Not for production use without validation &nbsp;"),
                      HTML("&nbsp;&#9888;&nbsp;"),
                      HTML("&nbsp; Please report issues to  Saha Lab, University of Tennessee &nbsp;")
            )
        )
    ),
    
    fluidRow(
      # Info boxes — shown after data is loaded
      infoBoxOutput("ib_rows",  width = 3),
      infoBoxOutput("ib_obs",   width = 3),
      infoBoxOutput("ib_gaps",  width = 3),
      infoBoxOutput("ib_pct",   width = 3)
    ),
    
    tabBox(id = "main_tabs", width = 12,
           
           # ── 0. Data Preview ───────────────────────────────────────────────────────
           tabPanel(title = tagList(icon("table"), " Data"), value = "tab_data",
                    box(width = 12, title = "Dataset preview", solidHeader = TRUE, status = "primary",
                        div(class = "preview-wrapper",
                            DT::dataTableOutput("tbl_preview")
                        ),
                        # ── Lab watermark ────────────────────────────────────────────────────
                        div(class = "lab-watermark",
                            div(class = "lab-watermark-logo", HTML("&#127758;")),
                            div(class = "lab-watermark-text",
                                HTML(" Saha Lab &nbsp;&bull;&nbsp; GHG Gap-Filler")),
                            div(class = "lab-watermark-sub",
                                HTML(" Soil Biogeochemitsry & Nutrient Cycling Lab &nbsp;&mdash;&nbsp; University of Tennessee")),
                            div(class = "lab-watermark-line")
                        )
                    )
           ),
           
           # ── 1. Summary ────────────────────────────────────────────────────────────
           tabPanel(title = tagList(icon("chart-bar"), " Summary"), value = "tab_summary",
                    conditionalPanel("!output.has_results",
                                     div(style = "text-align:center; padding:40px; color:#999;",
                                         icon("play-circle", style="font-size:48px; color:#ddd;"),
                                         tags$h4("Run gap-filling to see results", style="color:#ccc;")
                                     )
                    ),
                    conditionalPanel("output.has_results",
                                     fluidRow(
                                       box(width = 12, title = "Model comparison — 5-fold CV", solidHeader = TRUE,
                                           status = "success",
                                           DT::dataTableOutput("tbl_summary")
                                       )
                                     ),
                                     fluidRow(
                                       box(width = 6, title = "RMSE (lower = better)", solidHeader = TRUE, status = "info",
                                           plotly::plotlyOutput("plt_rmse", height = "280px")
                                       ),
                                       box(width = 6, title = "R\u00b2 (higher = better)", solidHeader = TRUE, status = "info",
                                           plotly::plotlyOutput("plt_r2", height = "280px")
                                       )
                                     )
                    )
           ),
           
           # ── 2. Scatter ────────────────────────────────────────────────────────────
           tabPanel(title = tagList(icon("circle-dot"), " Scatter"), value = "tab_scatter",
                    box(width = 12, title = "Predicted vs Observed — 5-fold CV",
                        solidHeader = TRUE, status = "primary",
                        plotly::plotlyOutput("plt_scatter", height = "550px")
                    )
           ),
           
           # ── 3. Time Series ────────────────────────────────────────────────────────
           tabPanel(title = tagList(icon("chart-line"), " Time Series"), value = "tab_ts",
                    fluidRow(
                      column(3,
                             selectInput("ts_model", "Model:", choices = NULL, width = "100%")
                      ),
                      column(3, style = "padding-top:24px;",
                             checkboxInput("ts_show_gf", "Show gap-filled values", TRUE)
                      ),
                      column(3, style = "padding-top:24px;",
                             checkboxInput("ts_show_cv", "Show CV predictions", TRUE)
                      )
                    ),
                    box(width = 12, title = "Full time series", solidHeader = TRUE, status = "warning",
                        plotly::plotlyOutput("plt_ts", height = "430px")
                    )
           ),
           
           # ── 4. Variable Importance ────────────────────────────────────────────────
           tabPanel(title = tagList(icon("ranking-star"), " Importance"), value = "tab_imp",
                    fluidRow(
                      column(3,
                             radioButtons("imp_view", "View as:",
                                          choices = c("Heatmap (all models)" = "heat",
                                                      "Bars (per model)"     = "bars"),
                                          selected = "heat", inline = TRUE)
                      )
                    ),
                    conditionalPanel("input.imp_view === 'heat'",
                                     box(width = 12, title = "Variable importance heatmap (%) — all models",
                                         solidHeader = TRUE, status = "danger",
                                         plotly::plotlyOutput("plt_imp_heat", height = "420px")
                                     )
                    ),
                    conditionalPanel("input.imp_view === 'bars'",
                                     box(width = 12, title = "Variable importance bars — all models",
                                         solidHeader = TRUE, status = "danger",
                                         plotly::plotlyOutput("plt_imp_bars", height = "520px")
                                     )
                    )
           ),
           
           # ── 5. Download ───────────────────────────────────────────────────────────
           tabPanel(title = tagList(icon("download"), " Download"), value = "tab_dl",
                    fluidRow(
                      box(width = 12, solidHeader = TRUE, status = "primary",
                          title = "Download gap-filled results",
                          tags$p(
                            "The gap-filled dataset contains the original data plus one filled column",
                            "per model (", tags$code("target_MODELKEY"), "). Observed values are",
                            "preserved unchanged."
                          ),
                          tags$br(),
                          fluidRow(
                            column(3,
                                   downloadButton("dl_csv", HTML("&#11015; Gap-filled CSV"),
                                                  class = "btn-success", style = "width:100%;")
                            ),
                            column(3,
                                   downloadButton("dl_xlsx", HTML("&#11015; Excel report (4 sheets)"),
                                                  class = "btn-info", style = "width:100%;")
                            )
                          ),
                          tags$br(), tags$hr(),
                          tags$h5("Preview — gap rows highlighted in yellow:"),
                          DT::dataTableOutput("tbl_gap_preview")
                      )
                    )
           )
    ) # end tabBox
  )   # end dashboardBody
)     # end dashboardPage


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER
# ═══════════════════════════════════════════════════════════════════════════════
server <- function(input, output, session) {
  
  rv <- reactiveValues(
    raw_data = NULL,
    results  = NULL,
    status   = "Upload a CSV to begin."
  )
  
  # ── Helpers ─────────────────────────────────────────────────────────────────
  set_status <- function(msg) { rv$status <- msg }
  
  # ── File upload ──────────────────────────────────────────────────────────────
  observeEvent(input$file_upload, {
    req(input$file_upload)
    df <- tryCatch(
      read.csv(input$file_upload$datapath, header = input$has_header,
               stringsAsFactors = FALSE, check.names = FALSE),
      error = function(e) {
        showNotification(paste("Cannot read file:", conditionMessage(e)), type = "error")
        NULL
      }
    )
    req(df)
    rv$raw_data <- df
    rv$results  <- NULL
    set_status(paste("Loaded:", nrow(df), "rows,", ncol(df), "columns"))
    
    cols         <- names(df)
    numeric_cols <- cols[vapply(df, is.numeric, logical(1))]
    pct_na       <- vapply(df[numeric_cols],
                           function(x) mean(is.na(x)), numeric(1))
    
    # Auto-detect target: numeric column with most NAs
    default_target <- if (length(numeric_cols) > 0 && any(pct_na > 0))
      numeric_cols[which.max(pct_na)]
    else if (length(numeric_cols) > 0) tail(numeric_cols, 1)
    else cols[1]
    
    # Auto-detect date column
    date_candidates <- cols[grepl("date|time|doy|year", cols, ignore.case = TRUE)]
    default_date    <- if (length(date_candidates) > 0) date_candidates[1] else "None"
    
    updateSelectInput(session, "target_col",
                      choices = numeric_cols, selected = default_target)
    updateSelectInput(session, "date_col",
                      choices = c("None", cols), selected = default_date)
    
    updateTabsetPanel(session, "main_tabs", selected = "tab_data")
  })
  
  # ── Predictor checkboxes ─────────────────────────────────────────────────────
  output$pred_checkboxes <- renderUI({
    req(rv$raw_data, input$target_col)
    df    <- rv$raw_data
    cols  <- names(df)
    excl  <- c(input$target_col,
               if (!is.null(input$date_col) && input$date_col != "None") input$date_col)
    
    num_cols <- cols[vapply(df, is.numeric, logical(1))]
    cat_cols <- setdiff(cols, c(num_cols, excl))
    cands    <- setdiff(cols, excl)
    
    labels <- ifelse(cands %in% cat_cols,
                     paste0(cands, " <span style='color:#E17A2F;font-size:10px'>[cat]</span>"),
                     cands)
    
    checkboxGroupInput("pred_cols", NULL,
                       choiceNames  = lapply(labels, function(l) HTML(l)),
                       choiceValues = cands,
                       selected     = setdiff(num_cols, excl)
    )
  })
  
  # Select / deselect all predictors
  observeEvent(input$sel_all_preds, {
    req(rv$raw_data, input$target_col)
    df   <- rv$raw_data
    excl <- c(input$target_col,
              if (!is.null(input$date_col) && input$date_col != "None") input$date_col)
    updateCheckboxGroupInput(session, "pred_cols",
                             selected = setdiff(names(df), excl))
  })
  observeEvent(input$desel_all_preds, {
    updateCheckboxGroupInput(session, "pred_cols", selected = character(0))
  })
  
  # ── Info boxes ───────────────────────────────────────────────────────────────
  output$ib_rows <- renderInfoBox({
    n <- if (!is.null(rv$raw_data)) nrow(rv$raw_data) else "—"
    infoBox("Total rows", n, icon = icon("table"), color = "blue",  fill = TRUE)
  })
  output$ib_obs <- renderInfoBox({
    n <- if (!is.null(rv$raw_data) && !is.null(input$target_col))
      sum(!is.na(rv$raw_data[[input$target_col]])) else "—"
    infoBox("Observed", n, icon = icon("check-circle"), color = "green", fill = TRUE)
  })
  output$ib_gaps <- renderInfoBox({
    n <- if (!is.null(rv$raw_data) && !is.null(input$target_col))
      sum(is.na(rv$raw_data[[input$target_col]])) else "—"
    infoBox("Gaps to fill", n, icon = icon("circle-question"), color = "orange", fill = TRUE)
  })
  output$ib_pct <- renderInfoBox({
    pct <- if (!is.null(rv$raw_data) && !is.null(input$target_col))
      paste0(round(100 * mean(is.na(rv$raw_data[[input$target_col]])), 1), "%") else "—"
    infoBox("Gap %", pct, icon = icon("percent"), color = "red", fill = TRUE)
  })
  
  # ── Data preview ─────────────────────────────────────────────────────────────
  output$tbl_preview <- DT::renderDataTable({
    req(rv$raw_data)
    DT::datatable(rv$raw_data,
                  rownames = FALSE,
                  options  = list(pageLength = 10, scrollX = TRUE, dom = "lrtip"),
                  class    = "compact stripe"
    )
  })
  
  # ── Status output ────────────────────────────────────────────────────────────
  output$sidebar_status <- renderText(rv$status)
  output$has_results    <- reactive(!is.null(rv$results))
  outputOptions(output, "has_results", suspendWhenHidden = FALSE)
  
  # ── Run button ───────────────────────────────────────────────────────────────
  observeEvent(input$run_btn, {
    req(rv$raw_data, input$target_col, input$pred_cols, input$sel_models)
    
    if (length(input$pred_cols) < 1) {
      showNotification("Select at least one predictor column.", type = "warning"); return()
    }
    if (length(input$sel_models) < 1) {
      showNotification("Select at least one model.", type = "warning"); return()
    }
    
    df     <- rv$raw_data
    target <- input$target_col
    preds  <- input$pred_cols
    
    # ── Encode categoricals ─────────────────────────────────────────────────
    df_proc <- df
    encoded <- character(0)
    for (col in preds) {
      if (!is.numeric(df_proc[[col]])) {
        df_proc[[col]] <- as.numeric(factor(df_proc[[col]]))
        encoded <- c(encoded, col)
      }
    }
    if (length(encoded))
      set_status(paste("Encoded:", paste(encoded, collapse=", ")))
    
    # ── Remove rows missing ALL predictors ──────────────────────────────────
    complete_rows <- complete.cases(df_proc[, preds, drop = FALSE])
    n_dropped     <- sum(!complete_rows)
    df_proc       <- df_proc[complete_rows, ]
    if (n_dropped > 0)
      showNotification(paste(n_dropped, "rows dropped (missing predictor values)."),
                       type = "warning")
    
    obs_df <- df_proc[!is.na(df_proc[[target]]), ]
    gap_df <- df_proc[ is.na(df_proc[[target]]), ]
    
    if (nrow(obs_df) < 10) {
      showNotification("Need at least 10 observed rows.", type = "error"); return()
    }
    
    set_status(paste0("Running ", length(input$sel_models), " models..."))
    
    withProgress(message = "Running gap-filling", value = 0, {
      res <- tryCatch(
        run_gap_filling(
          obs_df          = obs_df,
          gap_df          = gap_df,
          target_col      = target,
          predictor_cols  = preds,
          selected_models = input$sel_models,
          n_folds         = input$n_folds,
          progress_fn     = function(p, msg) setProgress(p, detail = msg)
        ),
        error = function(e) {
          showNotification(paste("Error:", conditionMessage(e)), type = "error")
          NULL
        }
      )
    })
    
    if (is.null(res)) { set_status("Run failed. Check console."); return() }
    
    rv$results <- res
    best       <- res$metrics_all$Model[1]
    best_r2    <- res$metrics_all$R2[1]
    set_status(paste0("Done! Best: ", best, " (R\u00b2=", best_r2, ")"))
    
    updateSelectInput(session, "ts_model",
                      choices  = names(res$results),
                      selected = res$metrics_all$Model_Key[1])
    
    updateTabsetPanel(session, "main_tabs", selected = "tab_summary")
    showNotification("Gap-filling complete!", type = "message", duration = 5)
  })
  
  # ── Summary table ─────────────────────────────────────────────────────────────
  output$tbl_summary <- DT::renderDataTable({
    req(rv$results)
    m    <- rv$results$metrics_all
    show <- intersect(c("Model","N","RMSE","MAE","Bias","R2","NSE","PBIAS",
                        "DeltaASum","Time_s","Rank"), names(m))
    DT::datatable(m[, show], rownames = FALSE, class = "compact stripe",
                  options = list(pageLength = 15, dom = "t"),
                  caption = htmltools::tags$caption(style="caption-side:bottom; font-size:11px;",
                                                    paste0("5-fold CV | Target: ", rv$results$target_col,
                                                           " | Observed: ", nrow(rv$results$obs_df),
                                                           " | Gaps filled: ", nrow(rv$results$gap_df)))
    ) |>
      DT::formatRound(intersect(c("RMSE","MAE","Bias","R2","NSE","PBIAS"), show), 3) |>
      DT::formatStyle("Rank",
                      backgroundColor = DT::styleInterval(c(2,3.5),c("#D5EDD5","#FFF3CD","#FDDEDE")),
                      fontWeight = "bold")
  })
  
  # ── RMSE bar ──────────────────────────────────────────────────────────────────
  output$plt_rmse <- plotly::renderPlotly({
    req(rv$results)
    m   <- rv$results$metrics_all
    m   <- m[order(m$RMSE), ]
    clr <- CLRS[seq_len(nrow(m))]
    plotly::plot_ly(m,
                    y    = ~factor(Model, levels = Model),
                    x    = ~RMSE,
                    type = "bar", orientation = "h",
                    marker = list(color = clr, line = list(color = "white", width = 0.5)),
                    text = ~paste0(" ", RMSE),
                    textposition = "outside"
    ) |>
      plotly::layout(
        xaxis  = list(title = "RMSE", zeroline = FALSE),
        yaxis  = list(title = ""),
        margin = list(l = 10, r = 60), plot_bgcolor = "#FAFAFA",
        paper_bgcolor = "white"
      )
  })
  
  # ── R² bar ────────────────────────────────────────────────────────────────────
  output$plt_r2 <- plotly::renderPlotly({
    req(rv$results)
    m   <- rv$results$metrics_all
    m   <- m[order(-m$R2), ]
    clr <- CLRS[seq_len(nrow(m))]
    plotly::plot_ly(m,
                    y    = ~factor(Model, levels = Model),
                    x    = ~R2,
                    type = "bar", orientation = "h",
                    marker = list(color = clr, line = list(color = "white", width = 0.5)),
                    text = ~paste0(" ", R2),
                    textposition = "outside"
    ) |>
      plotly::add_segments(x = 0, xend = 0, y = 0.4, yend = nrow(m) + 0.6,
                           line = list(color = "black", width = 1, dash = "dot")) |>
      plotly::layout(
        xaxis  = list(title = "R\u00b2", zeroline = FALSE),
        yaxis  = list(title = ""),
        margin = list(l = 10, r = 60), plot_bgcolor = "#FAFAFA",
        paper_bgcolor = "white"
      )
  })
  
  # ── Scatter plots ─────────────────────────────────────────────────────────────
  output$plt_scatter <- plotly::renderPlotly({
    req(rv$results)
    res   <- rv$results$results
    y_obs <- rv$results$y_obs
    mkeys <- names(res)
    ncols <- min(3L, length(mkeys))
    nrows <- ceiling(length(mkeys) / ncols)
    
    plots <- lapply(seq_along(mkeys), function(i) {
      mk  <- mkeys[i]
      oof <- res[[mk]]$cv_preds
      m_  <- res[[mk]]$metrics
      ok  <- !is.na(y_obs) & !is.na(oof)
      lo  <- min(c(y_obs[ok], oof[ok]), na.rm = TRUE)
      hi  <- max(c(y_obs[ok], oof[ok]), na.rm = TRUE)
      clr <- CLRS[(i - 1) %% length(CLRS) + 1]
      
      plotly::plot_ly(
        x = y_obs[ok], y = oof[ok],
        type = "scatter", mode = "markers",
        marker = list(size = 5, opacity = 0.45, color = clr),
        showlegend = FALSE
      ) |>
        plotly::add_segments(x = lo, xend = hi, y = lo, yend = hi,
                             line = list(color = "#333", dash = "dash", width = 1),
                             showlegend = FALSE) |>
        plotly::layout(
          annotations = list(list(
            x = 0.05, y = 0.97, xref = "paper", yref = "paper",
            text = paste0("<b>", res[[mk]]$label, "</b><br>RMSE=", m_$RMSE,
                          " R\u00b2=", m_$R2),
            showarrow = FALSE, align = "left",
            font = list(size = 10, color = clr)
          )),
          xaxis = list(title = "Observed"),
          yaxis = list(title = "Predicted")
        )
    })
    
    plotly::subplot(plots, nrows = nrows, shareX = FALSE, shareY = FALSE,
                    titleX = TRUE, titleY = TRUE, margin = 0.06) |>
      plotly::layout(plot_bgcolor = "#FAFAFA", paper_bgcolor = "white")
  })
  
  # ── Time series ───────────────────────────────────────────────────────────────
  output$plt_ts <- plotly::renderPlotly({
    req(rv$results, input$ts_model)
    res    <- rv$results$results
    obs_df <- rv$results$obs_df
    gap_df <- rv$results$gap_df
    target <- rv$results$target_col
    mk     <- if (input$ts_model %in% names(res)) input$ts_model else names(res)[1]
    lbl    <- res[[mk]]$label
    
    use_date <- !is.null(input$date_col) && input$date_col != "None" &&
      input$date_col %in% names(obs_df)
    obs_x    <- if (use_date) obs_df[[input$date_col]] else seq_len(nrow(obs_df))
    gap_x    <- if (use_date && input$date_col %in% names(gap_df))
      gap_df[[input$date_col]]
    else nrow(obs_df) + seq_len(nrow(gap_df))
    x_title  <- if (use_date) input$date_col else "Row index"
    
    clr <- unname(MODEL_COLORS[mk] %||% "#2E75B6")
    
    fig <- plotly::plot_ly() |>
      plotly::add_trace(x = obs_x, y = obs_df[[target]],
                        type = "scatter", mode = "lines",
                        name = "Observed (line)",
                        line = list(color = "#CCCCCC", width = 1), showlegend = TRUE) |>
      plotly::add_trace(x = obs_x, y = obs_df[[target]],
                        type = "scatter", mode = "markers",
                        name = "Observed",
                        marker = list(size = 4, color = "#888", opacity = 0.55))
    
    if (isTRUE(input$ts_show_cv)) {
      oof <- res[[mk]]$cv_preds
      fig <- fig |> plotly::add_trace(
        x = obs_x, y = oof,
        type = "scatter", mode = "markers",
        name = "CV predicted",
        marker = list(size = 5, color = clr, opacity = 0.8)
      )
    }
    
    if (isTRUE(input$ts_show_gf) && nrow(gap_df) > 0) {
      gf <- res[[mk]]$gap_preds
      fig <- fig |> plotly::add_trace(
        x = gap_x, y = gf,
        type = "scatter", mode = "markers",
        name = "Gap-filled",
        marker = list(size = 9, color = clr, symbol = "triangle-up", opacity = 0.9)
      )
    }
    
    fig |>
      plotly::layout(
        title  = list(text = paste0(lbl, " — Observed + CV predictions + Gap-filled"),
                      font = list(size = 12)),
        xaxis  = list(title = x_title),
        yaxis  = list(title = target),
        legend = list(orientation = "h", y = -0.12),
        plot_bgcolor = "#FAFAFA", paper_bgcolor = "white"
      )
  })
  
  # ── Importance heatmap ────────────────────────────────────────────────────────
  output$plt_imp_heat <- plotly::renderPlotly({
    req(rv$results)
    res  <- rv$results$results
    mkeys <- names(res)
    
    imp_all <- do.call(rbind, lapply(mkeys, function(mk) {
      df_ <- res[[mk]]$importance
      data.frame(Feature = df_$Feature, Importance = df_$Importance,
                 Model = res[[mk]]$label, stringsAsFactors = FALSE)
    }))
    
    feat_order <- imp_all |>
      dplyr::group_by(Feature) |>
      dplyr::summarise(mx = max(Importance, na.rm = TRUE), .groups = "drop") |>
      dplyr::arrange(mx) |>
      dplyr::pull(Feature)
    
    imp_wide <- tidyr::pivot_wider(imp_all, names_from = "Model",
                                   values_from = "Importance",
                                   values_fill  = 0)
    z    <- as.matrix(imp_wide[, -1, drop = FALSE])
    rows <- imp_wide$Feature
    z    <- z[match(feat_order, rows), , drop = FALSE]
    
    plotly::plot_ly(
      x = colnames(z), y = feat_order, z = z,
      type = "heatmap", colorscale = "YlOrRd",
      colorbar = list(title = "Imp (%)"),
      text = round(z, 1), texttemplate = "%{text}",
      showscale = TRUE
    ) |>
      plotly::layout(
        xaxis  = list(title = "", tickangle = -30),
        yaxis  = list(title = ""),
        margin = list(l = 130, b = 90),
        plot_bgcolor = "white", paper_bgcolor = "white"
      )
  })
  
  # ── Importance bars ───────────────────────────────────────────────────────────
  output$plt_imp_bars <- plotly::renderPlotly({
    req(rv$results)
    res   <- rv$results$results
    mkeys <- names(res)
    ncols <- min(3L, length(mkeys))
    nrows <- ceiling(length(mkeys) / ncols)
    
    plots <- lapply(seq_along(mkeys), function(i) {
      mk  <- mkeys[i]; lbl <- res[[mk]]$label
      imp <- res[[mk]]$importance |> dplyr::arrange(Importance)
      clr <- CLRS[(i - 1) %% length(CLRS) + 1]
      plotly::plot_ly(
        imp, y = ~factor(Feature, levels = Feature), x = ~Importance,
        type = "bar", orientation = "h",
        marker = list(color = clr, opacity = 0.85),
        showlegend = FALSE
      ) |>
        plotly::layout(
          title  = list(text = lbl, font = list(size = 11, color = clr)),
          xaxis  = list(title = "Importance (%)"),
          yaxis  = list(title = ""),
          margin = list(l = 10)
        )
    })
    
    plotly::subplot(plots, nrows = nrows, shareX = FALSE, shareY = FALSE,
                    titleX = TRUE, titleY = TRUE, margin = 0.07) |>
      plotly::layout(plot_bgcolor = "#FAFAFA", paper_bgcolor = "white")
  })
  
  # ── Build filled dataset ──────────────────────────────────────────────────────
  filled_data <- reactive({
    req(rv$results)
    res    <- rv$results
    obs_df <- res$obs_df
    gap_df <- res$gap_df
    target <- res$target_col
    mkeys  <- names(res$results)
    
    obs_df$Source <- "Observed"
    gap_df$Source <- "Gap"
    
    for (mk in mkeys) {
      col_name <- paste0(target, "_", mk)
      obs_df[[col_name]] <- obs_df[[target]]     # keep original for observed
      gap_df[[col_name]] <- res$results[[mk]]$gap_preds
    }
    
    combined <- dplyr::bind_rows(obs_df, gap_df)
    
    # Sort by date if available
    dc <- input$date_col
    if (!is.null(dc) && dc != "None" && dc %in% names(combined))
      combined <- combined[order(combined[[dc]]), ]
    
    combined
  })
  
  # ── Gap preview ───────────────────────────────────────────────────────────────
  output$tbl_gap_preview <- DT::renderDataTable({
    req(rv$results)
    fd     <- filled_data()
    target <- rv$results$target_col
    mkeys  <- names(rv$results$results)
    keep   <- unique(c(
      head(names(fd), min(4L, ncol(fd))),
      target,
      paste0(target, "_", mkeys),
      "Source"
    ))
    keep <- intersect(keep, names(fd))
    
    DT::datatable(fd[, keep], rownames = FALSE, class = "compact stripe",
                  options = list(pageLength = 10, scrollX = TRUE, dom = "lrtip")
    ) |>
      DT::formatRound(which(vapply(fd[, keep], is.numeric, logical(1))), 3) |>
      DT::formatStyle("Source",
                      backgroundColor = DT::styleEqual("Gap", "#FFF3CD"),
                      fontWeight      = DT::styleEqual("Gap", "bold"))
  })
  
  # ── Download CSV ──────────────────────────────────────────────────────────────
  output$dl_csv <- downloadHandler(
    filename = function()
      paste0("GapFilled_", rv$results$target_col, "_",
             format(Sys.Date(), "%Y%m%d"), ".csv"),
    content  = function(file) write.csv(filled_data(), file, row.names = FALSE)
  )
  
  # ── Download Excel ────────────────────────────────────────────────────────────
  output$dl_xlsx <- downloadHandler(
    filename = function()
      paste0("GapFill_Report_", format(Sys.Date(), "%Y%m%d"), ".xlsx"),
    content  = function(file) {
      res   <- rv$results
      mkeys <- names(res$results)
      
      imp_all <- do.call(rbind, lapply(mkeys, function(mk) {
        cbind(Model = res$results[[mk]]$label,
              res$results[[mk]]$importance)
      }))
      
      settings <- data.frame(
        Setting = c("Run date","Target column","Predictor columns",
                    "Models","CV folds","Observed rows","Gap rows",
                    "Best model","Best RMSE","Best R2"),
        Value   = c(format(Sys.time(), "%d %B %Y %H:%M"),
                    res$target_col,
                    paste(res$predictor_cols, collapse = ", "),
                    paste(sapply(mkeys, function(m) res$results[[m]]$label),
                          collapse = "; "),
                    as.character(input$n_folds),
                    as.character(nrow(res$obs_df)),
                    as.character(nrow(res$gap_df)),
                    as.character(res$metrics_all$Model[1]),
                    as.character(res$metrics_all$RMSE[1]),
                    as.character(res$metrics_all$R2[1])),
        stringsAsFactors = FALSE
      )
      
      writexl::write_xlsx(list(
        Summary              = res$metrics_all,
        Gap_Filled_Data      = filled_data(),
        Variable_Importance  = imp_all,
        Settings             = settings
      ), path = file)
    }
  )
}

# ── Launch ────────────────────────────────────────────────────────────────────
shinyApp(ui, server)

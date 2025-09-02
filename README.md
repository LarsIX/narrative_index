# AI Narrative Index (AINI)

This repository contains the full research pipeline for constructing the **AI Narrative Index (AINI)** — a time-series measure of how artificial intelligence is represented and emotionally framed in financial news. The index is derived from **Wall Street Journal (WSJ)** articles from **2023 to 2025** and used as an independent variable to estimate asset returns.

The project integrates **Transformer-based NLP**, **manual annotation**, **deep learning fine-tuning**, and **econometric inference** — all within a modular and reproducible architecture.

---

## Research Objectives

- **Develop multiple variants of an AI Narrative Index (AINI)** using both **supervised fine-tuning** and **zero-shot inference**.

- **Evaluate the temporal effects** of narrative hype on market dynamics using Granger Causality and Transfer Entropy.

- **Ensure scientific rigor** through pre-annotation protocols, dual-labeller verification, and formal statistical diagnostics.

---

## Construction of the AINI Index

To quantify the AI Narrative Index (AINI), this project implements **three complementary methods** — combining human annotation, Transformer models, and prompt-based weak supervision.

### 1. AINI via Manual Annotation and FinBERT Fine-Tuning

- A manually annotated dataset was created in collaboration with a **professional second annotator**, focusing on:
  - **AI Relevance** (binary classification)  

- A custom **FinBERT model** is fine-tuned on this dataset, using:
  - Class-weighted loss  
  - Window-based context extraction  
  - Early stopping and evaluation logging

- The resulting predictions form a **binary AI Narrative Index**, capturing the **topic salience** of AI-related narratives — i.e., how prominently AI is discussed in financial news coverage.

---

### 2. AINI via Standard FinBERT and Snippet Reduction

- The pretrained **FinBERT model** ([ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)) is applied to **pre-cleaned WSJ articles** without additional training.

- Articles are first **reduced using preprocessing and dimensionality reduction techniques**, including:
  - Heuristic snippet extraction around AI keywords  
  - BERT-based contextual windowing

- The resulting **probabilistic sentiment outputs** are aggregated into a daily AINI time series.

---

## Statistical Testing & Causal Inference

The project applies **rigorous time series diagnostics and hypothesis testing** to analyze the link between narrative indices and financial markets:

### Stationarity Testing

- Augmented Dickey-Fuller (ADF)
- Phillips-Perron (PP)
- Kwiatkowski-Phillips-Schmidt-Shin (KPSS) 

Used to validate input variables before time series modeling.

### Granger Causality (GC)

Tests predictive causality between AINI and log returns or volatility indices using:

- **bootstrapping**: to create empirical p-values
- **Benjamini-Hochberg correction** to reduce the false-discovery-rate

Supports both directions: `AINI ➝ returns` and `returns ➝ AINI`.

## Transfer Entropy (TE)

- KSG estimator for nonlinear, asymmetric dependencies
- IDTxl framework with permutation-based significance testing

## Project Structure

```text
AI_narrative_index/
│
├── src/
│   ├── fetch_data/
│   │   ├── load_financial_data.py              # Download and prepare financial market data
│   │   ├── wsj_archive_crawler.py              # Crawl WSJ archive pages to collect article URLs
│   │   └── wsj_archive_scraper.py              # Scrape full article content from collected WSJ URLs
│   │   
│   │
│   ├── preprocessing/
│   │   ├── clean_database.py                   # Clean and filter articles by section, length, and duplicates
│   │   ├── corpus_cleaning.py                  # Remove UI/meta elements from WSJ article text
│   │   ├── reduce_db_for_sentiment.py          # Extract article subset for sentiment prediction
│   │   ├── combine_article_dbs.py              # Concatenates yearly WSJ-Databases into single CSV
│   │   ├── section_filtering.py                # Filter irrelevant WSJ sections from the database
│   │   └── simple_ai_filter.py                 # Flag articles that mention AI-related terms
│   │   
│   ├── annotation/
│   │   ├── comparing_annotations.py            # Resolve disagreements between annotators
│   │   ├── label_articles.py                   # Launch interactive tool for AI and hype annotation
│   │   └── prompt_gpt.py                       # Prompt GPT to annotate articles
│   │
│   ├── modelling/
│   │   ├── ai_windows.py                       # Extract context-aware snippets around AI keywords
│   │   ├── calculate_summary_statistics.py     # Calculates summary statistics of AINI values
│   │   ├── compute_extrema.py                  # Compute extrema (max / min) of AINI variables by date
│   │   ├── construct_AINI_variables.py         # Build daily AINI index with normalization, EMA etc.
│   │   ├── CustomFinBERT.py                    # Define custom FinBERT model with dropout & class weights
│   │   ├── stationarity_testing.py             # Perform ADF and PP stationarity tests
│   │   ├── estimate_transfer_entropy.py        # Estimate Transfer Entropy between AINI and financial variables
│   │   ├── estimate_granger_causality.py       # Estimate Granger causality with heteroskedasticity-aware bootstrapping
│   │   ├── predict_binary_AINI_FinBERT.py      # Classify articles (AI-related vs. not) using custom FinBERT
│   │   └── predict_AINI_FinBERT_window.py      # Classify with windowed context using FinBERT
│   │
│   │
│   ├── visualizations/
│   │   ├── construct_latex_tables.py           # Construct latex tables for final thesis
│   │   └── plot_granger_causality.py           # Visualize Granger causality outputs
│   │  
│   ├── databases/
│   │   ├── fix_article_ids_in_db.py            # ensures unique article_id in database 
│   │   └── create_database.py                  # creates SQL database with article & articles_id table
│   │
│   ├── scripts/                                # CLI entry points for reproducible execution
│   │   ├── run_create_database.py              # Initialize article database schema and structure
│   │   ├── run_wsj_scraper.py                  # Run WSJ crawler and scraper
│   │   ├── run_clean_database.py               # Clean article database by year
│   │   ├── run_reduce_db_for_sentiment.py      # Reduce database for sentiment analysis
│   │   ├── run_predict_investor_sentiment.py   # Apply sentiment prediction using standard FinBERT
│   │   ├── run_predict_binary_AINI_FinBERT.py  # Run binary AINI classification pipeline
│   │   ├── run_predict_AINI_FinBERT_window.py  # Run context-aware classification on snippets
│   │   │── run_combine_article_dbs.py          # Concatenates yearly WSJ-Databases into single CSV
│   │   │── run_fix_article_ids.py              # Run function to ensure unique article_id in database
│   │   │── run_estimate_granger_causality.py   # Run function to estimate Granger Causality
│   │   │── run_estimate_OLS.py                 # Run function to estimate OLS
│   │   │── run_naive_labeling.py               # Label AI-relavance based on naive keywords
│   │   └── run_construct_AINI_variables.py     # Construct final AINI index file for modeling
│
├── notebooks/
│       ├── compare_annotations.ipynb          # Resolve annotation disagreements interactively
│       ├── train_FinBERT_annot.ipynb          # Fine-tune FinBERT on AI-relatedness labels
│       ├── explore_FinBERT_annotat.ipynb      # Inspect FinBERT predictions across configurations
│       ├── exploratory_analysis_wsj.ipynb     # Explore WSJ article dataset and structure
│       ├── exploratory_analysis_aini.ipynb    # Explore FinBERT predictions & AINI variables
│       ├── sample_articles.ipynb              # Sample articles for manual annotation
│       ├── label_manually.ipynb               # Manually annotate AI relevance and sentiment 
│       ├── benchmark_windows.ipynb            # Benchmarks naive AI windows against manual annotation
│       ├── test_stationarity.ipynb            # Perform visual and statistical stationarity tests
│       ├── predict_transfer_entropy.ipynb     # Estimate transfer entropy for causal analysis
│       ├── predict_granger_causality.ipynb    # Estimate Apply Granger causality between AINI and asset returns
│       ├── analyse_gc_results.ipynb           # Inspect and visualize Granger causality results
│       ├── compare_classification_vars.ipynb  # Benchmarks different classification methods
│       ├── visualize_ksg.ipynb                # Explain the Kraskov estimator for entropy
│       ├── label_with_gpt.ipynb               # Annotate articles with GPT
│       ├── construct_latex_tables.ipynb       # Construct latex tables for final thesis
│       └── visualize_aini_variables.ipynb     # Explore AINI index trends and dynamics
│
├── data/
│   ├── raw/                                   # Contain scraped articles and raw financial data
│   ├── interim/                               # Store intermediate outputs (e.g. annotated WSJ batches)
│   └── processed/
│       ├── variables/                         # Store constructed variables (e.g. AINI, sentiment, TE)
│       └── articles/                          # Store cleaned and filtered article texts
│
└── models/                                    # Store fine-tuned FinBERT and sentiment models

```

# Data Catalogue 

This catalogue documents datasets used and produced in the **AI Narrative Index (AINI)** project.  
It is structured according to **MLOps best practices**:  
- **Traceability**: each dataset is linked to its generating script.  
- **Reproducibility**: raw → interim → final → processed pipeline is explicit.  
- **Versioning**: datasets with `{year}` or `{vers}` placeholders ensure temporal/version separation.  
- **Auditability**: provenance of all transformations is documented.  


## 📂 `data/raw/`

Raw inputs, immutable once collected.  
Source of truth for all downstream datasets.

### 📂 `articles/`

| File | Description | Provenance |
|------|-------------|------------|
| `articlesWSJ_{year}.db` | WSJ raw databases containing `articles` and `articles_index` | Generated by `create_database.py`, populated via `wsj_archive_scraper.py` & `wsj_archive_crawler.py` |

### 📂 `financial/`

All contain OHLCV financial data: `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.  
Scraped via `load_financial_data.py`.

| File | Description |
|------|-------------|
| `{TICKER}_full_2023_2025.csv` | Daily OHLCV for single ticker (e.g., `AAPL`, `NVDA`, `MSFT`) |
| `full_daily_2023_2025.csv` | Aggregated OHLCV data for all tickers |

---

## 📂 `data/interim/`

Staging area for experiments, human-in-the-loop tasks, and partially processed data.  
Used for **annotation, sampling, and benchmarking**.

### Sampling & Annotation

| File | Description | Provenance |
|------|-------------|------------|
| `articles_WSJ_batch_{1–4}.csv` | Raw sampled batches for annotation | `sample_articles.ipynb` |
| `articles_WSJ_sub500.csv` | Initial 500-article subset (basis for batches) | `sample_articles.ipynb` |
| `articles_WSJ_batch_{n}_annotator.csv` | Professional annotator labels | External annotator → imported |
| `articles_WSJ_batch_{n}_author.csv` | Author labels | `label_manually.ipynb` |
| `*_subsample_author.csv` | 25% author-labeled subsample | Manual selection |

---

## 📂 `data/processed/`

Canonical datasets ready for **training, evaluation, and benchmarking**.  
All interim ambiguities (noise, disagreement) resolved.

### 📂 `articles/`

| File | Description | Provenance |
|------|-------------|------------|
| `articlesWSJ_clean_{year}.db` | Cleaned WSJ articles (polluting patterns removed) | `clean_database.py` (patterns: `corpus_cleaning.py`) |
| `annotated_subsample_WSJ_final.csv` | Consensus labels after disagreement resolution | `compare_annotations.ipynb` |
| `articles_WSJ_batch_{n}_final.csv` | Final reconciled annotation batches | `compare_annotations.ipynb` |


---

## 📂 `variables/`

Model outputs, diagnostics, and statistical analysis results.  
All **results are reproducible** from source code.

| File | Description | Provenance |
|------|-------------|------------|
| `w0_AINI_variables.csv`, `w1_AINI_variables.csv`, `w2_AINI_variables.csv`, `binary_AINI_variables.csv` | AINI variables (normalized + EMA α=0.2/0.8) | `construct_AINI_variables.py` |
| `FinBERT_AINI_prediction_{year}_windowsize_{n}.csv` | Context-window predictions (-1, 0, 1) | `predict_AINI_FinBERT_window.py` |
| `FinBERT_binary_prediction_{year}.csv` | Binary FinBERT predictions on pre-labeled data | `predict_AINI_FinBERT_prelabeled_fin.py` |
| `granger_causality_{spec}.csv` | GC results (AINI ↔ returns) with 10k bootstrap + FDR | `estimate_granger_causality.py` |
| `ols_sameday_mbboot_fdr_{spec}.csv` | Same-day OLS contemporaneous effects | `estimate_OLS.py` |
| `diagnostics_{spec}.csv` | OLS residual diagnostics (Ljung-Box, BG, ARCH LM, etc.) | `ols_residual_diagnostics.py` |
| `combined_te_results_window_1.csv` | Transfer Entropy results | `calc_entropy.py` |
| `extrema.csv` | AINI minima/maxima | `exploratory_analysis_aini.ipynb` |
| `{vers}_AINI_variables.csv` | AINI measures (normalized + EMA, relative) | `run_construct_AINI_variables.py` |
| `extrema.csv` | Min/max AINI values by measure count | `exploratory_analysis_results.ipynb` |
| `naive_AI_labels_{year}.csv` | Dictionary-based AI relevance labels | `label_articles.py` (`naive_labeling`) |
| `n_articles.csv` | article count per day | `exploratory_analysis_aini.ipynb`|

---


### Note

The scraped data is used strictly for academic and scientific research purposes and is not shared publicly due to copyright and licensing restrictions.
The trained model can be shared upon request, subject to data sharing considerations (e.g., size and usage context).

## Author and Context

This repository forms the empirical foundation of a master's thesis in Economics, with the aim of combining:

- **Behavioral finance and narrative theory**
- **NLP and deep learning with Transformers**

---

## Contact

- 📧 `lars.augustat@icloud.com`
- 🌐 [LinkedIn Profile](https://www.linkedin.com/in/lars-augustat/)

# AI Narrative Index (AINI)

This repository contains the full research pipeline for constructing the **AI Narrative Index (AINI)** — a time-series measure of how artificial intelligence is represented and emotionally framed in financial news. The index is derived from **Wall Street Journal (WSJ)** articles from **2023 to 2025** and used as an independent variable to estimate asset returns.

The project integrates **Transformer-based NLP**, **manual annotation**, **deep learning fine-tuning**, and **econometric and information-theoretic inference** — all within a modular and reproducible architecture.

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

### 3. AINI via ChatGPT Labeling and Prompt Engineering *(modules under construction, uploaded soon)*

- WSJ articles are labeled for **AI relevance** using **GPT-4 via the OpenAI API**, with few-shot prompting and *Chain-of-Thought* reasoning.

- Only articles classified as **AI-related** are passed to the **pretrained FinBERT model** to predict their **sentiment**.

- As in Method 2, sentiment scores are aggregated into a daily AINI time series — but here restricted to **GPT-identified AI narratives**.

- Ongoing evaluation focuses on:
  - Agreement between GPT labels and human annotations  
  - Stability across prompt designs and reasoning chains  
  - Effectiveness of GPT labeling as a **scalable weak supervision** method

## Statistical Testing & Causal Inference

The project applies **rigorous time series diagnostics and hypothesis testing** to analyze the link between narrative indices and financial markets:

### Stationarity Testing

- Augmented Dickey-Fuller (ADF)
- Phillips-Perron (PP)

Used to validate input variables before time series modeling.

### Granger Causality (GC)

Tests predictive causality between AINI and log returns or volatility indices using:

- **bootstrapping**: to create empirical p-values
- **Benjamini-Hochberg correction** to reduce the false-discovery-rate

Supports both directions: `AINI ➝ returns` and `returns ➝ AINI`.

### Transfer Entropy (TE)

- KSG estimator for nonlinear, asymmetric dependencies
- IDTxl framework with permutation-based significance testing

## Project Structure

<pre>
## Project Structure

<pre>
AI_narrative_index/
│
├── src/
│   ├── fetch_data/
│   │   ├── load_financial_data.py              # Download and prepare financial market data
│   │   ├── wsj_archive_crawler.py              # Crawl WSJ archive pages to collect article URLs
│   │   ├── wsj_archive_scraper.py              # Scrape full article content from collected WSJ URLs
│   │   └── run_predict_AINI.py                 # Apply trained AINI model to WSJ text for predictions
│
│   ├── preprocessing/
│   │   ├── clean_database.py                   # Clean and filter articles by section, length, and duplicates
│   │   ├── corpus_cleaning.py                  # Remove UI/meta elements from WSJ article text
│   │   ├── reduce_db_for_sentiment.py          # Extract article subset for sentiment prediction
│   │   ├── section_filtering.py                # Filter irrelevant WSJ sections from the database
│   │   └── simple_ai_filter.py                 # Flag articles that mention AI-related terms
│
│   ├── annotation/
│   │   ├── comparing_annotations.py            # Resolve disagreements between annotators
│   │   └── label_articles.py                   # Launch interactive tool for AI and hype annotation
│
│   ├── modelling/
│   │   ├── ai_windows.py                       # Extract context-aware snippets around AI keywords
│   │   ├── predict_binary_AINI_FinBERT.py      # Classify articles (AI-related vs. not) using FinBERT
│   │   ├── predict_AINI_FinBERT_window.py      # Classify with windowed context using FinBERT
│   │   ├── CustomFinBERT.py                    # Define custom FinBERT model with dropout & class weights
│   │   ├── stationarity_testing.py             # Perform ADF and PP stationarity tests
│   │   ├── transfer_entropy.py                 # Estimate Transfer Entropy between AINI and financial variables
│   │   ├── granger_causality.py                # Run Granger causality with heteroskedasticity-aware bootstrapping
│   │   └── construct_AINI_variables.py         # Build daily AINI index with normalization, EMA, growth etc.
│
│   ├── visualizations/
│   │   └── plot_granger_causality.py           # Visualize Granger causality outputs
│
│   ├── scripts/                                # CLI entry points for reproducible execution
│   │   ├── run_create_database.py              # Initialize article database schema and structure
│   │   ├── run_wsj_scraper.py                  # Run WSJ crawler and scraper
│   │   ├── run_clean_database.py               # Clean article database by year
│   │   ├── run_reduce_db_for_sentiment.py      # Reduce database for sentiment analysis
│   │   ├── run_predict_investor_sentiment.py   # Apply sentiment prediction using standard FinBERT
│   │   ├── run_predict_binary_AINI_FinBERT.py  # Run binary AINI classification pipeline
│   │   ├── run_predict_AINI_FinBERT_window.py  # Run context-aware classification on snippets
│   │   └── run_construct_AINI_variables.py     # Construct final AINI index file for modeling
│
├── notebooks/
│   ├── compare_resolve_annotations.ipynb       # Resolve annotation disagreements interactively
│   ├── training_FinBERT_annotation.ipynb       # Fine-tune FinBERT on AI-relatedness labels
│   ├── investigating_FinBERT_annotations.ipynb # Inspect FinBERT predictions across configurations
│   ├── exploratory_analysis_wsj.ipynb          # Explore WSJ article dataset and structure
│   ├── sample_articles.ipynb                   # Sample articles for manual annotation
│   ├── label_articles.ipynb                    # Manually annotate AI and hype relevance
│   ├── test_stationarity.ipynb                 # Perform visual and statistical stationarity tests
│   ├── estimate_transfer_entropy.ipynb         # Estimate transfer entropy for causal analysis
│   ├── estimate_granger_causality.ipynb        # Apply Granger causality to AINI and financial data
│   ├── analyse_gc_results.ipynb                # Inspect and visualize Granger causality results
│   ├── visualize_ksg.ipynb                     # Explain the Kraskov estimator for entropy
│   └── visualize_aini_variables.ipynb          # Explore AINI index trends and dynamics
│
├── data/
│   ├── raw/                                    # Contain scraped articles and raw financial data
│   ├── interim/                                # Store intermediate outputs (e.g. annotated WSJ batches)
│   └── processed/
│       ├── variables/                          # Store constructed variables (e.g. AINI, sentiment, TE)
│       └── articles/                           # Store cleaned and filtered article texts
│
└── models/                                     # Store fine-tuned FinBERT and sentiment models

</pre>
---

### Note

The scraped data is used strictly for academic and scientific research purposes and is not shared publicly due to copyright and licensing restrictions.
The trained model can be shared upon request, subject to data sharing considerations (e.g., size and usage context).

## Author and Context

This repository forms the empirical foundation of a master's thesis in Economics, with the aim of combining:

- **Behavioral finance and narrative theory**
- **NLP and deep learning with Transformers**
- **Causal inference using both econometrics and information theory**

---

## Contact

- 📧 `lars.augustat@icloud.com`
- 🌐 [LinkedIn Profile](https://www.linkedin.com/in/lars-augustat/)

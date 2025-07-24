 🧠 AI Narrative Index (AINI)

This repository contains the full research pipeline for constructing the **AI Narrative Index (AINI)** — a time-series measure of how artificial intelligence is represented and emotionally framed in financial news. The index is derived from **Wall Street Journal (WSJ)** articles from **2023 to 2025**, and used to assess impacts on asset returns, investor sentiment, and volatility.

The project integrates **Transformer-based NLP**, **manual annotation**, **deep learning fine-tuning**, and **econometric and information-theoretic causal inference** — all within a modular, reproducible, and industry-ready architecture.

---

## 🎯 Research Objectives

- **Develop multiple variants of an AI Narrative Index (AINI)** using both **supervised fine-tuning** and **zero-shot inference**.

- **Evaluate the temporal and causal effects** of narrative hype on market dynamics using Granger causality and transfer entropy.

- **Ensure scientific rigor** through pre-annotation protocols, dual-labeller verification, and formal statistical diagnostics.

---

## 🧩 Construction of the AINI Index

This project explores **two distinct and complementary approaches** to quantifyin

### 1. AINI via Manual Annotation and FinBERT Fine-Tuning

- A manually annotated dataset (2023–2024) was created in collaboration with a **professional second annotator**, focusing on:
  - **AI Relevance** (binary classification)  
  - **Narrative Hype Level** (sentiment-style score)

- A custom **FinBERT model** is trained on this dataset, using:
  - Class-weighted loss  
  - Window-based context extraction  
  - Early stopping and evaluation logging

- The result is a **binary AI index** used as a base for further analysis.

---

### 2. AINI via Standard FinBERT and Snippet Reduction

- The pretrained **FinBERT model** ([ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)) is applied to **pre-cleaned WSJ articles**.

- Articles are first **reduced using preprocessing and dimensionality reduction techniques**, including:
  - Heuristic snippet extraction around AI keywords  
  - BERT-based contextual windowing

- The resulting **probabilistic sentiment outputs** are aggregated into a daily AINI time series.

---

### 3. AINI via ChatGPT Labeling and Prompt Engineering *(modules under construction, uploaded soon)*

- WSJ articles are labeled for **AI relevance** using **GPT-4 via the OpenAI API**, using few-shot prompting with *Chain-of-Thought* reasoning.

- Only articles classified as **AI-related** are then passed to the **pretrained FinBERT model** to predict their **sentiment** (positive, neutral, negative).

- As in Method 2, the resulting sentiment scores are aggregated into a daily AINI time series—this time restricted to **GPT-identified AI narratives**.

- Ongoing evaluation focuses on:
  - Agreement between GPT labels and human annotations  
  - Stability across prompt designs and reasoning chains  
  - Effectiveness of GPT labeling as a **scalable weak supervision** approach

## 📐 Statistical Testing & Causal Inference

The project applies **rigorous time series diagnostics and hypothesis testing** to analyze the link between narrative indices and financial markets:

### 🔬 Stationarity Testing

- Augmented Dickey-Fuller (ADF)
- Phillips-Perron (PP)

Used to validate input variables before time series modeling.

### 🔁 Granger Causality (GC)

- Bootstrap-based F-tests
- False discovery rate (FDR) control (Benjamini-Hochberg)
- Heteroskedasticity-consistent diagnostics

Estimates predictive causality between AINI and log returns or volatility indices.

### 🔄 Transfer Entropy (TE)

- KSG estimator for nonlinear, asymmetric dependencies
- IDTxl framework with permutation-based significance testing

## 📁 Project Structure

<pre>
AI_narrative_index/
│
├── src/
│   ├── fetch_data/
│   │   ├── load_financial_data.py              # Downloads and prepares financial market data
│   │   ├── wsj_archive_crawler.py              # Crawls WSJ archive pages to collect article URLs
│   │   ├── wsj_archive_scraper.py              # Scrapes full article content from collected WSJ URLs
│   │   └── run_predict_AINI.py                 # Applies trained AINI model to WSJ text for predictions
│     
│   ├── visualizations/
│       └── plot_granger_causality.py           # Visualization of Granger causality outputs
│  
│   ├── preprocessing/
│   │   ├── clean_database.py                   # Cleans and filters articles by section, length, and duplicates
│   │   ├── corpus_cleaning.py                  # Removes UI/meta elements from WSJ article text
│   │   ├── reduce_db_for_sentiment.py          # Extracts subset of articles for sentiment prediction
│   │   ├── section_filtering.py                # Filters irrelevant WSJ sections from database
│   │   └── simple_ai_filter.py                 # Flags articles that mention AI-related terms
│   
│   ├── annotation/
│   │   ├── comparing_annotations.py            # Resolves annotation disagreements between author and second coder
│   │   └── label_articles.py                   # Interactive annotation tool for AI relevance and hype
│
│   ├── modelling/
│   │   ├── ai_windows.py                       # Extracts context-aware text snippets based on AI keywords
│   │   ├── predict_binary_AINI_FinBERT.py      # Binary classification (AI-related or not) using custom FinBERT
│   │   ├── predict_AINI_FinBERT_window.py      # Classifies AI relevance with windowed context
│   │   ├── CustomFinBERT.py                    # Custom FinBERT model with dropout and class weights
│   │   ├── stationarity_testing.py             # Performs ADF and PP tests for time series stationarity
│   │   ├── transfer_entropy.py                 # Estimates Transfer Entropy between AINI and financial variables
│   │   ├── granger_causality.py                # Granger causality with heteroskedasticity-aware bootstrapping
│   │   └── construct_AINI_variables.py         # Builds daily AINI index with normalization, EMA, growth etc.
│
│   ├── scripts/
        # CLI entry points — wrap functions from `modelling/`, `preprocessing/`, and `fetch_data/`
│   │   ├── run_create_database.py              # Initializes article database schema and structure
│   │   ├── run_wsj_scraper.py                  # Runs crawler and scraper for WSJ articles
│   │   ├── run_clean_database.py               # Cleans article databases by year
│   │   ├── run_reduce_db_for_sentiment.py      # Extracts articles suitable for sentiment analysis
│   │   ├── run_predict_investor_sentiment.py   # Applies sentiment prediction using standard FinBERT
│   │   ├── run_predict_binary_AINI_FinBERT.py  # Runs binary classification pipeline on WSJ articles
│   │   ├── run_predict_AINI_FinBERT_window.py  # Runs context-aware classification on snippets
│   │   └── run_construct_AINI_variables.py     # Builds final AINI index file for modeling
│   
│
├── notebooks/
│   ├── compare_resolve_annotations.ipynb       # Interactive resolution of annotation disagreements
│   ├── training_FinBERT_annotation.ipynb       # Fine-tunes FinBERT on binary AI-relatedness annotations
│   ├── investigating_FinBERT_annotations.ipynb # Inspects FinBERT predictions across multiple configurations
│   ├── exploratory_analysis_wsj.ipynb          # First look into WSJ article dataset and structure
│   ├── sample_articles.ipynb                   # Samples WSJ articles for manual annotation
│   ├── label_articles.ipynb                    # Manual labeling of AI and hype annotations
│   ├── test_stationarity.ipynb                 # Visual and statistical tests of time series stationarity
│   ├── estimate_transfer_entropy.ipynb         # Transfer entropy analysis for causal relationships
│   ├── estimate_granger_causality.ipynb        # Applies Granger causality to AINI and financial data
│   ├── analyse_gc_results.ipynb                # Detailed inspection and plotting of GC outcomes
│   ├── visualize_ksg.ipynb                     # Visual explanation of the Kraskov estimator (TE)
│   └── visualize_aini_variables.ipynb          # Plots and explores AINI index dynamics``` 

</pre>
---

### ℹ️ Note

The scraped data is used strictly for academic and scientific research purposes and is not shared publicly due to copyright and licensing restrictions.
The trained model can be shared upon request, subject to data sharing considerations (e.g., size and usage context).

## 💼 Author and Context

This repository forms the empirical foundation of a master's thesis in Economics, with the aim of combining:

- **Behavioral finance and narrative theory**
- **NLP and deep learning with Transformers**
- **Causal inference using both econometrics and information theory**

---

## 📬 Contact

- 📧 `lars.augustat@icloud.com`
- 🌐 [LinkedIn Profile](https://www.linkedin.com/in/lars-augustat/)
- 📄 Thesis summary available on request

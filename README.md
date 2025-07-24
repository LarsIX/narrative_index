
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

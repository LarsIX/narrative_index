# Notes on Finetuning & Labeling

To construct the initial AI narrative index, a curated dictionary of AI-related keywords was compiled, drawing on terminology from Stanford University, the International Monetary Fund (IMF), and the UK Parliament (see Thesis and ai_glossaries.csv). The following terms were included:

### AI-related keywords:
AI, A.I., AGI, artificial intelligence, artificial general intelligence, human-level AI, human-centered artificial intelligence, large language model, LLM, algorithmic bias, machine learning, supervised learning, unsupervised learning, reinforcement learning, deep learning, neural network, artificial neural network, transformer model, transformers, fine-tuning, generative AI, generative adversarial network, GANs, prompt engineering, prompt, hallucination, autonomous systems, speech recognition, facial recognition, social chatbots, human-robot interaction, automated decision-making, superintelligence, computer vision, deepfake, foundation model, frontier AI, graphical processing unit, GPU, interpretability, open-source, responsible AI, training dataset.

Using this keyword set, AI-relevant snippets were extracted and classified via a fine-tuned FinBERT model. On the 2024 training and evaluation set, the model achieved strong performance, with weighted and unweighted F1-scores close to or above 90%. The average normalized AI Narrative Index (AINI) for 2024 was approximately 0.0792, indicating that around 7.9% of daily articles discussed AI.

However, when applying the same model to the unlabeled 2023 corpus, the mean AINI increased sharply to ≈ 0.3984, suggesting an implausibly high prevalence of AI-related content. Manual inspection revealed a key source of distortion: uncleaned hyperlink text (e.g., footers or sidebar links) frequently included AI-related phrases without the surrounding article being substantively about AI.

A notable example is the headline:

    “MIT says it no longer stands behind AI research paper”

This phrase appeared 1,520 times in 2023. Excluding all articles containing it reduced the mean AINI to ≈ 0.1346 (or 0.1336 in the subset post-March 2023), significantly correcting the inflation. All instances had been labeled as true positives by the model.

Further manual review identified additional problematic phrases, including:

    "how to use generative ai tools for everyday tasks"

    "ai can be a force for deregulation"

    "ai helped heal my chronic pain"

    "massive ai chip deal"

Excluding articles containing these phrases (total removal: 1,618 articles) further reduced the 2023 mean AINI to ≈ 0.1061, closely matching the drop rate, and aligning the measure with more realistic levels.

Another systematic false positive stemmed from the keyword "prompt". While relevant in technical contexts (e.g., "prompt engineering"), its broader usage—e.g.,

    “Possibly ‘Bogus’ 911 Call Prompted Senate Buildings’ Lockdown, Police Say”
    led to misclassifications. Of 521 articles containing "prompt", 50.29% were labeled AI-positive, despite many not discussing AI. Removing the generic keyword “prompt” from the filter substantially improved classification quality.

Based on these insights, all results reported in the thesis are derived from the cleaned and subsetted 2023 corpus (from April onward), with problematic articles excluded using the clean_database function located in root/cleaning/clean_database.py. The script can be executed via Typer using the CLI flag --year 2023 or --year 2024..
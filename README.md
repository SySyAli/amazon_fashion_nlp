# Temporal vs. Reviewer Lifespan Drift in Amazon Fashion Reviews    

## Problem Statement

Most NLP systems treat consumer reviews as independent documents, disconnected from their author and the moment they were written. This is a meaningful limitation: the vocabulary available to a reviewer depends heavily on when they are writing. For example, COVID-19 introduced entirely new product categories and language into the Amazon corpus. Past work have found that classifiers trained on one era degrade when deployed in another. The cause of that degradation, however, is not well understood. Is it driven by individual reviewers evolving linguistically over their careers, or by the historical era in which they happen to be writing? Prior works have studied reviewer lifecycles but did not isolate these two sources of variation. This project addresses that gap directly through a controlled parallel experiment on 2.4 million Amazon Fashion reviews spanning 2002--2023.

## Notebooks

### `nlp_project_part_1.ipynb` — Exploratory Lexical Analysis

The first checkpoint explores vocabulary shift across time using Dirichlet Log-odds Analysis (DLA) based on Monroe et al. (2008). The full Amazon Fashion corpus (~2.4M reviews) is streamed from HuggingFace and split into three non-overlapping time windows (2014–2016, 2017–2019, 2020–2022). Within each window, DLA is run comparing 5-star and 1-star reviews to surface the most statistically distinctive tokens for each polarity. A custom regex tokenizer handles HTML tags, prices, numbers, contractions, and hyphenated compounds. Results include per-window bar charts of top distinctive tokens, a cross-window comparison to identify words unique to a single era, and a set of stable words that remain distinctive across all three windows. This part establishes that the lexicon of sentiment does shift meaningfully over time and motivates the controlled experiment in the final notebook.

---

### `nlp_project_part_2.ipynb` — Sentiment Classification Baselines

The second checkpoint builds and evaluates three sentiment classifiers to establish performance baselines before studying temporal drift. The corpus is balanced by class (100K reviews per class for binary; 50K per class for 5-class) and split 80/20 into train and test sets. Three feature representations are compared:

- **CountVectorizer + Logistic Regression** — bag-of-words unigrams
- **TF-IDF (1+2-gram) + Logistic Regression** — sublinear TF-IDF with bigrams
- **Word2Vec (200d skip-gram) + Logistic Regression** — TF-IDF-weighted average embeddings

For binary classification (1-star vs 5-star), all three methods reach ~95% accuracy. For 5-class classification (all star ratings), accuracy falls to ~51–52%, well above the 20% chance baseline but reflecting the inherent difficulty of distinguishing adjacent ratings. Word embeddings are visualized with UMAP, grouping terms by sentiment polarity, material/fit semantics, and era-distinctive vocabulary surfaced in Part 1. Mockup visualizations preview the temporal drift cross-validation matrix and a hypothetical BERT comparison planned for the final notebook.

---

### `nlp_final.ipynb` — Isolating Temporal vs. Lifespan Drift

The final notebook contains the core experiment. Two parallel axes of variation are constructed from the same 2.4M reviews:

**Temporal axis** — Reviews are split into an early era (2014–2016, ~475K reviews) and a late era (2020–2022, ~793K reviews), separated by a buffer to avoid overlap.

**Lifespan axis** — Reviewers who posted at least 5 reviews and then went inactive for over a year are classified as "departed." Their reviews are ranked chronologically within their career and binned into deciles; early-career reviews (bins 1–4) are compared against late-career reviews (bins 7–10), yielding ~9,900 and ~11,000 reviews respectively across 1,682 eligible users.

Each axis is analyzed with four methods:

1. **DLA** — Dirichlet log-odds z-scores identify the most distinctive vocabulary on each side of both splits. Temporal DLA captures the COVID-19 vocabulary shift (masks, breathable, photo, seller). Lifespan DLA shows subtler stylistic changes.

2. **TF-IDF logistic regression** — A balanced binary classifier (50K per class, 75/25 split) is trained to distinguish early from late reviews on each axis. Temporal classification significantly outperforms lifespan classification, consistent with the hypothesis that era dominates individual career evolution.

3. **Bigram language model perplexity** — A smoothed bigram LM trained on early-era reviews is evaluated on held-out early reviews (self-test) and late-era reviews. A Mann-Whitney U test checks whether late reviews are significantly more surprising. The same procedure is applied to the lifespan split.

4. **Word2Vec centroid drift** — A 100-dimensional CBOW Word2Vec model is trained on the full corpus. Cosine distances between group centroids are measured for both splits. The raw lifespan centroid distance (0.0024) is an order of magnitude smaller than the temporal distance (0.0346). Monthly centroids are computed for 2014–2022, and each early/late-career review has its monthly centroid subtracted to remove era-level signal. After this residual correction, the lifespan centroid distance rises to 0.0321 — nearly matching the temporal distance — indicating that the apparent absence of lifespan drift is a confound: departed users' early and late reviews happen to fall in different eras.

Visualizations include side-by-side PCA plots of raw and residual document embeddings, a month-over-month semantic drift line chart (with COVID-19 shading), a UMAP trajectory of monthly centroids colored by time with word landmarks, and a rating-by-life-stage plot showing how mean star rating changes across a reviewer's career. A Claude API zero-shot classification experiment probes whether a large language model can detect era or career stage from review text alone.

---

## Authors

Zayaan Rahman and Syed Ali — Vanderbilt University, Spring 2026

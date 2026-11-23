# Results on the Clough & Stevenson corpus

95 answers (cut: 19, light: 19, heavy: 19, non: 38), each compared with the source text for its task. Feature extraction: 9.9 ms per pair.

## 1. Single measures: reused (cut/light/heavy) vs original

AUC with 95% bootstrap interval. Last column is the mean score per category.

| measure | ROC-AUC | avg. precision | cut / light / heavy / non |
|---|---|---|---|
| containment_1 | 0.975 [0.935, 1.000] | 0.988 | 0.81 / 0.80 / 0.68 / 0.40 |
| containment_3 | 0.964 [0.912, 1.000] | 0.983 | 0.66 / 0.51 / 0.30 / 0.02 |
| containment_5 | 0.965 [0.925, 0.995] | 0.975 | 0.61 / 0.38 / 0.20 / 0.00 |
| jaccard_3 | 0.965 [0.914, 0.999] | 0.984 | 0.35 / 0.25 / 0.15 / 0.01 |
| lcs | 0.962 [0.919, 0.991] | 0.979 | 0.71 / 0.71 / 0.50 / 0.24 |
| winnow | 0.961 [0.916, 0.995] | 0.975 | 0.63 / 0.41 / 0.22 / 0.01 |
| word_cosine | 0.970 [0.921, 1.000] | 0.986 | 0.60 / 0.57 / 0.47 / 0.18 |
| char_cosine | 0.970 [0.920, 1.000] | 0.987 | 0.75 / 0.74 / 0.69 / 0.47 |
| tfidf_cosine (corpus IDF) | 0.970 [0.916, 1.000] | 0.987 | 0.51 / 0.46 / 0.34 / 0.09 |
| lsa_cosine (100 dims) | 0.970 [0.916, 1.000] | 0.987 | 0.52 / 0.46 / 0.34 / 0.09 |
| v2.0 combined score (old) | 0.971 [0.924, 1.000] | 0.987 | 0.45 / 0.38 / 0.28 / 0.08 |

## 2. Flagging reused text (leave-one-task-out)

| method | accuracy | F1 (reused) | ROC-AUC |
|---|---|---|---|
| v2.0 score, its default threshold 0.75 | 0.442 | 0.131 | - |
| v2.0 score, threshold tuned in CV | 0.937 | 0.946 | - |
| containment_3, threshold tuned in CV | 0.937 | 0.946 | - |
| winnow, threshold tuned in CV | 0.926 | 0.936 | - |
| lcs, threshold tuned in CV | 0.863 | 0.887 | - |
| **logistic regression, all features (shipped model)** | **0.958** | **0.964** | 0.971 |

## 3. Predicting the type of reuse (cut / light / heavy / non)

- lexical features (shipped model): accuracy **0.779**, macro-F1 **0.723**
- same + corpus TF-IDF and LSA: accuracy 0.779, macro-F1 0.722
- if cut and light are merged into one class: accuracy 0.821

Confusion matrix (rows = true, columns = predicted):

|  | cut | light | heavy | non |
|---|---|---|---|---|
| cut | 16 | 1 | 0 | 2 |
| light | 3 | 9 | 7 | 0 |
| heavy | 2 | 4 | 11 | 2 |
| non | 0 | 0 | 0 | 38 |

## 4. Ablation: drop one feature, retrain

Change relative to the full model. Negative = the feature was helping.

| dropped | Δ macro-F1 (4-way) | Δ AUC (binary) |
|---|---|---|
| containment_1 | -0.012 | -0.005 |
| containment_3 | +0.000 | +0.000 |
| containment_5 | -0.001 | +0.000 |
| jaccard_3 | -0.001 | +0.001 |
| lcs | -0.015 | -0.002 |
| winnow | -0.012 | +0.001 |
| word_cosine | -0.001 | +0.000 |
| char_cosine | +0.014 | +0.001 |

## 5. Same topic, no copying

129 pairs of independently written answers to the same question. Mean reuse score 0.126; flagged at 0.5: 5 (3.9%). (Model trained on all answer/source pairs, so this is a check on transfer, not CV.)

## 6. MinHash LSH as a pre-filter

Share of answer/source pairs that LSH proposes as candidates. Higher recall on cut/light/heavy is good. The last column is how many of all 4950 document pairs would still need full scoring. That number is high here because the corpus is 5 topics with ~20 answers each; most surviving pairs are same-topic pairs.

| index | recall cut / light / heavy | non proposed | pairs kept |
|---|---|---|---|
| whole doc, 1-gram shingles, 64x2 (J≈0.12) | 1.00 / 1.00 / 0.89 | 0.61 | 21.5% |
| whole doc, 1-gram shingles, 32x4 (J≈0.42) | 0.58 / 0.58 / 0.21 | 0.03 | 2.2% |
| whole doc, 2-gram shingles, 64x2 (J≈0.12) | 0.89 / 0.84 / 0.53 | 0.05 | 4.2% |
| whole doc, 2-gram shingles, 32x4 (J≈0.42) | 0.26 / 0.32 / 0.05 | 0.00 | 0.6% |
| 50-word windows, stride 25, 32x4 | 0.89 / 1.00 / 0.74 | 0.11 | 5.2% |
| 30-word windows, stride 15, 32x4 | 0.89 / 0.95 / 0.95 | 0.11 | 6.5% |
| 25-word windows, stride 8, 16x8 | 0.89 / 0.74 / 0.53 | 0.00 | 2.8% |
| 25-word windows, stride 8, 32x4 **(default)** | 0.89 / 1.00 / 1.00 | 0.26 | 8.0% |

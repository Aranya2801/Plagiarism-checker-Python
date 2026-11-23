"""Evaluation on the Clough & Stevenson corpus.

    python experiments/benchmark.py --data path/to/corpus

Writes experiments/results/ (tables + plot) and, with --save-model, retrains
the shipped model on the full corpus and writes plagiarism_checker/data/model.json.

All cross-validation is leave-one-task-out: train on four topics, test on the
fifth. Random splits would leak, because answers to the same task share a
source text and a lot of vocabulary.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from experiments.corpus import CATEGORIES, load  # noqa: E402
from plagiarism_checker.features import FEATURES, CorpusIndex, pair_features  # noqa: E402
from plagiarism_checker.fingerprint import ChunkedCandidates, LSHIndex, MinHasher, lsh_threshold  # noqa: E402
from plagiarism_checker.model import DEFAULT_MODEL, ReuseModel  # noqa: E402

RESULTS = Path(__file__).with_name("results")


# ---------------------------------------------------------------------------
# the v2.0 scoring from the previous version of this repo, kept as a baseline
# (2-document TF-IDF cosine, unigram Jaccard, trigram Jaccard, 0.5/0.25/0.25)
# ---------------------------------------------------------------------------

_V2_STOP = set("""i me my myself we our ours ourselves you your yours yourself he him his
she her hers it its they them their what which who whom this that these those am is are
was were be been being have has had having do does did doing a an the and but if or
because as until while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then
once here there when where why how all both each few more most other some such no nor
not only own same so than too very s t can will just don should now""".split())


def _v2_tokens(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower().split()
    return [t for t in text if len(t) >= 3 and t not in _V2_STOP]


def legacy_v2_score(t1: str, t2: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    a, b = _v2_tokens(t1), _v2_tokens(t2)
    try:
        m = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform([" ".join(a), " ".join(b)])
        cos = float((m[0] @ m[1].T).toarray()[0, 0])
    except ValueError:
        cos = 0.0
    sa, sb = set(a), set(b)
    jac = len(sa & sb) / len(sa | sb) if sa | sb else 1.0
    ga = {tuple(a[i:i + 3]) for i in range(len(a) - 2)}
    gb = {tuple(b[i:i + 3]) for i in range(len(b) - 2)}
    ng = len(ga & gb) / len(ga | gb) if ga | gb else 1.0
    return 0.5 * cos + 0.25 * jac + 0.25 * ng


# ---------------------------------------------------------------------------

def bootstrap_auc(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(vals, [2.5, 97.5])


def make_lr(binary=False):
    # C=1 is sklearn's default. I tried 0.1-10: 4-way accuracy stays within
    # 0.77-0.78 for C in [0.3, 3] and drops below 0.3, binary accuracy doesn't move.
    return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced" if binary else None)


def cv_predict(X, y, groups, binary=False):
    """Leave-one-group-out predictions. Returns (pred labels, proba matrix, classes)."""
    classes = sorted(set(y))
    proba = np.zeros((len(y), len(classes)))
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        sc = StandardScaler().fit(X[tr])
        clf = make_lr(binary).fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))
        for j, c in enumerate(clf.classes_):
            proba[te, classes.index(c)] = p[:, j]
    pred = np.array(classes)[proba.argmax(1)]
    return pred, proba, classes


def cv_threshold(score, yb, groups):
    """Best single threshold picked on training folds only, applied to the held-out fold."""
    pred = np.zeros_like(yb)
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        cands = np.unique(score[tr])
        best = max(cands, key=lambda t: accuracy_score(yb[tr], score[tr] >= t))
        pred[te] = score[te] >= best
    return pred


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with file_information.csv")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    corpus = load(args.data)
    answers = corpus.answers
    print(f"{len(corpus.sources)} sources, {len(answers)} answers")

    t0 = time.perf_counter()
    feats = [pair_features(a.doc, corpus.sources[a.task]) for a in answers]
    t_feat = (time.perf_counter() - t0) / len(answers)
    X = np.array([[f[k] for k in FEATURES] for f in feats])
    cat = np.array([a.category for a in answers])
    task = np.array([a.task for a in answers])
    yb = (cat != "non").astype(int)

    index = CorpusIndex(corpus.all_docs)
    src_of = {a.doc.name: corpus.sources[a.task].name for a in answers}
    extra = {
        "tfidf_cosine (corpus IDF)": np.array([index.tfidf_cosine(a.doc.name, src_of[a.doc.name]) for a in answers]),
        "lsa_cosine (100 dims)": np.array([index.lsa_cosine(a.doc.name, src_of[a.doc.name]) for a in answers]),
    }
    legacy = np.array([legacy_v2_score(a.doc.text, corpus.sources[a.task].text) for a in answers])

    RESULTS.mkdir(exist_ok=True)
    report = ["# Results on the Clough & Stevenson corpus\n",
              f"{len(answers)} answers ({', '.join(f'{c}: {(cat == c).sum()}' for c in CATEGORIES)}), "
              f"each compared with the source text for its task. "
              f"Feature extraction: {t_feat * 1000:.1f} ms per pair.\n"]

    # -- 1. single measures, reuse vs original --------------------------------
    rows = []
    singles = {k: X[:, i] for i, k in enumerate(FEATURES)}
    singles.update(extra)
    singles["v2.0 combined score (old)"] = legacy
    for name, s in singles.items():
        lo, hi = bootstrap_auc(yb, s)
        means = " / ".join(f"{s[cat == c].mean():.2f}" for c in CATEGORIES)
        rows.append((name, f"{roc_auc_score(yb, s):.3f} [{lo:.3f}, {hi:.3f}]",
                     f"{average_precision_score(yb, s):.3f}", means))
    report += ["## 1. Single measures: reused (cut/light/heavy) vs original\n",
               "AUC with 95% bootstrap interval. Last column is the mean score per category.\n",
               md_table(["measure", "ROC-AUC", "avg. precision", "cut / light / heavy / non"], rows), ""]

    # -- 2. binary decision accuracy ------------------------------------------
    rows = []
    v2_pred = (legacy >= 0.75).astype(int)  # what the old CLI would have flagged
    rows.append(("v2.0 score, its default threshold 0.75",
                 f"{accuracy_score(yb, v2_pred):.3f}", f"{f1_score(yb, v2_pred):.3f}", "-"))
    p = cv_threshold(legacy, yb, task)
    rows.append(("v2.0 score, threshold tuned in CV", f"{accuracy_score(yb, p):.3f}", f"{f1_score(yb, p):.3f}", "-"))
    for k in ("containment_3", "winnow", "lcs"):
        p = cv_threshold(singles[k], yb, task)
        rows.append((f"{k}, threshold tuned in CV", f"{accuracy_score(yb, p):.3f}", f"{f1_score(yb, p):.3f}", "-"))

    _, proba4, classes4 = cv_predict(X, cat, task)
    reuse4 = 1 - proba4[:, classes4.index("non")]
    p = (reuse4 >= 0.5).astype(int)
    rows.append(("**logistic regression, all features (shipped model)**", f"**{accuracy_score(yb, p):.3f}**",
                 f"**{f1_score(yb, p):.3f}**", f"{roc_auc_score(yb, reuse4):.3f}"))
    report += ["## 2. Flagging reused text (leave-one-task-out)\n",
               md_table(["method", "accuracy", "F1 (reused)", "ROC-AUC"], rows), ""]

    # -- 3. four-way classification -------------------------------------------
    pred4 = np.array(classes4)[proba4.argmax(1)]
    acc4, mf1 = accuracy_score(cat, pred4), f1_score(cat, pred4, average="macro")
    cm = confusion_matrix(cat, pred4, labels=list(CATEGORIES))
    merged = lambda a: np.where(np.isin(a, ["cut", "light"]), "cut+light", a)  # noqa: E731
    acc3 = accuracy_score(merged(cat), merged(pred4))
    Xe = np.column_stack([X] + list(extra.values()))
    pred_e, _, _ = cv_predict(Xe, cat, task)
    report += ["## 3. Predicting the type of reuse (cut / light / heavy / non)\n",
               f"- lexical features (shipped model): accuracy **{acc4:.3f}**, macro-F1 **{mf1:.3f}**",
               f"- same + corpus TF-IDF and LSA: accuracy {accuracy_score(cat, pred_e):.3f}, "
               f"macro-F1 {f1_score(cat, pred_e, average='macro'):.3f}",
               f"- if cut and light are merged into one class: accuracy {acc3:.3f}\n",
               "Confusion matrix (rows = true, columns = predicted):\n",
               md_table([""] + list(CATEGORIES), [[c] + list(r) for c, r in zip(CATEGORIES, cm)]), ""]

    # -- 4. ablation -----------------------------------------------------------
    rows = []
    for i, k in enumerate(FEATURES):
        keep = [j for j in range(len(FEATURES)) if j != i]
        pr, pb, cl = cv_predict(X[:, keep], cat, task)
        rows.append((k, f"{f1_score(cat, pr, average='macro') - mf1:+.3f}",
                     f"{roc_auc_score(yb, 1 - pb[:, cl.index('non')]) - roc_auc_score(yb, reuse4):+.3f}"))
    report += ["## 4. Ablation: drop one feature, retrain\n",
               "Change relative to the full model. Negative = the feature was helping.\n",
               md_table(["dropped", "Δ macro-F1 (4-way)", "Δ AUC (binary)"], rows), ""]

    # -- 5. topical false positives -------------------------------------------
    # Two people answering the same question independently share vocabulary.
    # Does the model mistake that for copying?
    by_task_non = {}
    for a in answers:
        if a.category == "non":
            by_task_non.setdefault(a.task, []).append(a.doc)
    model_all = fit_full(X, cat)
    fp_scores = [model_all.predict(pair_features(d1, d2)).reuse_score
                 for docs in by_task_non.values() for d1, d2 in combinations(docs, 2)]
    fp_scores = np.array(fp_scores)
    report += ["## 5. Same topic, no copying\n",
               f"{len(fp_scores)} pairs of independently written answers to the same question. "
               f"Mean reuse score {fp_scores.mean():.3f}; flagged at 0.5: "
               f"{(fp_scores >= 0.5).sum()} ({(fp_scores >= 0.5).mean():.1%}). "
               "(Model trained on all answer/source pairs, so this is a check on transfer, not CV.)\n"]

    # -- 6. LSH candidate recall ----------------------------------------------
    n_all = len(corpus.all_docs) * (len(corpus.all_docs) - 1) // 2

    def recall_row(label, cand):
        cand = {tuple(sorted(p)) for p in cand}
        rec = {c: np.mean([tuple(sorted((a.doc.name, corpus.sources[a.task].name))) in cand
                           for a in answers if a.category == c]) for c in CATEGORIES}
        return (label, f"{rec['cut']:.2f} / {rec['light']:.2f} / {rec['heavy']:.2f}",
                f"{rec['non']:.2f}", f"{len(cand) / n_all:.1%}")

    rows = []
    for shingle_n, bands, r in ((1, 64, 2), (1, 32, 4), (2, 64, 2), (2, 32, 4)):
        mh, idx = MinHasher(num_perm=bands * r), LSHIndex(bands, r)
        for d in corpus.all_docs:
            w = d.content_words
            idx.add(d.name, mh.signature(" ".join(w[i:i + shingle_n]) for i in range(len(w) - shingle_n + 1)))
        rows.append(recall_row(f"whole doc, {shingle_n}-gram shingles, {bands}x{r} "
                               f"(J≈{lsh_threshold(bands, r):.2f})", idx.candidate_pairs()))
    for window, stride, bands, r in ((50, 25, 32, 4), (30, 15, 32, 4), (25, 8, 16, 8), (25, 8, 32, 4)):
        cc = ChunkedCandidates(window=window, stride=stride, bands=bands, rows=r)
        for d in corpus.all_docs:
            cc.add(d.name, d.content_words)
        rows.append(recall_row(f"{window}-word windows, stride {stride}, {bands}x{r}" +
                               (" **(default)**" if (window, stride, bands) == (25, 8, 32) else ""), cc.pairs()))
    report += ["## 6. MinHash LSH as a pre-filter\n",
               "Share of answer/source pairs that LSH proposes as candidates. Higher recall on "
               "cut/light/heavy is good. The last column is how many of all "
               f"{n_all} document pairs would still need full scoring. That number is high here because "
               "the corpus is 5 topics with ~20 answers each; most surviving pairs are same-topic pairs.\n",
               md_table(["index", "recall cut / light / heavy", "non proposed", "pairs kept"], rows), ""]

    (RESULTS / "results.md").write_text("\n".join(report), encoding="utf-8")
    with open(RESULTS / "pair_features.csv", "w", encoding="utf-8") as f:
        f.write("file,task,category," + ",".join(FEATURES) + ",v2_score\n")
        for a, row, lg in zip(answers, X, legacy):
            f.write(f"{a.doc.name},{a.task},{a.category}," + ",".join(f"{v:.4f}" for v in row) + f",{lg:.4f}\n")
    print("\n".join(report))

    if not args.no_plot:
        plot(X, cat, legacy, reuse4)

    if args.save_model:
        model_all.save(DEFAULT_MODEL)
        print(f"model written to {DEFAULT_MODEL}")


def fit_full(X, cat) -> ReuseModel:
    sc = StandardScaler().fit(X)
    clf = make_lr().fit(sc.transform(X), cat)
    return ReuseModel.from_sklearn(clf, sc, list(FEATURES), meta={
        "trained_on": "Clough & Stevenson (2011), 95 answer/source pairs",
        "classifier": "multinomial logistic regression, C=1.0, standardised features",
    })


def plot(X, cat, legacy, reuse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("v2.0 combined score", legacy),
              ("trigram containment", X[:, FEATURES.index("containment_3")]),
              ("winnowing overlap", X[:, FEATURES.index("winnow")]),
              ("model: P(reused), out-of-fold", reuse)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (title, s) in zip(axes, panels):
        for i, c in enumerate(CATEGORIES):
            v = s[cat == c]
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, alpha=0.7, color=f"C{i}")
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color="k", lw=1.5)
        ax.set_xticks(range(4), CATEGORIES)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.3)
    axes[0].axhline(0.75, ls="--", color="grey", lw=1)
    axes[0].text(3.4, 0.77, "old threshold", ha="right", fontsize=8, color="grey")
    axes[0].set_ylabel("score")
    fig.tight_layout()
    fig.savefig(RESULTS / "scores_by_category.png", dpi=140)
    print(f"plot written to {RESULTS / 'scores_by_category.png'}")


if __name__ == "__main__":
    main()

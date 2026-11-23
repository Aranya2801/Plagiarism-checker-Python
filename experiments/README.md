# Experiments

## Data

The corpus of plagiarised short answers (Clough & Stevenson, 2011) isn't included in this repo. Download it from the authors' page:

https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html

Unzip it anywhere. The folder you pass to `--data` should contain `file_information.csv` and the 100 `.txt` files (`g0pA_taska.txt` ... `orig_taske.txt`). Several copies on GitHub use the same layout, so any of those works too.

## Running

```
pip install -e ".[experiments]"
python experiments/benchmark.py --data path/to/corpus
python experiments/scaling.py
```

`benchmark.py` takes about 40 seconds and writes:

- `results/results.md`: all tables (single measures, CV accuracy, confusion matrix, ablation, same-topic false positives, LSH recall)
- `results/pair_features.csv`: the 8 features for every answer/source pair, plus the old v2 score
- `results/scores_by_category.png`

Add `--save-model` to retrain `plagiarism_checker/data/model.json` on all 95 pairs. The shipped file was produced exactly this way.

`scaling.py` builds synthetic collections of 500 to 10,000 documents with planted copies and measures how many candidate pairs the LSH index keeps, and how many planted pairs survive. It takes about 40 seconds.

## Notes

- Cross-validation is leave-one-task-out everywhere. With random folds, answers to the same question end up in train and test, and the numbers look a few points better than they should.
- AUC intervals are a percentile bootstrap (2,000 resamples). With 95 examples they're wide. Most single measures overlap with each other, so don't over-read their ordering.
- The ablation in section 4 is noisy for the same reason. Dropping `char_cosine` *improves* 4-way macro-F1 by about 0.01 on this data. I kept it because it helps on inflected or slightly edited text, which this corpus doesn't have much of.

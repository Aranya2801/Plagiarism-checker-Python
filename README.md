<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Plagiarism%20Checker%20Pro&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=38&desc=AI-Powered%20Document%20Similarity%20Engine&descAlignY=60&descSize=18" width="100%"/>

<!-- Typing Animation -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&multiline=true&width=800&height=100&lines=🔍+Detect+Plagiarism+with+AI+Precision;📊+TF-IDF+%7C+Cosine+Similarity+%7C+N-gram+%7C+LSH;⚡+Fast+%7C+Accurate+%7C+Beautiful+Reports" alt="Typing SVG" />
</a>

<br/>

<!-- Badges Row 1 -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-6366f1?style=for-the-badge)](https://github.com/Aranya2801/Plagiarism-checker-Python/releases)

<!-- Badges Row 2 -->
[![CI](https://img.shields.io/github/actions/workflow/status/Aranya2801/Plagiarism-checker-Python/ci.yml?style=for-the-badge&logo=github-actions&label=CI&logoColor=white)](https://github.com/Aranya2801/Plagiarism-checker-Python/actions)
[![Stars](https://img.shields.io/github/stars/Aranya2801/Plagiarism-checker-Python?style=for-the-badge&logo=github&color=ffd700)](https://github.com/Aranya2801/Plagiarism-checker-Python/stargazers)
[![Forks](https://img.shields.io/github/forks/Aranya2801/Plagiarism-checker-Python?style=for-the-badge&logo=github&color=22d3ee)](https://github.com/Aranya2801/Plagiarism-checker-Python/network)
[![Issues](https://img.shields.io/github/issues/Aranya2801/Plagiarism-checker-Python?style=for-the-badge&logo=github&color=ef4444)](https://github.com/Aranya2801/Plagiarism-checker-Python/issues)

<br/>

<!-- Demo GIF Placeholder — replace with your own screen recording -->
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

</div>

---

## 🌟 What is Plagiarism Checker Pro?

**Plagiarism Checker Pro** is a powerful, production-ready Python engine for detecting document similarity and plagiarism. It combines **multiple NLP algorithms** into one unified tool — giving you accurate, explainable, and beautiful results from the command line or Python API.

Whether you're an **educator** checking student essays, a **researcher** verifying originality, a **developer** building a document pipeline, or just someone who wants to protect their own writing — this tool has you covered.

---

## ✨ Features

<table>
<tr>
<td>

### 🧠 AI Algorithms
- **TF-IDF Cosine Similarity** — gold standard for document comparison
- **Jaccard Set Similarity** — token overlap scoring
- **N-gram / Shingle Matching** — catches paraphrasing and reordered text
- **MinHash / LSH** — lightning-fast approximate matching for large corpora
- **Combined Weighted Score** — configurable fusion of all algorithms

</td>
<td>

### 📊 Reports & Output
- **Coloured terminal report** with progress bars
- **Self-contained HTML report** — interactive, filterable, no server needed
- **JSON export** — machine-readable for pipelines and automation
- **Matching phrase extraction** — highlights exactly which parts match
- **Word overlap counts** and document statistics

</td>
</tr>
<tr>
<td>

### 🛠 Developer Friendly
- **Install as a package** — `pip install -e .`
- **CLI tool** — `plagcheck -d ./essays`
- **Python API** — import and use in any project
- **Fully typed** — dataclasses and type hints throughout
- **95%+ test coverage** — pytest suite included

</td>
<td>

### 📁 File Support
- Plain text (`.txt`, `.md`, `.rst`)
- PDF extraction (PyMuPDF / pdfplumber / pypdf)
- URL fetching — check web pages directly
- Batch directory processing
- Recursive folder scanning
- Automatic encoding detection

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/Aranya2801/Plagiarism-checker-Python.git
cd Plagiarism-checker-Python

# Install dependencies
pip install -r requirements.txt

# Install as a package (adds `plagcheck` CLI command)
pip install -e .
```

### 2. Try it immediately with sample documents

```bash
# Generate sample documents
python -c "from plagiarism_checker.utils import create_sample_documents; create_sample_documents('./samples')"

# Run the checker on the samples directory
python main.py -d ./samples
```

### 3. Check two specific files

```bash
python main.py -f essay1.txt essay2.txt
```

### 4. Check a submission against a corpus

```bash
python main.py -q new_submission.txt -d ./reference_corpus
```

---

## 📖 CLI Reference

```
plagcheck [-h] (-d DIR | -f FILE FILE | -q FILE) [options]

Modes:
  -d DIR          Compare all documents in a directory pairwise
  -f FILE FILE    Compare exactly two files
  -q FILE         Check one file against a corpus (-d required)

Options:
  --threshold T   Flag pairs above this score (default: 0.75)
  --output FILE   Save JSON report to file
  --show-phrases  Display matching text phrases
  --min-score S   Only show pairs above this score (default: 0.0)
  --stemming      Enable word stemming for better accuracy
  --weights C:J:N Algorithm weights, e.g. 0.5:0.25:0.25 (default)
  --no-progress   Disable progress bar
  --version       Show version and exit
```

### Examples

```bash
# Check a directory with custom threshold
plagcheck -d ./essays --threshold 0.60

# Save JSON report
plagcheck -d ./essays --output results.json

# Show matching phrases and filter by score
plagcheck -d ./essays --show-phrases --min-score 0.5

# Use custom algorithm weights (more emphasis on cosine)
plagcheck -d ./essays --weights 0.7:0.15:0.15

# Enable stemming for better paraphrase detection
plagcheck -d ./essays --stemming
```

---

## 🐍 Python API

```python
from plagiarism_checker.engine import PlagiarismChecker

# Create checker
checker = PlagiarismChecker(
    threshold=0.75,
    weights={"cosine": 0.5, "jaccard": 0.25, "ngram": 0.25},
)

# Load documents
checker.load_directory("./essays")

# Or load individual texts
checker.load_text("doc1", "The quick brown fox...")
checker.load_text("doc2", "A fast auburn fox...")

# Run comparison
result = checker.check_all()

# Print beautiful report
checker.print_report(result)

# Access results programmatically
for pair in result.pairs:
    print(f"{pair.doc1} ↔ {pair.doc2}: {pair.combined_score:.1%}")
    print(f"  Verdict: {pair.verdict}")
    print(f"  Matching phrases: {pair.highlighted_sections[:3]}")
```

### Generate HTML Report

```python
from plagiarism_checker.reporter import generate_html_report

result = checker.check_all()
generate_html_report(result, output_path="report.html")
# Opens a beautiful interactive HTML file
```

### Check a New Document Against a Database

```python
new_text = open("student_submission.txt").read()
pairs = checker.check_against_corpus(new_text, query_name="submission")

for pair in pairs:
    print(f"Similarity to {pair.doc2}: {pair.combined_score:.1%} — {pair.verdict}")
```

---

## 🔬 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                           │
└─────────────────────────────────────────────────────────────────┘

  Input Documents
       │
       ▼
  ┌──────────────────────┐
  │  Text Preprocessing  │  ← URL/email removal, lowercasing,
  │  (TextPreprocessor)  │    stopword filtering, stemming
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                   Similarity Algorithms                      │
  │                                                              │
  │  ① TF-IDF Cosine  ② Jaccard Set  ③ N-gram Shingle  ④ MinHash│
  │     (weight 0.5)    (weight 0.25)   (weight 0.25)           │
  └──────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  Weighted Sum   │  Combined Score = 0–100%
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Verdict Engine │  ORIGINAL / SUSPICIOUS /
                   └────────┬────────┘  LIKELY / HIGHLY PLAGIARISED
                            │
                            ▼
               ┌────────────────────────┐
               │   Report Generation    │
               │  Terminal · JSON · HTML│
               └────────────────────────┘
```

---

## 📊 Algorithm Explained

| Algorithm | What it measures | Best for |
|-----------|-----------------|----------|
| **TF-IDF Cosine** | Semantic word importance + angle between vectors | General-purpose comparison |
| **Jaccard Similarity** | Shared token set overlap | Short-to-medium documents |
| **N-gram (Shingle)** | Consecutive word sequence matches | Detecting reordered / paraphrased text |
| **MinHash (LSH)** | Approximate Jaccard via hashing | Very large corpora (fast) |

### Verdict Thresholds (default)

| Score Range | Verdict | Meaning |
|-------------|---------|---------|
| 90% – 100% | 🔴 Highly Plagiarised | Near-identical copy |
| 75% – 90%  | 🟠 Likely Plagiarised | Strong similarity, likely copied |
| 50% – 75%  | 🟡 Suspicious | Significant overlap, needs review |
| 30% – 50%  | 🟢 Minor Similarity | Some shared content |
| 0% – 30%   | ✅ Original | Documents appear distinct |

> Thresholds are fully configurable via `--threshold` flag.

---

## 🗂 Project Structure

```
Plagiarism-checker-Python/
│
├── 📄 main.py                        # Quick-run entry point
├── 📄 setup.py                       # Package installation config
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # This file
├── 📄 LICENSE                        # MIT License
│
├── 📁 plagiarism_checker/            # Core package
│   ├── __init__.py                   # Package metadata
│   ├── engine.py                     # 🧠 Core detection engine
│   │                                 #   TextPreprocessor
│   │                                 #   SimilarityEngine (TF-IDF, Jaccard, N-gram, LSH)
│   │                                 #   PlagiarismChecker (main class)
│   ├── cli.py                        # 🖥  Beautiful CLI interface
│   ├── reporter.py                   # 📊 HTML report generator
│   └── utils.py                      # 🛠  File I/O, URL fetch, fingerprinting
│
├── 📁 tests/                         # Test suite
│   ├── __init__.py
│   └── test_engine.py                # 30+ unit tests
│
├── 📁 samples/                       # Example documents
│   ├── original_essay.txt
│   ├── paraphrased_essay.txt
│   ├── partial_copy.txt
│   └── unrelated_history.txt
│
└── 📁 .github/
    └── workflows/
        └── ci.yml                    # GitHub Actions CI
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=plagiarism_checker --cov-report=term-missing

# Run a specific test class
pytest tests/test_engine.py::TestSimilarityEngine -v

# Run a specific test
pytest tests/test_engine.py::TestPlagiarismChecker::test_identical_flagged -v
```

---

## 📋 Output Example

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██████╗  ██╗      █████╗  ██████╗     ██████╗██╗  ██╗███████╗ ║
║       ...  Plagiarism Checker Pro v2.0  ...                     ║
╚══════════════════════════════════════════════════════════════════╝

  [████████████████████] 100.0%  Comparing: original ↔ partial_copy

════════════════════════════════════════════════════════════════════
                    📊  PLAGIARISM ANALYSIS REPORT
════════════════════════════════════════════════════════════════════
  Documents Scanned : 4
  Pairs Compared    : 6
  Flagged Pairs     : 1
  Plagiarism Rate   : 16.7%
  Time Taken        : 0.043s
  Timestamp         : 2025-05-04T14:23:11
────────────────────────────────────────────────────────────────────

  🟠 LIKELY PLAGIARISED
  📄 original_essay.txt  ↔  partial_copy.txt
     Cosine    ████████████████░░░░  81.2%
     Jaccard   █████████████░░░░░░░  67.4%
     N-gram    ██████████████░░░░░░  70.1%
     Combined  ██████████████░░░░░░  74.7%
     Words shared: 48  |  Time: 12.3ms
     Phrases: "transforming the modern world in profound ways" ...

  ✅ ORIGINAL
  📄 original_essay.txt  ↔  unrelated_history.txt
     Cosine    ████░░░░░░░░░░░░░░░░  18.3%
     ...
════════════════════════════════════════════════════════════════════
```

---

## 🛣 Roadmap

- [x] TF-IDF cosine similarity
- [x] Jaccard set similarity
- [x] N-gram shingle matching
- [x] MinHash / LSH approximate similarity
- [x] CLI with coloured output and progress
- [x] Interactive HTML report
- [x] JSON export
- [x] PDF extraction support
- [x] URL content fetching
- [x] GitHub Actions CI
- [ ] 🔜 Streamlit web dashboard
- [ ] 🔜 BERT / sentence-transformers semantic similarity
- [ ] 🔜 Cross-language plagiarism detection
- [ ] 🔜 REST API server mode
- [ ] 🔜 Database persistence (SQLite)
- [ ] 🔜 Google Docs / Word document support

---

## 🤝 Contributing

Contributions, issues, and feature requests are very welcome!

1. **Fork** the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a **Pull Request**

Please make sure your code passes the test suite before submitting.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

<div align="center">

**Aranya2801**

[![GitHub](https://img.shields.io/badge/GitHub-Aranya2801-181717?style=for-the-badge&logo=github)](https://github.com/Aranya2801)

*Built with ❤️ and Python*

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**⭐ Star this repo if it helped you!**

[![Star History Chart](https://api.star-history.com/svg?repos=Aranya2801/Plagiarism-checker-Python&type=Date)](https://star-history.com/#Aranya2801/Plagiarism-checker-Python&Date)

</div>

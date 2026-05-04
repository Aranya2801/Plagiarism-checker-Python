"""
Setup configuration for Plagiarism Checker Pro.
Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="plagiarism-checker-pro",
    version="2.0.0",
    author="Aranya2801",
    author_email="",
    description="Advanced AI-powered plagiarism detection engine using TF-IDF, Cosine Similarity, Jaccard & N-gram",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aranya2801/Plagiarism-checker-Python",
    project_urls={
        "Bug Reports": "https://github.com/Aranya2801/Plagiarism-checker-Python/issues",
        "Source":      "https://github.com/Aranya2801/Plagiarism-checker-Python",
    },
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "pdf":  ["PyMuPDF>=1.23.0"],
        "full": ["PyMuPDF>=1.23.0", "streamlit>=1.28.0"],
    },
    entry_points={
        "console_scripts": [
            "plagcheck=plagiarism_checker.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="plagiarism detection NLP TF-IDF cosine similarity text mining",
)

# Plagiarism-checker-Python

This is a Python program that checks for plagiarism by comparing two documents and detecting the percentage of similarity between them.

Dependencies
Python 3.x
NumPy
NLTK
PyPDF2

You can install NumPy and NLTK by running the following command:
pip install numpy nltk

To install PyPDF2, run the following command:

pip install pypdf2
Usage

The plagiarism_checker.py script takes two file paths as arguments and outputs the percentage of similarity between them.

php

python plagiarism_checker.py <file1_path> <file2_path>
For example:


python plagiarism_checker.py original.txt plagiarized.txt
This will output the percentage of similarity between the two files.

How it works
The program uses the NLTK library to tokenize the text of both documents into individual words, and then it compares the word frequencies in the two documents using cosine similarity. Cosine similarity measures the angle between two vectors, where each vector represents the word frequency in one of the documents.

Limitations
The program has some limitations:

It only compares text files and PDF files.
It does not detect the context of words in sentences.
It does not take into account synonyms or other forms of similar words.
License
This project is licensed under the MIT License.

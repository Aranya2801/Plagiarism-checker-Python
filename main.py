#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           PLAGIARISM CHECKER PRO — Quick Run Script          ║
║                                                              ║
║  Usage:                                                      ║
║    python main.py -d ./samples                               ║
║    python main.py -f file1.txt file2.txt                     ║
║    python main.py -q new_paper.txt -d ./corpus               ║
║    python main.py -d ./essays --output report.json           ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
from plagiarism_checker.cli import main

if __name__ == "__main__":
    sys.exit(main())

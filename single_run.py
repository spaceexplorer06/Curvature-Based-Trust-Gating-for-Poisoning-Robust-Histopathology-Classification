#!/usr/bin/env python3
"""
Test a single experiment run with full output.
Usage: python single_run.py --run_name test_clean --poison_rate 0.0 --epochs 2

This is a thin wrapper around training_defense.main() (no subprocess, so
tracebacks and tqdm bars show up directly). It does NOT parse its own args -
training_defense.main() parses sys.argv itself, so a separate argparse here
was dead code (whatever it produced was discarded). Run --help to see the
full flag list, which lives in training_defense.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_defense import main as train_main

if __name__ == "__main__":
    train_main()
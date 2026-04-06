"""Shared expanding train/test windows for journal-style OOS evaluation.

Each row:
  (train_start, train_end, test_start, test_end, window_tag)

Used by both training (one checkpoint per ``train_end``) and evaluation.
"""

from typing import List, Tuple

# (train_start, train_end, test_start, test_end, window_name)
EXPANDING_WINDOWS: List[Tuple[str, str, str, str, str]] = [
    ("1985-01-01", "2000-12-31", "2001-01-01", "2005-12-31", "w1_01-05"),
    ("1985-01-01", "2005-12-31", "2006-01-01", "2011-12-31", "w2_06-11"),
    ("1985-01-01", "2011-12-31", "2012-01-01", "2017-12-31", "w3_12-17"),
]

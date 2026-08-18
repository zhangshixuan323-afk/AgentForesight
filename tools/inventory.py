# -*- coding: utf-8 -*-
"""Data inventory for the sample100_by_benchmark corpus (read-only).

Prints the test-set composition (default: 17 train/validation files excluded)
and the exclusion list, plus basic sanity stats used by the report.

    python -m tools.inventory [data_dir]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from inference.data import load_sample100

def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample100_by_benchmark")
    records, excluded = load_sample100(data_dir)

    print(f"data_dir : {data_dir.resolve()}")
    print(f"test-set : {len(records)} samples (train/validation excluded: {len(excluded)})")
    print()
    print("per benchmark:")
    for dom, n in Counter(r.domain for r in records).most_common():
        print(f"  {dom:18s} {n}")
    print()
    print("excluded (train/validation, NOT in test set):")
    for p in excluded:
        print(f"  {p.parent.name:18s} {p.stem}")
    print()
    print("sanity:")
    print(f"  steps total      : {sum(r.num_turns for r in records)}")
    print(f"  max steps/sample : {max(r.num_turns for r in records)}")
    print(f"  gt_step>=0       : {sum(1 for r in records if r.mistake_step >= 0)}")
    print(f"  gt_step<0        : {sum(1 for r in records if r.mistake_step < 0)}")
    print(f"  dup conv_ids     : {len(records) - len({r.conv_id for r in records})}")
    print(f"  calls to gt (perfect detector): {sum(r.mistake_step + 1 for r in records)}")
    print(f"  calls worst (never alarm)     : {sum(r.num_turns for r in records)}")
    print()
    print("gt agent distribution:")
    for a, n in Counter(r.mistake_agent for r in records).most_common():
        print(f"  {n:3d}  {a}")
    print()
    print("history name @ gt step distribution:")
    for a, n in Counter(
        (r.turns[r.mistake_step].get("name") if 0 <= r.mistake_step < r.num_turns else "?")
        for r in records
    ).most_common():
        print(f"  {n:3d}  {a}")

if __name__ == "__main__":
    main()

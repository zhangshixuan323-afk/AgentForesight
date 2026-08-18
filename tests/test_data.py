# -*- coding: utf-8 -*-
"""Unit tests for the sample100 data loader (and later the prompt/metrics).

Runs standalone:  python -m tests.test_data
Also pytest-compatible (functions named test_*).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.data import detect_split, load_sample100
from inference.prompts import build_messages

DATA_DIR = Path(__file__).resolve().parents[1] / "sample100_by_benchmark"


def test_loads_83_records() -> None:
    records, excluded = load_sample100(DATA_DIR)
    assert len(records) == 83, f"expected 83 test samples, got {len(records)}"
    assert len(excluded) == 17, f"expected 17 excluded, got {len(excluded)}"
    # every test sample is unsafe with an in-range annotation
    for r in records:
        assert r.label == "unsafe"
        assert 0 <= r.mistake_step < r.num_turns, (r.conv_id, r.mistake_step, r.num_turns)
        assert r.mistake_agent


def test_steps_contiguous() -> None:
    records, _ = load_sample100(DATA_DIR)
    for r in records:
        assert len(r.turns) == r.num_turns
        for i, t in enumerate(r.turns):
            assert t["step"] == i, (r.conv_id, i, t.get("step"))
            assert "name" in t and "role" in t and "content" in t


def test_no_duplicate_conv_ids() -> None:
    records, _ = load_sample100(DATA_DIR)
    ids = [r.conv_id for r in records]
    assert len(ids) == len(set(ids))


def test_split_detection() -> None:
    assert detect_split("travelplanner__kimi_k2_5__magentic_one_groupchat_test__0026__run_0001") == "test"
    assert detect_split("travelplanner__minimax_m2_5__magentic_one_groupchat_validation__0041__run_0001") == "validation"
    assert detect_split("travelplanner__qwen3_5_plus__magentic_one_groupchat_train__0008__run_0001") == "train"
    assert detect_split("swe_bench_pro__instance_ansible__ansible-42355d__iWDKhRg") == ""


def test_excluded_are_train_validation() -> None:
    _, excluded = load_sample100(DATA_DIR)
    for p in excluded:
        assert detect_split(p.stem) in ("train", "validation"), p


def test_no_gt_leakage_in_prompt() -> None:
    """The audit prompt must never contain the ground-truth annotation."""
    records, _ = load_sample100(DATA_DIR)
    for r in records:
        msgs = build_messages(r.turns, tools=r.tools, current_step=r.mistake_step)
        text = "\n".join(m.get("content", "") for m in msgs)
        for forbidden in ('"mistake_agent"', '"mistake_step"', '"mistake_reason"'):
            assert forbidden not in text, (r.conv_id, forbidden)
        assert r.conv_id not in text, (r.conv_id, "question_ID leaked")


def test_known_annotation_spot_check() -> None:
    records, _ = load_sample100(DATA_DIR)
    by_id = {r.conv_id: r for r in records}
    r = by_id["travelplanner__kimi_k2_5__magentic_one_groupchat_test__0026__run_0001"]
    assert r.mistake_step == 86
    assert r.mistake_agent == "Action_Expert"
    assert r.domain == "travelplanner"


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

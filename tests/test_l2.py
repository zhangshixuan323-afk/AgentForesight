# -*- coding: utf-8 -*-
"""Unit tests for P5 local pieces: L2 rolling-summary prompt support
(prior_summary / allow_collapse / over_budget) and the summary chunk plan.

Runs standalone:  python -m tests.test_l2
Also pytest-compatible.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.prompts import build_messages, build_summary_messages


def _turns(n: int, per: int = 1200) -> list[dict]:
    return [
        {"step": i, "role": "assistant", "name": "agent_x",
         "content": f"Tool call: t{i}\nArguments: {{\"k\": {i}}} " + "y" * per}
        for i in range(n)
    ]


def test_allow_collapse_false_reports_over_budget() -> None:
    turns = _turns(60)
    turns.insert(0, {"step": 0, "role": "user", "name": "Computer_terminal", "content": "task"})
    for i, t in enumerate(turns):
        t["step"] = i
    # Tiny budget: even the L1 floor (2x800 verbatim + 120-char summaries)
    # cannot fit, so allow_collapse=False must report over_budget.
    msgs, stats = build_messages(turns, current_step=len(turns) - 1,
                                 return_stats=True, allow_collapse=False,
                                 max_input_tokens=1500)
    assert stats["over_budget"] is True
    assert stats["collapsed_middle"] is False
    # default path (allow_collapse=True) still fits the budget via collapse
    _, stats2 = build_messages(turns, current_step=len(turns) - 1,
                               return_stats=True, allow_collapse=True,
                               max_input_tokens=1500)
    assert stats2["over_budget"] is False
    assert stats2["collapsed_middle"] is True


def test_prior_summary_render() -> None:
    turns = _turns(12)
    turns.insert(0, {"step": 0, "role": "user", "name": "Computer_terminal", "content": "task"})
    for i, t in enumerate(turns):
        t["step"] = i
    msgs, stats = build_messages(turns, current_step=len(turns) - 1,
                                 return_stats=True, allow_collapse=False,
                                 prior_summary="SUMMARY TEXT", tail_start=8,
                                 max_input_tokens=30720)
    text = "\n".join(m.get("content", "") for m in msgs)
    assert stats["prior_summary"] is True
    assert "SUMMARY TEXT" in text
    assert "[Prior summary of steps 0..7" in text
    assert "Step 8 -" in text          # tail starts at 8
    assert "Step 0 -" not in text      # span 0..7 replaced by the summary


def test_summary_messages_bounded_input() -> None:
    turns = _turns(40, per=2000)
    msgs = build_summary_messages(turns, 0, 15, prior_summary=None,
                                  budget_chars=2000, per_turn_chars=12000)
    user = msgs[1]["content"]
    assert "New turns (steps 0..15)" in user
    assert "Write the updated summary covering steps 0..15." in user
    # span rendered verbatim but capped per turn; no later turns leaked in
    assert "step 16" not in user or "New turns (steps" not in user.split("step 16")[1]


def _summary_plan(covered, m, interval):
    """Reference implementation of the rolling-chunk plan (mirrors audit.py)."""
    plan = []
    c = covered
    while c is None or c < m:
        start = 0 if c is None else c + 1
        target = min(m, start + interval - 1)
        plan.append((start, target))
        c = target
    return plan


def test_rolling_chunk_plan() -> None:
    assert _summary_plan(None, 100, 16) == [(0, 15), (16, 31), (32, 47), (48, 63),
                                            (64, 79), (80, 95), (96, 100)]
    assert _summary_plan(50, 100, 16) == [(51, 66), (67, 82), (83, 98), (99, 100)]
    assert _summary_plan(100, 100, 16) == []
    assert _summary_plan(None, 5, 16) == [(0, 5)]


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

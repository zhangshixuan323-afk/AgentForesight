# -*- coding: utf-8 -*-
"""Unit tests for P1: prompt identity display, team legend, compression V1,
parse_response tolerance.

Runs standalone:  python -m tests.test_prompts
Also pytest-compatible.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.data import load_sample100
from inference.prompts import build_messages, parse_response

DATA_DIR = Path(__file__).resolve().parents[1] / "sample100_by_benchmark"


def _text_of(messages) -> str:
    return "\n".join(m.get("content", "") for m in messages)


def test_render_shows_agent_name_and_real_step() -> None:
    records, _ = load_sample100(DATA_DIR)
    r = next(x for x in records if x.conv_id.endswith("magentic_one_groupchat_test__0026__run_0001"))
    assert r.turns[r.mistake_step]["name"] == "verifier_agent"
    msgs = build_messages(r.turns, tools=r.tools, current_step=r.mistake_step)
    text = _text_of(msgs)
    assert f"Step {r.mistake_step} - verifier_agent (assistant)" in text
    assert "Step 86 - verifier_agent" in text


def test_agent_legend_present() -> None:
    records, _ = load_sample100(DATA_DIR)
    r = next(x for x in records if x.conv_id.endswith("magentic_one_groupchat_test__0026__run_0001"))
    msgs = build_messages(r.turns, tools=r.tools, current_step=r.mistake_step)
    text = _text_of(msgs)
    assert "Agent identities in this trajectory:" in text
    assert "Task_Planner" in text and "verifier_agent" in text


def test_verbatim_within_budget() -> None:
    # Synthetic short trajectory: full content must appear verbatim, no truncation.
    long_content = "The quick brown fox jumps over the lazy dog. " * 40
    turns = [
        {"step": 0, "role": "user", "name": "Computer_terminal", "content": "task: " + long_content},
        {"step": 1, "role": "assistant", "name": "Task_Planner", "content": "planning: " + long_content},
    ]
    msgs, stats = build_messages(turns, current_step=1, return_stats=True)
    text = _text_of(msgs)
    assert stats["verbatim"] is True
    assert long_content in text                      # full content kept
    assert "[truncated]" not in text
    assert "[summarized" not in text


def test_compression_kicks_in_over_budget() -> None:
    # Synthetic 60-turn trajectory; default 30720-token budget cannot hold it
    # verbatim, so windowing + L1 summaries must engage and stay under budget.
    turns = [
        {"step": i, "role": "assistant", "name": "agent_x",
         "content": "Tool call: flight_search\nArguments: {\"origin\": \"Baltimore\", \"destination\": \"Milwaukee\"} " + "x" * 2500}
        for i in range(60)
    ]
    turns.insert(0, {"step": 0, "role": "user", "name": "Computer_terminal", "content": "task"})
    # renumber steps
    for i, t in enumerate(turns):
        t["step"] = i
    msgs, stats = build_messages(turns, current_step=len(turns) - 1, return_stats=True,
                                 max_input_tokens=30720)
    text = _text_of(msgs)
    assert stats["verbatim"] is False
    assert stats["compressed_turns"] > 0
    assert stats["input_tokens_estimate"] <= stats["max_input_tokens"]
    assert "[summarized" in text                      # L1 summaries actually used
    assert "Step 0 - Computer_terminal (user)" in text  # first turn kept verbatim


def test_parse_response_safe() -> None:
    v = parse_response('<think>ok</think><answer>{"answer": "SAFE", "agent": null, "reason": null}</answer>')
    assert v.valid and v.pred_step == -1


def test_parse_response_valid_step() -> None:
    v = parse_response('<answer>{"answer": 5, "agent": "Action_Expert", "reason": "r"}</answer>', max_step=10)
    assert v.valid and v.pred_step == 5 and v.pred_agent == "Action_Expert"


def test_parse_response_out_of_range_invalid() -> None:
    v = parse_response('<answer>{"answer": 7, "agent": "A", "reason": "r"}</answer>', max_step=3)
    assert not v.valid and v.pred_step == -1


def test_parse_response_negative_invalid() -> None:
    v = parse_response('<answer>{"answer": -1, "agent": "A", "reason": "r"}</answer>')
    assert not v.valid and v.pred_step == -1


def test_parse_response_garbage_invalid() -> None:
    assert not parse_response("no blocks here").valid
    assert not parse_response("<answer>{not json}</answer>").valid
    assert not parse_response("").valid


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

# -*- coding: utf-8 -*-
"""Unit tests for P3: audit engine (oracle/mock backends, detection semantics,
jsonl writing, resume).

Runs standalone:  python -m tests.test_audit
Also pytest-compatible.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.audit import MockAuditor, OracleAuditor, audit_one, run_audit
from inference.data import TrajectoryRecord


def _rec(n: int = 3, gt_step: int = 2, gt_agent: str = "Action_Expert",
         conv_id: str = "t") -> TrajectoryRecord:
    turns = [
        {"step": i, "role": "assistant" if i else "user",
         "name": "agent_x" if i else "Computer_terminal", "content": f"step {i}"}
        for i in range(n)
    ]
    return TrajectoryRecord(
        conv_id=conv_id, domain="d", label="unsafe", task="", gold_answer="",
        num_turns=n, turns=turns, tools=[], mistake_step=gt_step,
        mistake_agent=gt_agent, mistake_reason="r", split="test", src_path="x.json",
    )


def test_oracle_alarms_exactly_at_gt() -> None:
    row = audit_one(_rec(n=5, gt_step=2), OracleAuditor(), 2048, 0.0)
    assert row["pred_step"] == 2 and row["detection_step"] == 2
    assert row["num_calls"] == 3            # prefixes 0,1,2 then alarm
    assert row["step_correct"] is True
    assert row["agent_ok_strict"] is True


def test_mock_safe_never_alarms() -> None:
    row = audit_one(_rec(n=5, gt_step=2), MockAuditor("safe"), 2048, 0.0)
    assert row["pred_step"] == -1 and row["detection_step"] == -1
    assert row["num_calls"] == 5
    assert row["step_correct"] is False


def test_mock_last_alarms_at_end() -> None:
    row = audit_one(_rec(n=5, gt_step=2), MockAuditor("last"), 2048, 0.0)
    assert row["pred_step"] == 4 and row["detection_step"] == 4
    assert row["num_calls"] == 5


def test_run_audit_writes_jsonl_and_resumes() -> None:
    import shutil
    recs = [_rec(n=3, gt_step=1, conv_id=f"c{i}") for i in range(4)]
    tmp = Path(__file__).resolve().parents[1] / "outputs" / "_audit_test_tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        auditor = MockAuditor("safe")
        run_audit(recs, auditor, tmp, backend="mock", resume=False, snapshot_every=0)
        jsonl = tmp / "per_sample.jsonl"
        lines = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l]
        assert lines[0]["_record_type"] == "config"
        assert {r["conv_id"] for r in lines[1:]} == {"c0", "c1", "c2", "c3"}
        # second run resumes: nothing left to do
        run_audit(recs, auditor, tmp, backend="mock", resume=True, snapshot_every=0)
        lines2 = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l]
        assert len([l for l in lines2 if l.get("_record_type") != "config"]) == 4  # no duplicates
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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

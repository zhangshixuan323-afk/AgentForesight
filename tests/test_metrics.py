# -*- coding: utf-8 -*-
"""Unit tests for P2: agent-alias normalization (L1-L3) and B.2 metrics.

Runs standalone:  python -m tests.test_metrics
Also pytest-compatible.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import agent_aliases as aa
from inference.metrics import (
    agent_row_fields,
    aggregate,
    aggregate_by_domain,
    aggregate_cost,
    per_sample_metrics,
)


def _row(pred_step, gt_step, pred_agent="", gt_agent="", domain="d", **extra):
    row = {
        "conv_id": "x", "domain": domain, "label": "unsafe",
        "pred_step": pred_step, "gt_step": gt_step,
    }
    row.update(agent_row_fields(pred_agent, gt_agent))
    row.update(per_sample_metrics(pred_step, gt_step))
    row.update(extra)
    return row


# --- alias pipeline (L1/L2/L3) ----------------------------------------------

def test_l1_deterministic_families() -> None:
    assert aa.strict_match("transport_agent", "Action_Expert")
    assert aa.strict_match("stay_agent", "Action_Expert")
    assert aa.strict_match("food_attr_agent", "Action_Expert")
    assert aa.strict_match("Action_Expert", "Action_Expert")
    assert aa.strict_match("Task_Planner", "Task_Planner")
    assert aa.strict_match("ActionAgent", "ActionAgent")
    assert aa.strict_match("Verification_Expert", "Verification_Expert")
    assert aa.strict_match("JudgeAgent", "JudgeAgent")
    assert aa.strict_match("DiagnostAgent", "DiagnostAgent")


def test_l2_semantic_disambiguation() -> None:
    assert aa.strict_match("manager_agent", "Task_Planner")
    assert not aa.strict_match("manager_agent", "Action_Expert")   # L2 wins
    assert aa.strict_match("verifier_agent", "Verification_Expert")
    assert not aa.strict_match("verifier_agent", "Action_Expert")  # L2 wins


def test_compound_label_matching() -> None:
    gt = "DiagnostAgent (-> ActionAgent)"
    assert aa.strict_match(gt, gt)
    assert not aa.strict_match("DiagnostAgent", gt)          # strict: exact only
    assert aa.loose_match("DiagnostAgent", gt)               # loose: source end
    assert aa.loose_match("ActionAgent", gt)                 # loose: target end


def test_loose_family_excludes_ambiguous_names() -> None:
    # manager_agent / verifier_agent are noisy -> NOT family members, so the
    # loose metric does not degenerate into always-correct on those rows.
    assert aa.loose_match("transport_agent", "Action_Expert")
    assert not aa.loose_match("manager_agent", "Action_Expert")
    assert not aa.loose_match("verifier_agent", "Action_Expert")


def test_special_identities() -> None:
    assert aa.is_special("Computer_terminal")
    assert aa.is_special("human")
    assert not aa.is_special("Action_Expert")
    assert not aa.is_special("")


def test_agent_row_fields() -> None:
    f = agent_row_fields("transport_agent", "Action_Expert")
    assert f["agent_na"] is False and f["agent_ok_strict"] is True and f["agent_ok_loose"] is True
    f = agent_row_fields("manager_agent", "Action_Expert")
    assert f["agent_ok_strict"] is False and f["agent_ok_loose"] is False
    f = agent_row_fields("whatever", "Computer_terminal")
    assert f["agent_na"] is True and f["agent_ok_strict"] is False


# --- B.2 aggregation (hand-computed) -----------------------------------------

def test_aggregate_b2_hand_computed() -> None:
    rows = [
        _row(5, 5),      # Udet, correct
        _row(3, 5),      # Udet, wrong (shift 2)
        _row(-1, 5),     # miss (not in Udet)
        _row(0, 0),      # Udet, correct
        _row(7, 2),      # Udet, wrong (shift 5)
    ]
    m = aggregate(rows)
    # U = 5, Udet = 4, tp = 2
    assert m["n"] == 5 and m["n_unsafe"] == 5 and m["n_detected"] == 4
    assert abs(m["recall"] - 2 / 5 * 100) < 1e-9
    assert abs(m["precision"] - 2 / 4 * 100) < 1e-9
    assert abs(m["exact_f1"] - 2 * 0.4 * 0.5 / 0.9 * 100) < 1e-9
    assert abs(m["ass_mean"] - 1.75) < 1e-9         # (0 + 2 + 0 + 5) / 4 over Udet
    assert abs(m["step_acc"] - 40.0) < 1e-9          # == Recall_step
    assert m["n_agent_eligible"] == 5


def test_agent_acc_strict_loose_cond() -> None:
    rows = [
        _row(1, 1, pred_agent="transport_agent", gt_agent="Action_Expert"),   # step ok, agent ok
        _row(2, 2, pred_agent="Task_Planner", gt_agent="Action_Expert"),      # step ok, agent wrong
        _row(3, 1, pred_agent="stay_agent", gt_agent="Action_Expert"),        # step wrong, agent ok
        _row(0, 0, pred_agent="x", gt_agent="Computer_terminal"),             # L3: excluded
    ]
    m = aggregate(rows)
    # eligible = 3; strict ok = 2 -> 66.67; loose ok = 2 -> 66.67
    assert m["n_agent_eligible"] == 3
    assert abs(m["agent_acc_strict"] - 2 / 3 * 100) < 1e-9
    assert abs(m["agent_acc_loose"] - 2 / 3 * 100) < 1e-9
    # conditional: step-correct & eligible = 2 (rows 1,2); strict ok = 1 -> 50
    assert abs(m["agent_acc_cond"] - 50.0) < 1e-9


def test_aggregate_by_domain() -> None:
    rows = [_row(1, 1, domain="travelplanner"), _row(2, 3, domain="swe_bench_pro")]
    by = aggregate_by_domain(rows)
    assert set(by) == {"travelplanner", "swe_bench_pro", "overall"}
    assert by["travelplanner"]["step_acc"] == 100.0
    assert by["swe_bench_pro"]["step_acc"] == 0.0


def test_aggregate_cost() -> None:
    rows = [
        _row(0, 0, num_calls=2, gen_time_s=1.0, wall_time_s=2.0,
             prompt_tokens=100, completion_tokens=20, total_tokens=120),
        _row(1, 1, num_calls=4, gen_time_s=3.0, wall_time_s=5.0,
             prompt_tokens=300, completion_tokens=40, total_tokens=340),
        _row(2, 2),   # no cost data -> excluded from cost aggregation
    ]
    c = aggregate_cost(rows)
    assert c["n"] == 2
    assert c["num_calls_mean"] == 3.0 and c["num_calls_total"] == 6
    assert c["gen_time_s_total"] == 4.0
    assert c["wall_time_s_mean"] == 3.5
    assert c["prompt_tokens_total"] == 400
    assert c["completion_tokens_total"] == 60
    assert c["total_tokens_total"] == 460
    assert c["total_tokens_median"] == 230.0


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

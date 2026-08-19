# -*- coding: utf-8 -*-
"""Workload profiling for the online-audit cost estimate (model-independent).

For each sample of the test set it computes, per prefix call k, the input
tokens of prefix 0..k, under three detection scenarios:

- perfect : the auditor alarms exactly at the annotated decisive step
            (d_s = gt_step)   -> calls = gt_step + 1
- worst   : never alarms      -> calls = n_s
- measured: detection_step read from a prior run's per_sample.jsonl
            (--measured), -1 (never alarmed) -> calls = n_s

Two input-token columns are reported per scenario:
- naive   : sum of full prefix tokens (no caching, O(n^2) prefill)
- comp_ub : sum of min(prefix_tokens, max_input_tokens) -- the windowed
            compression caps every call at the input budget B, so this is an
            upper bound of the real compressed input (L1 summaries reduce it
            further; the exact number comes from a real run).

Token counting uses the Qwen tokenizer when --tokenizer is given (exact), else
a chars-per-token estimate (default 3.0). Output tokens per call default to 200
(the auditor's compact <think>+<answer>); override with --out-tokens-per-call.

    python -m tools.profile --data-dir sample100_by_benchmark --output outputs/profile.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from inference.data import load_sample100

DEFAULT_BUDGET = 30720
DEFAULT_OUT_TOKENS = 200


def _read_measured(jsonl: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not jsonl.exists():
        return out
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("_record_type") == "config" or not obj.get("conv_id"):
            continue
        out[obj["conv_id"]] = int(obj.get("detection_step", -1))
    return out


def _make_counter(tokenizer, cpt: float):
    if tokenizer is not None:
        def count(text: str) -> float:
            return float(len(tokenizer(text, add_special_tokens=False).input_ids))
        return count
    def count(text: str) -> float:
        return max(1.0, len(text) / cpt)
    return count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="sample100_by_benchmark")
    p.add_argument("--output", default="outputs/profile.json")
    p.add_argument("--tokenizer", default=None,
                   help="HF id or local path of the Qwen tokenizer (exact counts); "
                        "absent -> chars-per-token estimate.")
    p.add_argument("--chars-per-token", type=float, default=3.0)
    p.add_argument("--out-tokens-per-call", type=int, default=DEFAULT_OUT_TOKENS)
    p.add_argument("--max-input-tokens", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--measured", default=None,
                   help="Optional per_sample.jsonl from a prior run, for the "
                        "'measured' detection scenario.")
    p.add_argument("--exclude-splits", default="train,validation")
    args = p.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    count = _make_counter(tokenizer, args.chars_per_token)

    records, excluded = load_sample100(
        args.data_dir,
        exclude_splits=tuple(s.strip() for s in args.exclude_splits.split(",")),
    )
    measured = _read_measured(Path(args.measured)) if args.measured else {}

    per_sample: list[dict] = []
    for r in records:
        n = r.num_turns
        per_turn = [count(str(t.get("content") or "")) for t in r.turns]
        prefix = []
        acc = 0.0
        for v in per_turn:
            acc += v
            prefix.append(acc)
        d_perfect = r.mistake_step
        d_measured = measured.get(r.conv_id, -1)
        scenarios = {
            "perfect": d_perfect,
            "worst": n - 1,
        }
        if args.measured:
            scenarios["measured"] = d_measured if d_measured >= 0 else n - 1
        rows = {}
        for name, d in scenarios.items():
            calls = d + 1
            naive = sum(prefix[:calls])
            comp_ub = sum(min(prefix[:calls][k], args.max_input_tokens) for k in range(calls))
            rows[name] = {
                "calls": calls,
                "input_tokens_naive": round(naive),
                "input_tokens_compressed_ub": round(comp_ub),
                "output_tokens": calls * args.out_tokens_per_call,
                "total_tokens_naive": round(naive) + calls * args.out_tokens_per_call,
                "total_tokens_compressed_ub": round(comp_ub) + calls * args.out_tokens_per_call,
            }
        per_sample.append({
            "conv_id": r.conv_id, "domain": r.domain,
            "n_steps": n, "gt_step": r.mistake_step,
            "detection_measured": d_measured,
            "prefix_tokens": [int(round(v)) for v in prefix],
            "scenarios": rows,
        })

    def agg(rows, key) -> dict:
        vals = [r[key] for r in rows]
        vs = sorted(vals)
        p90 = vs[min(len(vs) - 1, int(round(0.9 * (len(vs) - 1))))]
        return {"mean": statistics.mean(vals), "median": statistics.median(vals),
                "p90": p90, "total": sum(vals)}

    scenarios = ["perfect", "worst"] + (["measured"] if args.measured else [])
    totals: dict = {}
    for sc in scenarios:
        rows = [r["scenarios"][sc] for r in per_sample]
        totals[sc] = {k: agg(rows, k) for k in ("calls", "input_tokens_naive",
                                                 "input_tokens_compressed_ub",
                                                 "output_tokens",
                                                 "total_tokens_naive",
                                                 "total_tokens_compressed_ub")}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {
            "tokenizer": args.tokenizer,
            "chars_per_token": args.chars_per_token,
            "out_tokens_per_call": args.out_tokens_per_call,
            "max_input_tokens": args.max_input_tokens,
            "n_test_samples": len(records),
            "n_excluded": len(excluded),
        },
        "totals": totals,
        "per_sample": per_sample,
    }, indent=2), encoding="utf-8")

    print(f"test set: {len(records)} samples (excluded {len(excluded)}); "
          f"out_tokens_per_call={args.out_tokens_per_call} budget={args.max_input_tokens}")
    hdr = (f"{'scenario':10s} {'calls':>8s} {'in_naive':>12s} {'in_compUB':>12s} "
           f"{'out':>9s} {'tot_naive':>12s} {'tot_compUB':>12s}")
    print(hdr); print("-" * len(hdr))
    for sc in scenarios:
        t = totals[sc]
        print(f"{sc:10s} {t['calls']['total']:8.0f} "
              f"{t['input_tokens_naive']['total']:12,.0f} {t['input_tokens_compressed_ub']['total']:12,.0f} "
              f"{t['output_tokens']['total']:9,.0f} "
              f"{t['total_tokens_naive']['total']:12,.0f} {t['total_tokens_compressed_ub']['total']:12,.0f}")
    print(f"\nper-sample naive input tokens (perfect scenario): "
          f"mean={totals['perfect']['input_tokens_naive']['mean']:,.0f} "
          f"median={totals['perfect']['input_tokens_naive']['median']:,.0f} "
          f"p90={totals['perfect']['input_tokens_naive']['p90']:,.0f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Combine engine calibration with the workload profile into cost estimates.

    python3 -m tools.extrapolate --profile outputs/profile.json \
        --output outputs/extrapolate.json \
        [--calib outputs/calib.json]        # real calibration (preferred)
        [--measured outputs/qwen/per_sample.jsonl]   # residual check

Without --calib, roofline defaults are used (7B bf16, A800): decode ~100 tok/s,
prefill ~11,100 tok/s -- replace with real numbers once calibrate.py ran.

--paper mode: quick average-based estimate for another dataset (e.g. the
paper's AFTraj-2K), using the classic avg-per-call x avg-steps formula:
    python3 -m tools.extrapolate --paper --paper-n 2276 --paper-avg-steps 60 \
        --paper-avg-alarm 45 --paper-avg-chars-per-step 400 --calib outputs/calib.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PREFILL_TPS = 11_143.0   # roofline default (312 TFLOPS, 7B, eff 0.5)
DEFAULT_DECODE_TPS = 100.0       # roofline default (2 TB/s, 7B bf16, eff 0.7)


def _load_calib(path: Path | None) -> dict:
    if path and path.exists():
        c = json.loads(path.read_text(encoding="utf-8"))
        return {
            "prefill_tps": float(c.get("prefill_tps_median", DEFAULT_PREFILL_TPS)),
            "decode_tps": float(c.get("decode_tps_median", DEFAULT_DECODE_TPS)),
            "source": str(path),
        }
    return {"prefill_tps": DEFAULT_PREFILL_TPS, "decode_tps": DEFAULT_DECODE_TPS,
            "source": "roofline-default"}


def estimate(input_tokens: float, output_tokens: float, calib: dict) -> dict:
    t_pre = input_tokens / calib["prefill_tps"]
    t_dec = output_tokens / calib["decode_tps"]
    return {"prefill_s": t_pre, "decode_s": t_dec, "total_s": t_pre + t_dec,
            "input_tokens": input_tokens, "output_tokens": output_tokens}


def _pct(pred: float, meas: float) -> float | None:
    return (pred - meas) / meas * 100 if meas else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="outputs/profile.json")
    p.add_argument("--calib", default=None)
    p.add_argument("--measured", default=None,
                   help="per_sample.jsonl of a real run (residual check).")
    p.add_argument("--output", default="outputs/extrapolate.json")
    p.add_argument("--scenario", default="perfect", choices=["perfect", "worst", "measured"])
    # paper (average-based) mode
    p.add_argument("--paper", action="store_true")
    p.add_argument("--paper-n", type=int, default=2276)
    p.add_argument("--paper-avg-steps", type=float, default=60.0)
    p.add_argument("--paper-avg-alarm", type=float, default=45.0)
    p.add_argument("--paper-avg-chars-per-step", type=float, default=400.0)
    p.add_argument("--chars-per-token", type=float, default=3.0)
    p.add_argument("--out-tokens-per-call", type=int, default=200)
    args = p.parse_args()

    calib = _load_calib(Path(args.calib) if args.calib else None)
    print(f"engine: prefill_tps={calib['prefill_tps']:,.1f}  decode_tps={calib['decode_tps']:,.1f}  ({calib['source']})")

    result: dict = {"engine": calib}
    if args.paper:
        calls = args.paper_n * (args.paper_avg_alarm + 1)
        input_tok = calls * (args.paper_avg_alarm + 1) * args.paper_avg_chars_per_step / args.chars_per_token
        output_tok = calls * args.out_tokens_per_call
        est = estimate(input_tok, output_tok, calib)
        result["paper"] = {
            "n_samples": args.paper_n, "avg_steps": args.paper_avg_steps,
            "avg_alarm_step": args.paper_avg_alarm, "calls": calls,
            "input_tokens": input_tok, "output_tokens": output_tok,
            **est,
        }
        print(f"\npaper estimate (n={args.paper_n}, avg_steps={args.paper_avg_steps}, "
              f"avg_alarm={args.paper_avg_alarm}): calls={calls:,.0f}")
        print(f"  input={input_tok:,.0f} tok, output={output_tok:,.0f} tok, "
              f"total={input_tok + output_tok:,.0f} tok")
        print(f"  time ~{est['total_s']/3600:.2f} h (1x A800; {calib['source']})")

    prof_path = Path(args.profile)
    if prof_path.exists():
        prof = json.loads(prof_path.read_text(encoding="utf-8"))
        scenarios = list(prof["totals"].keys())
        for sc in scenarios:
            t = prof["totals"][sc]
            for col in ("naive", "compressed_ub"):
                key = "total_tokens_" + col
                est = estimate(t["input_tokens_" + col]["total"], t["output_tokens"]["total"], calib)
                print(f"\n[{sc} / {col}] calls={t['calls']['total']:,.0f} "
                      f"input={t['input_tokens_' + col]['total']:,.0f} "
                      f"output={t['output_tokens']['total']:,.0f} tok "
                      f"-> time ~{est['total_s']/3600:.2f} h (1x A800)")
                result[f"{sc}_{col}"] = est
            # naive vs compressed saving
            t_n = t["total_tokens_naive"]["total"]; t_c = t["total_tokens_compressed_ub"]["total"]
            result[f"{sc}_compression_saving"] = {"naive": t_n, "compressed_ub": t_c,
                                                  "saving_pct": (1 - t_c / t_n) * 100 if t_n else None}

    if args.measured:
        meas_path = Path(args.measured)
        raw = meas_path.read_text(encoding="utf-8")
        try:
            meas = json.loads(raw)                       # results.json form
            rows = meas.get("per_sample") or []
        except json.JSONDecodeError:                      # per_sample.jsonl form
            rows = []
            for line in raw.splitlines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_record_type") == "config" or not obj.get("conv_id"):
                    continue
                rows.append(obj)
        seen: set[str] = set()
        dedup = []
        for r in rows:
            if r["conv_id"] in seen:
                continue
            seen.add(r["conv_id"])
            dedup.append(r)
        rows = dedup
        m = {
            "calls": sum(r.get("num_calls", 0) for r in rows),
            "gen_s": sum(r.get("gen_time_s", 0) or 0 for r in rows),
            "wall_s": sum(r.get("wall_time_s", 0) or 0 for r in rows),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
            "n": len(rows),
        }
        m["total_tokens"] = m["prompt_tokens"] + m["completion_tokens"]
        est_n = estimate(m["prompt_tokens"], m["completion_tokens"], calib)
        rg = _pct(est_n["total_s"], m["gen_s"])
        rw = _pct(est_n["total_s"], m["wall_s"])
        print(f"\nmeasured run ({meas_path}): n={m['n']} calls={m['calls']:,.0f} "
              f"gen={m['gen_s']:.1f}s wall={m['wall_s']:.1f}s "
              f"tokens={m['total_tokens']:,.0f}")
        print(f"  predicted time (measured tokens): {est_n['total_s']:.1f}s "
              f"-> residual gen {f'{rg:.1f}%' if rg is not None else 'n/a'}, "
              f"wall {f'{rw:.1f}%' if rw is not None else 'n/a'}")
        result["measured"] = {"measured": m, "predicted": est_n}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

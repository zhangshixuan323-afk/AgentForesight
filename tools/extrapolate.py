# -*- coding: utf-8 -*-
"""Length-aware cost extrapolation + measured-vs-predicted residual check.

Engine model (literature: prefill/decode decomposition, PagedAttention
TTFT/TPOT; FLOPs/bandwidth roofline from Korthikanti et al.):

    t(call k) = input_tokens(k) / prefill_tps(input_tokens(k))
              + output_tokens / decode_tps(ctx_tokens(k))

where prefill_tps / decode_tps are *interpolated per length* from calibrate.py's
measured rows (decode throughput collapses at long context; a global median is
biased). Workload comes from profile.py (per-sample per-call prefix token
curves; use --tokenizer for exact counts). --measured compares predicted vs a
real run's per_sample rows (residual %).

    python3 -m tools.extrapolate --profile outputs/profile_tok.json \
        --calib outputs/calib.json --measured outputs/qwen/final/results.json \
        --output outputs/qwen/extrapolate.json

--paper mode: quick average-based estimate for another dataset (the classic
avg-per-call x avg-steps formula, with input scaled by the alarm-time prefix):
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
            "prefill_rows": c.get("prefill_rows", []),
            "decode_rows": c.get("decode_rows", []),
            "source": str(path),
        }
    return {"prefill_tps": DEFAULT_PREFILL_TPS, "decode_tps": DEFAULT_DECODE_TPS,
            "prefill_rows": [], "decode_rows": [], "source": "roofline-default"}


def _tps_fn(rows: list[dict], default_tps: float):
    """Return fn(length) -> interpolated tokens/s from measured (length, tps) rows."""
    pts = sorted((float(r.get("input_tokens", r.get("ctx_tokens", 0))), float(r["tps"]))
                 for r in rows if r.get("tps"))
    if not pts:
        return lambda _len: default_tps
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    def fn(length: float) -> float:
        if length <= xs[0]:
            return ys[0]
        if length >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= length <= xs[i + 1]:
                t = (length - xs[i]) / max(1e-9, xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]

    return fn


def _pct(pred: float, meas: float) -> float | None:
    return (pred - meas) / meas * 100 if meas else None


def _estimate_calls(inputs: list[float], out_per_call: float,
                    prefill_fn, decode_fn) -> dict:
    """Time of a call sequence with per-call input token counts."""
    t_pre = sum(inp / prefill_fn(inp) for inp in inputs)
    ctxs = inputs  # decode context == input length of the same call
    t_dec = sum(out_per_call / decode_fn(ctx) for ctx in ctxs)
    return {"prefill_s": t_pre, "decode_s": t_dec, "total_s": t_pre + t_dec,
            "input_tokens": sum(inputs), "output_tokens": out_per_call * len(inputs),
            "calls": len(inputs)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="outputs/profile.json")
    p.add_argument("--calib", default=None)
    p.add_argument("--measured", default=None,
                   help="results.json or per_sample.jsonl of a real run (residual check).")
    p.add_argument("--output", default="outputs/extrapolate.json")
    p.add_argument("--out-tokens-per-call", type=int, default=200)
    # paper (average-based) mode
    p.add_argument("--paper", action="store_true")
    p.add_argument("--paper-n", type=int, default=2276)
    p.add_argument("--paper-avg-steps", type=float, default=60.0)
    p.add_argument("--paper-avg-alarm", type=float, default=45.0)
    p.add_argument("--paper-avg-chars-per-step", type=float, default=400.0)
    p.add_argument("--chars-per-token", type=float, default=3.0)
    args = p.parse_args()

    calib = _load_calib(Path(args.calib) if args.calib else None)
    prefill_fn = _tps_fn(calib["prefill_rows"], calib["prefill_tps"])
    decode_fn = _tps_fn(calib["decode_rows"], calib["decode_tps"])
    print(f"engine: prefill_med={calib['prefill_tps']:,.1f} decode_med={calib['decode_tps']:,.1f} "
          f"tok/s (length-aware interp, {calib['source']})")

    result: dict = {"engine": {"prefill_tps": calib["prefill_tps"],
                               "decode_tps": calib["decode_tps"],
                               "source": calib["source"]}}

    if args.paper:
        calls = args.paper_n * (args.paper_avg_alarm + 1)
        input_len = (args.paper_avg_alarm + 1) * args.paper_avg_chars_per_step / args.chars_per_token
        est = _estimate_calls([input_len] * int(calls), args.out_tokens_per_call,
                              prefill_fn, decode_fn)
        result["paper"] = {"n_samples": args.paper_n, "avg_steps": args.paper_avg_steps,
                           "avg_alarm_step": args.paper_avg_alarm, **est}
        print(f"\npaper estimate (n={args.paper_n}, avg_alarm={args.paper_avg_alarm}): "
              f"calls={calls:,.0f}, tokens={est['input_tokens'] + est['output_tokens']:,.0f}, "
              f"time ~{est['total_s'] / 3600:.2f} h (1x A800)")

    prof_path = Path(args.profile)
    if prof_path.exists():
        prof = json.loads(prof_path.read_text(encoding="utf-8"))
        out_per_call = prof.get("config", {}).get("out_tokens_per_call", args.out_tokens_per_call)
        for sc in prof["totals"]:
            t = prof["totals"][sc]
            col = "naive"
            tot_tok = t["total_tokens_" + col]["total"]
            # length-aware per-sample estimate when prefix curves exist
            samples = prof.get("per_sample", [])
            with_curves = [s for s in samples if s.get("prefix_tokens")]
            if with_curves and sc in ("perfect", "worst", "measured"):
                total_s = 0.0
                for s in samples:
                    curve = s.get("prefix_tokens") or []
                    if not curve:
                        continue
                    d = {"perfect": s["gt_step"], "worst": s["n_steps"] - 1}.get(sc, s.get("detection_measured", -1))
                    if sc == "measured" and (d is None or d < 0):
                        d = s["n_steps"] - 1
                    inputs = curve[:d + 1]
                    total_s += _estimate_calls(inputs, out_per_call, prefill_fn, decode_fn)["total_s"]
            else:
                total_s = (t["input_tokens_" + col]["total"] / calib["prefill_tps"]
                           + t["output_tokens"]["total"] / calib["decode_tps"])
            print(f"\n[{sc} / {col}] calls={t['calls']['total']:,.0f} "
                  f"tokens={tot_tok:,.0f} -> time ~{total_s / 3600:.2f} h (1x A800, length-aware)")
            result[f"{sc}_naive"] = {"calls": t["calls"]["total"], "total_tokens": tot_tok,
                                     "total_s": total_s, "hours_1gpu": total_s / 3600}
            t_c = t["total_tokens_compressed_ub"]["total"]
            result[f"{sc}_compression_saving"] = {"naive": tot_tok, "compressed_ub": t_c,
                                                  "saving_pct": (1 - t_c / tot_tok) * 100 if tot_tok else None}

    if args.measured:
        meas_path = Path(args.measured)
        raw = meas_path.read_text(encoding="utf-8")
        try:
            rows = (json.loads(raw) or {}).get("per_sample") or []
        except json.JSONDecodeError:
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
        dedup = [r for r in rows if not (r["conv_id"] in seen or seen.add(r["conv_id"]))]
        m = {
            "calls": sum(r.get("num_calls", 0) for r in dedup),
            "gen_s": sum(r.get("gen_time_s", 0) or 0 for r in dedup),
            "wall_s": sum(r.get("wall_time_s", 0) or 0 for r in dedup),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in dedup),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in dedup),
            "n": len(dedup),
        }
        m["total_tokens"] = m["prompt_tokens"] + m["completion_tokens"]
        # length-aware prediction per measured sample (avg input length per call)
        pred_s = 0.0
        for r in dedup:
            n = max(1, r.get("num_calls", 1))
            avg_in = max(1.0, r.get("prompt_tokens", 0) / n)
            avg_out = max(0.0, r.get("completion_tokens", 0) / n)
            pred_s += avg_in * n / prefill_fn(avg_in) + avg_out * n / decode_fn(avg_in)
        rg = _pct(pred_s, m["gen_s"])
        rw = _pct(pred_s, m["wall_s"])
        print(f"\nmeasured run ({meas_path}): n={m['n']} calls={m['calls']:,.0f} "
              f"gen={m['gen_s']:.1f}s wall={m['wall_s']:.1f}s tokens={m['total_tokens']:,.0f}")
        print(f"  predicted (length-aware)={pred_s:.1f}s -> residual gen "
              f"{f'{rg:.1f}%' if rg is not None else 'n/a'}, wall {f'{rw:.1f}%' if rw is not None else 'n/a'}")
        result["measured"] = {"measured": m, "predicted_s": pred_s,
                              "residual_gen_pct": rg, "residual_wall_pct": rw}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

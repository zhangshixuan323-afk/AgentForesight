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


def _ols3(x1: list[float], x2: list[float], y: list[float]) -> tuple[float, float, float]:
    """Least-squares fit y = a*x1 + b*x2 + c via normal equations (3x3).

    Used to derive *run-condition* engine constants from a measured run:
    gen_s ~ prompt_tokens/prefill_tps + completion_tokens/decode_tps + overhead.
    """
    n = len(x1)
    A = [[sum(v * v for v in x1), sum(a * b for a, b in zip(x1, x2)), sum(x1)],
         [sum(a * b for a, b in zip(x1, x2)), sum(v * v for v in x2), sum(x2)],
         [sum(x1), sum(x2), float(n)]]
    bv = [sum(a * b for a, b in zip(x1, y)),
          sum(a * b for a, b in zip(x2, y)),
          sum(y)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        bv[col], bv[piv] = bv[piv], bv[col]
        for r in range(col + 1, 3):
            f = A[r][col] / A[col][col]
            for c2 in range(col, 3):
                A[r][c2] -= f * A[col][c2]
            bv[r] -= f * bv[col]
    x = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        x[r] = (bv[r] - sum(A[r][c2] * x[c2] for c2 in range(r + 1, 3))) / A[r][r]
    return (x[0], x[1], x[2])


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

    # --- measured run: run-condition engine fit + bias correction -------------
    c_scale = None          # correction factor for length-aware scenario estimates
    fitted = None
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
        # 1) length-aware prediction using the microbenchmark curves
        pred_s = 0.0
        for r in dedup:
            n = max(1, r.get("num_calls", 1))
            avg_in = max(1.0, r.get("prompt_tokens", 0) / n)
            avg_out = max(0.0, r.get("completion_tokens", 0) / n)
            pred_s += avg_in * n / prefill_fn(avg_in) + avg_out * n / decode_fn(avg_in)
        # 2) run-condition engine constants: gen_s = pt/P_eff + ct/D_eff + overhead
        pt_l = [float(r.get("prompt_tokens", 0)) for r in dedup]
        ct_l = [float(r.get("completion_tokens", 0)) for r in dedup]
        g_l = [float(r.get("gen_time_s", 0) or 0) for r in dedup]
        a, b, c = _ols3(pt_l, ct_l, g_l)
        p_eff = 1.0 / a if a > 1e-12 else None
        d_eff = 1.0 / b if b > 1e-12 else None
        pred_fit = [a * pt + b * ct + c for pt, ct in zip(pt_l, ct_l)]
        fitted = {"prefill_tps_eff": p_eff, "decode_tps_eff": d_eff,
                  "overhead_s_per_sample": c}
        rg = _pct(pred_s, m["gen_s"])
        rf = _pct(sum(pred_fit), m["gen_s"])
        rw = _pct(pred_s, m["wall_s"])
        c_scale = m["gen_s"] / pred_s if pred_s > 0 else None
        print(f"\nmeasured run ({meas_path}): n={m['n']} calls={m['calls']:,.0f} "
              f"gen={m['gen_s']:.1f}s wall={m['wall_s']:.1f}s tokens={m['total_tokens']:,.0f}")
        print(f"  predicted (microbench, length-aware)={pred_s:.1f}s -> residual gen "
              f"{f'{rg:.1f}%' if rg is not None else 'n/a'}")
        print(f"  run-condition fit: prefill_eff={p_eff:,.0f} decode_eff={d_eff:,.1f} tok/s "
              f"(overhead {c:.2f}s/sample) -> residual {f'{rf:.1f}%' if rf is not None else 'n/a'}")
        result["measured"] = {"measured": m, "predicted_microbench_s": pred_s,
                              "residual_gen_pct": rg, "residual_wall_pct": rw,
                              "run_condition_fit": fitted,
                              "residual_fit_pct": rf,
                              "correction_scale": c_scale}

    # --- workload scenarios (length-aware, raw + bias-corrected) --------------
    prof_path = Path(args.profile)
    if prof_path.exists():
        prof = json.loads(prof_path.read_text(encoding="utf-8"))
        out_per_call = prof.get("config", {}).get("out_tokens_per_call", args.out_tokens_per_call)
        for sc in prof["totals"]:
            t = prof["totals"][sc]
            col = "naive"
            tot_tok = t["total_tokens_" + col]["total"]
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
            corr_s = total_s * c_scale if c_scale is not None else None
            line = f"[{sc} / {col}] calls={t['calls']['total']:,.0f} tokens={tot_tok:,.0f} -> " \
                   f"~{total_s / 3600:.2f} h (1x A800, microbench)"
            if corr_s is not None:
                line += f" | bias-corrected ~{corr_s / 3600:.2f} h (x{c_scale:.3f})"
            print(line)
            result[f"{sc}_naive"] = {"calls": t["calls"]["total"], "total_tokens": tot_tok,
                                     "total_s": total_s, "hours_1gpu": total_s / 3600,
                                     "hours_1gpu_corrected": corr_s / 3600 if corr_s is not None else None}
            t_c = t["total_tokens_compressed_ub"]["total"]
            result[f"{sc}_compression_saving"] = {"naive": tot_tok, "compressed_ub": t_c,
                                                  "saving_pct": (1 - t_c / tot_tok) * 100 if tot_tok else None}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

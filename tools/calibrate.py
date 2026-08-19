# -*- coding: utf-8 -*-
"""Hardware calibration for the engine cost model (run on the target GPU).

Measures prefill throughput (tokens/s) at several input lengths and decode
throughput (tokens/s) at several context lengths for the auditor model, then
fits the cost function

    t(input, output, ctx) = input / prefill_tps + output / decode_tps(ctx)

and cross-checks against roofline bounds:

    decode_theoretical   = HBM_BW_bytes_per_s / model_bytes          (bandwidth)
    prefill_theoretical  = peak_FLOPS / (2 * params) * efficiency    (compute)

Usage (on the target 3x A800, once the model is downloaded):

    python3 -m tools.calibrate --model-path /path/to/Qwen2.5-7B-Instruct \
        --output outputs/calib.json

--dry-run prints the plan and roofline bounds without loading any model
(no torch/transformers needed).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def roofline(params_billions: float, hbm_bw_gbps: float, peak_tflops: float,
             prefill_eff: float, decode_eff: float) -> dict:
    model_bytes = params_billions * 1e9 * 2.0            # bf16 weights
    decode_theoretical = hbm_bw_gbps * 1e9 / model_bytes
    prefill_theoretical = peak_tflops * 1e12 / (2 * params_billions * 1e9)
    return {
        "model_bytes_bf16": model_bytes,
        "decode_theoretical_tps": decode_theoretical,
        "prefill_theoretical_tps": prefill_theoretical,
        "decode_default_tps": decode_theoretical * decode_eff,
        "prefill_default_tps": prefill_theoretical * prefill_eff,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=None, help="HF id / local path of the auditor (Qwen2.5-7B-Instruct).")
    p.add_argument("--output", default="outputs/calib.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--input-lengths", default="512,2048,4096,8192,16384,24576,30720",
                   help="Comma-separated input lengths (tokens) for prefill/decode measurement.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2, help="Warm-up repetitions per length.")
    p.add_argument("--params-billions", type=float, default=7.0)
    p.add_argument("--hbm-bw-gbps", type=float, default=2000.0, help="A800/A100 HBM bandwidth (GB/s).")
    p.add_argument("--peak-tflops", type=float, default=312.0, help="bf16 dense TFLOPS (A800/A100).")
    p.add_argument("--prefill-eff", type=float, default=0.5, help="assumed prefill kernel efficiency.")
    p.add_argument("--decode-eff", type=float, default=0.7, help="assumed decode bandwidth efficiency.")
    p.add_argument("--dry-run", action="store_true", help="Print plan + roofline only, no model load.")
    args = p.parse_args()

    rl = roofline(args.params_billions, args.hbm_bw_gbps, args.peak_tflops,
                  args.prefill_eff, args.decode_eff)
    lengths = [int(x) for x in args.input_lengths.split(",") if x.strip()]
    print(f"roofline ({args.params_billions}B bf16, {args.hbm_bw_gbps} GB/s, {args.peak_tflops} TFLOPS):")
    print(f"  decode  theoretical ~{rl['decode_theoretical_tps']:.0f} tok/s "
          f"(eff {args.decode_eff} -> default {rl['decode_default_tps']:.0f})")
    print(f"  prefill theoretical ~{rl['prefill_theoretical_tps']:.0f} tok/s "
          f"(eff {args.prefill_eff} -> default {rl['prefill_default_tps']:.0f})")

    if args.dry_run:
        print("\ndry-run: no model loaded. Real run would measure:")
        print(f"  input lengths: {lengths}; max_new_tokens={args.max_new_tokens}")
        return

    if not args.model_path:
        p.error("--model-path is required (or use --dry-run)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nloading {args.model_path} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=args.device)
    model.eval()
    pad_id = tok.eos_token_id or tok.pad_token_id or 0
    dev = model.device

    prefill_rows, decode_rows = [], []
    with torch.no_grad():
        for L in lengths:
            ids = torch.full((1, L), pad_id, dtype=torch.long, device=dev)
            for _ in range(args.warmup):
                model(ids, use_cache=True)
            t0 = time.perf_counter()
            model(ids, use_cache=True)
            t_pre = time.perf_counter() - t0
            prefill_tps = L / t_pre

            t0 = time.perf_counter()
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            t_gen = time.perf_counter() - t0
            n_out = out.shape[1] - L
            decode_tps = n_out / max(1e-6, t_gen - t_pre)
            prefill_rows.append({"input_tokens": L, "t_s": round(t_pre, 4), "tps": round(prefill_tps, 1)})
            decode_rows.append({"ctx_tokens": L, "t_s": round(t_gen - t_pre, 4),
                                "tps": round(decode_tps, 1), "out_tokens": int(n_out)})
            print(f"L={L:6d}: prefill {prefill_tps:9.1f} tok/s | decode {decode_tps:7.1f} tok/s (ctx {L})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model_path": args.model_path,
        "config": {"params_billions": args.params_billions, "hbm_bw_gbps": args.hbm_bw_gbps,
                   "peak_tflops": args.peak_tflops, "max_new_tokens": args.max_new_tokens},
        "roofline": rl,
        "prefill_tps_median": statistics.median(r["tps"] for r in prefill_rows),
        "decode_tps_median": statistics.median(r["tps"] for r in decode_rows),
        "prefill_rows": prefill_rows,
        "decode_rows": decode_rows,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"fitted: prefill_tps_median={result['prefill_tps_median']:.1f}  "
          f"decode_tps_median={result['decode_tps_median']:.1f}")


if __name__ == "__main__":
    main()

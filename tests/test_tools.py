# -*- coding: utf-8 -*-
"""Unit tests for P5 cost tooling: length-aware tps interpolation.

Runs standalone:  python -m tests.test_tools
Also pytest-compatible.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.extrapolate import _tps_fn

# measured on the remote A800 (Qwen2.5-7B-Instruct, shared GPU)
PREFILL_ROWS = [{"input_tokens": 512, "tps": 12763.6}, {"input_tokens": 2048, "tps": 5755.9},
                {"input_tokens": 4096, "tps": 5979.2}, {"input_tokens": 8192, "tps": 5901.7},
                {"input_tokens": 16384, "tps": 5599.3}, {"input_tokens": 24576, "tps": 5242.5},
                {"input_tokens": 30720, "tps": 4972.5}]
DECODE_ROWS = [{"ctx_tokens": 512, "tps": 19.6}, {"ctx_tokens": 2048, "tps": 19.0},
               {"ctx_tokens": 4096, "tps": 16.7}, {"ctx_tokens": 8192, "tps": 12.9},
               {"ctx_tokens": 16384, "tps": 8.7}, {"ctx_tokens": 24576, "tps": 6.4},
               {"ctx_tokens": 30720, "tps": 4.3}]


def test_interp_clamps_and_interpolates() -> None:
    fn = _tps_fn(PREFILL_ROWS, 5000.0)
    assert abs(fn(100) - 12763.6) < 1e-6          # below range -> first point
    assert abs(fn(1_000_000) - 4972.5) < 1e-6      # above range -> last point
    assert abs(fn(2048) - 5755.9) < 1e-6           # exact point
    # midpoint between 2048 (5755.9) and 4096 (5979.2)
    mid = fn(3072)
    assert abs(mid - (5755.9 + 5979.2) / 2) < 1e-6
    assert fn(500) > fn(30000)                      # short input faster than long


def test_empty_rows_use_default() -> None:
    fn = _tps_fn([], 100.0)
    assert fn(12345) == 100.0


def test_decode_long_context_slower() -> None:
    fn = _tps_fn(DECODE_ROWS, 12.9)
    assert fn(512) > fn(30720)                      # 19.6 vs 4.3


def _run_extrapolate(args: list[str]) -> None:
    import subprocess
    sys.exit_code = subprocess.call([sys.executable, "-m", "tools.extrapolate", *args],
                                    cwd=str(Path(__file__).resolve().parents[1]))
    assert sys.exit_code == 0, f"extrapolate exited {sys.exit_code}"


def test_extrapolate_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    tmp = root / "outputs" / "_tool_test_tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        calib = {"prefill_tps_median": 5755.9, "decode_tps_median": 12.9,
                 "prefill_rows": PREFILL_ROWS, "decode_rows": DECODE_ROWS}
        (tmp / "calib.json").write_text(json.dumps(calib), encoding="utf-8")
        # synthetic measured rows (3 samples)
        rows = [
            {"conv_id": "a", "num_calls": 3, "prompt_tokens": 3000,
             "completion_tokens": 150, "gen_time_s": 60.0, "wall_time_s": 62.0},
            {"conv_id": "b", "num_calls": 5, "prompt_tokens": 25000,
             "completion_tokens": 250, "gen_time_s": 200.0, "wall_time_s": 203.0},
            {"conv_id": "c", "num_calls": 10, "prompt_tokens": 200000,
             "completion_tokens": 500, "gen_time_s": 800.0, "wall_time_s": 805.0},
        ]
        (tmp / "measured.json").write_text(json.dumps({"per_sample": rows}), encoding="utf-8")
        _run_extrapolate(["--profile", "outputs/profile.json",
                          "--calib", str(tmp / "calib.json"),
                          "--measured", str(tmp / "measured.json"),
                          "--output", str(tmp / "out.json")])
        out = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
        assert "perfect_naive" in out and "worst_naive" in out
        assert "residual_gen_pct" in out["measured"]
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

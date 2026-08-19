"""Shared online-audit engine and auditor backends.

Auditor backends
----------------
- ``LocalAuditor`` : transformers model (used by infer_local.py)
- ``ApiAuditor``   : OpenAI-compatible endpoint (used by infer_api.py)
- ``OracleAuditor``: ground-truth replay. Validates the whole pipeline:
  Step-Acc must be 100, ASS 0, Agent-Acc 100 (eligible set).
- ``MockAuditor``  : rule-based fake verdicts. Validates timing capture,
  resume, sharding and the report skeleton without any model.

The engine implements the paper's online protocol: for every unsafe trajectory
the auditor is queried at prefixes k = 0..N-1 and the run stops at the FIRST
valid non-SAFE verdict (``d(τ)`` = detection_step); SAFE-only samples get one
full-trajectory call. Per-sample rows carry step/agent verdicts plus time and
token statistics (wall_time_s, gen_time_s, num_calls, prompt/completion tokens,
compressed-call count) — the "per-sample inference time" requirement.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from inference.data import TrajectoryRecord
from inference.metrics import (
    agent_row_fields,
    aggregate_by_domain,
    aggregate_cost,
    format_cost_report,
    format_report,
    per_sample_metrics,
)
from inference.prompts import AuditVerdict, build_messages, parse_response


# --- call statistics ---------------------------------------------------------

class CallStats:
    __slots__ = ("gen_s", "prompt_tokens", "completion_tokens", "compress")

    def __init__(self, gen_s: float = 0.0, prompt_tokens: int = 0,
                 completion_tokens: int = 0, compress: dict | None = None):
        self.gen_s = gen_s
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.compress = compress


# --- backends ----------------------------------------------------------------

class OracleAuditor:
    """Ground-truth replay: alarms exactly at the annotated decisive step."""

    name = "oracle"

    def judge(self, rec: TrajectoryRecord, k: int | None,
              max_new_tokens: int = 2048, temperature: float = 0.0,
              ) -> tuple[AuditVerdict, CallStats]:
        if k is None or k >= rec.mistake_step:
            return (AuditVerdict(rec.mistake_step, rec.mistake_agent, "oracle replay", True, ""),
                    CallStats())
        return AuditVerdict(-1, "", "", True, ""), CallStats()


class MockAuditor:
    """Rule-based fake auditor for pipeline/timing/resume validation."""

    name = "mock"

    def __init__(self, mode: str = "safe", seed: int = 0):
        self.mode = mode
        self.rng = random.Random(seed)

    def judge(self, rec: TrajectoryRecord, k: int | None,
              max_new_tokens: int = 2048, temperature: float = 0.0,
              ) -> tuple[AuditVerdict, CallStats]:
        n = rec.num_turns
        last = k if k is not None else n - 1
        if self.mode == "safe":
            return AuditVerdict(-1, "", "", True, ""), CallStats()
        if self.mode == "last":
            if last == n - 1:
                name = rec.turns[last].get("name", "")
                return AuditVerdict(last, name, "mock last", True, ""), CallStats()
            return AuditVerdict(-1, "", "", True, ""), CallStats()
        if self.mode == "random":
            if self.rng.random() < 0.5:
                step = self.rng.randint(0, last)
                return AuditVerdict(step, rec.turns[step].get("name", ""), "mock random", True, ""), CallStats()
            return AuditVerdict(-1, "", "", True, ""), CallStats()
        raise ValueError(f"unknown mock mode: {self.mode}")


class LocalAuditor:
    """transformers auditor (bf16, device_map='auto')."""

    name = "local"

    def __init__(self, model, tokenizer, *, max_input_tokens: int = 30720,
                 keep_recent: int = 24, per_turn_chars: int = 12000,
                 summary_chars: int = 600):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.keep_recent = keep_recent
        self.per_turn_chars = per_turn_chars
        self.summary_chars = summary_chars

    def _generate(self, messages: list[dict], max_new_tokens: int,
                  temperature: float) -> tuple[str, CallStats]:
        import torch  # lazy: oracle/mock runs need no torch

        t0 = time.perf_counter()
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        with torch.no_grad():
            out = self.model.generate(inputs.input_ids, **gen_kwargs)
        new_ids = out[0][inputs.input_ids.shape[1]:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        return raw, CallStats(gen_s=time.perf_counter() - t0,
                              prompt_tokens=int(inputs.input_ids.shape[1]),
                              completion_tokens=int(new_ids.shape[0]))

    def judge(self, rec: TrajectoryRecord, k: int | None,
              max_new_tokens: int = 2048, temperature: float = 0.0,
              ) -> tuple[AuditVerdict, CallStats]:
        msgs, cstats = build_messages(
            rec.turns, tools=rec.tools, current_step=k,
            tokenizer=self.tokenizer, max_input_tokens=self.max_input_tokens,
            keep_recent=self.keep_recent, per_turn_chars=self.per_turn_chars,
            summary_chars=self.summary_chars, return_stats=True,
        )
        raw, stats = self._generate(msgs, max_new_tokens, temperature)
        stats.compress = cstats
        return parse_response(raw, max_step=k), stats


class ApiAuditor:
    """OpenAI-compatible chat-completions auditor."""

    name = "api"

    def __init__(self, client, model: str, *, max_input_tokens: int = 30720,
                 keep_recent: int = 24, per_turn_chars: int = 12000,
                 summary_chars: int = 600):
        self.client = client
        self.model = model
        self.max_input_tokens = max_input_tokens
        self.keep_recent = keep_recent
        self.per_turn_chars = per_turn_chars
        self.summary_chars = summary_chars

    def judge(self, rec: TrajectoryRecord, k: int | None,
              max_new_tokens: int = 2048, temperature: float = 0.0,
              ) -> tuple[AuditVerdict, CallStats]:
        msgs, cstats = build_messages(
            rec.turns, tools=rec.tools, current_step=k,
            tokenizer=None, max_input_tokens=self.max_input_tokens,
            keep_recent=self.keep_recent, per_turn_chars=self.per_turn_chars,
            summary_chars=self.summary_chars, return_stats=True,
        )
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=msgs,
            max_tokens=max_new_tokens, temperature=temperature,
        )
        raw = resp.choices[0].message.content or ""
        gen_s = time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", None) or cstats["input_tokens_estimate"]
        ct = getattr(usage, "completion_tokens", None) or max(1, len(raw) // 3)
        return parse_response(raw, max_step=k), CallStats(gen_s, int(pt), int(ct), cstats)


# --- engine ------------------------------------------------------------------

def audit_one(rec: TrajectoryRecord, auditor, max_new_tokens: int,
              temperature: float) -> dict:
    wall0 = time.perf_counter()
    total_gen = 0.0
    n_calls = 0
    pt = ct = n_compressed = 0

    if rec.label == "safe":
        verdict, st = auditor.judge(rec, None, max_new_tokens, temperature)
        total_gen += st.gen_s
        n_calls = 1
        pt += st.prompt_tokens
        ct += st.completion_tokens
        if st.compress and st.compress.get("verbatim") is False:
            n_compressed += 1
        detection_step = -1
    else:
        last: AuditVerdict | None = None
        detection_step = -1
        for k in range(rec.num_turns):
            verdict, st = auditor.judge(rec, k, max_new_tokens, temperature)
            total_gen += st.gen_s
            n_calls += 1
            pt += st.prompt_tokens
            ct += st.completion_tokens
            if st.compress and st.compress.get("verbatim") is False:
                n_compressed += 1
            last = verdict
            if verdict.valid and verdict.pred_step >= 0:
                detection_step = k          # d(tau): earliest ALARM prefix
                break
        verdict = last if last is not None else AuditVerdict(-1, "", "", False, "")

    wall = time.perf_counter() - wall0
    return {
        "conv_id": rec.conv_id,
        "domain": rec.domain,
        "label": rec.label,
        "split": rec.split,
        "src_path": rec.src_path,
        "gt_step": rec.mistake_step,
        "gt_agent": rec.mistake_agent,
        "pred_step": verdict.pred_step,
        "pred_agent": verdict.pred_agent,
        "pred_reason": verdict.pred_reason,
        "format_valid": verdict.valid,
        "detection_step": detection_step,     # d(tau); -1 == never alarmed
        "num_turns": rec.num_turns,
        "num_calls": n_calls,
        "gen_time_s": round(total_gen, 4),
        "wall_time_s": round(wall, 4),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "n_compressed_calls": n_compressed,
        "raw_response": verdict.raw_response[:3000],
        **per_sample_metrics(verdict.pred_step, rec.mistake_step),
        **agent_row_fields(verdict.pred_agent, rec.mistake_agent),
    }


def _read_rows(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not jsonl_path.exists():
        return rows
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("_record_type") != "config" and obj.get("conv_id"):
            rows.append(obj)
    return rows


def _write_summary(json_path: Path, rows: list[dict]) -> None:
    json_path.write_text(
        json.dumps({"by_domain": aggregate_by_domain(rows),
                    "cost": aggregate_cost(rows),
                    "per_sample": rows}, indent=2, default=str),
        encoding="utf-8",
    )


def run_audit(records: list[TrajectoryRecord], auditor, output_dir,
              *, backend: str = "", max_new_tokens: int = 2048,
              temperature: float = 0.0, resume: bool = True,
              snapshot_every: int = 25, extra_config: dict | None = None) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "per_sample.jsonl"
    json_path = out_dir / "results.json"

    per_sample: list[dict] = []
    if resume and jsonl_path.exists():
        per_sample = _read_rows(jsonl_path)
        done = {r["conv_id"] for r in per_sample}
        todo = [r for r in records if r.conv_id not in done]
        if done:
            print(f"resume: {len(done)} samples already done, {len(todo)} remaining")
        records = todo

    fresh = not jsonl_path.exists()
    with open(jsonl_path, "a", encoding="utf-8") as f:
        if fresh:
            config = {
                "_record_type": "config",
                "backend": backend,
                "auditor": getattr(auditor, "name", type(auditor).__name__),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                **(extra_config or {}),
            }
            f.write(json.dumps(config, ensure_ascii=False) + "\n")
            f.flush()
        for idx, rec in enumerate(tqdm(records, desc=f"audit[{backend}]")):
            row = audit_one(rec, auditor, max_new_tokens, temperature)
            per_sample.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            f.flush()
            if snapshot_every and (idx + 1) % snapshot_every == 0:
                _write_summary(json_path, per_sample)

    _write_summary(json_path, per_sample)
    by_domain = aggregate_by_domain(per_sample)
    cost = aggregate_cost(per_sample)
    print("\n" + format_report(by_domain))
    print("\n" + format_cost_report(cost))
    print(f"\nResults written to {json_path}  (n={len(per_sample)})")
    return {"by_domain": by_domain, "cost": cost, "n": len(per_sample)}

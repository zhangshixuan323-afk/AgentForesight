"""OpenAI-compatible API (or oracle/mock) online auditing inference.

Same protocol as ``infer_local.py`` but the auditor is queried over an
OpenAI-compatible chat completions endpoint. Set ``OPENAI_API_KEY`` and,
optionally, ``OPENAI_BASE_URL`` to target any OpenAI-compatible provider
(DeepSeek, vLLM-served local model, etc.). Token statistics use the provider's
``usage`` when present, with a char-based fallback.

Example::

    export OPENAI_API_KEY=sk-...
    python -m inference.infer_api \\
        --model       gpt-4.1 \\
        --data-dir    ./data \\
        --output-dir  ./outputs/gpt41
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from inference.audit import ApiAuditor, MockAuditor, OracleAuditor, run_audit
from inference.data import load_aftraj, load_sample100


def _load_records(args) -> list:
    data_dir = Path(args.data_dir)
    is_aftraj = (data_dir / "aftraj_safe.parquet").exists() or (data_dir / "aftraj_unsafe.parquet").exists()
    if args.data_format == "sample100" or (args.data_format == "auto" and not is_aftraj):
        records, excluded = load_sample100(
            data_dir,
            exclude_splits=tuple(s.strip() for s in args.exclude_splits.split(",")),
            domains=args.domains,
            limit=args.max_trajs,
        )
        print(f"Loaded {len(records)} sample100 trajectories "
              f"({len(excluded)} train/validation files excluded)")
        return records
    records = load_aftraj(data_dir, domains=args.domains, limit=args.max_trajs,
                          paper_test_split=args.paper_test_split)
    print(f"Loaded {len(records)} AFTraj trajectories from {data_dir}")
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="api", choices=["api", "oracle", "mock"])
    p.add_argument("--mock-mode", default="safe", choices=["safe", "last", "random"])
    p.add_argument("--model", default=None, help="OpenAI-compatible model name (e.g. gpt-4.1).")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--data-format", default="auto", choices=["auto", "aftraj", "sample100"])
    p.add_argument("--output-dir", default="./outputs_api")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-trajs", type=int, default=None, help="Optional cap (smoke test).")
    p.add_argument("--domains", type=str, default=None, help="Comma-separated domain whitelist.")
    p.add_argument("--base-url", default=None, help="Override OpenAI base URL (also $OPENAI_BASE_URL).")
    p.add_argument("--paper-test-split", action="store_true",
                   help="Restrict to the held-out test split used in the paper's main table.")
    p.add_argument("--macro-domain", action="store_true",
                   help="Aggregate metrics by the paper's 3-way macro buckets (Math/Coding/Agentic).")
    p.add_argument("--exclude-splits", default="train,validation",
                   help="Comma-separated split markers to exclude (sample100).")
    p.add_argument("--max-input-tokens", type=int, default=30720,
                   help="Input budget B (default = 32k context minus 2k generation margin).")
    p.add_argument("--keep-recent", type=int, default=24)
    p.add_argument("--per-turn-chars", type=int, default=12000)
    p.add_argument("--summary-chars", type=int, default=600)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--snapshot-every", type=int, default=25)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()

    if args.backend == "api" and not args.model:
        p.error("--model is required for --backend api")

    records = _load_records(args)
    if args.shard_count > 1:
        records = [r for i, r in enumerate(records) if i % args.shard_count == args.shard_index]
        print(f"shard {args.shard_index}/{args.shard_count}: {len(records)} samples")

    domains = [d.strip() for d in args.domains.split(",")] if args.domains else None

    if args.backend == "oracle":
        auditor = OracleAuditor()
    elif args.backend == "mock":
        auditor = MockAuditor(args.mock_mode)
    else:
        if "OPENAI_API_KEY" not in os.environ:
            raise SystemExit("OPENAI_API_KEY is not set.")
        from openai import OpenAI
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
        auditor = ApiAuditor(client, args.model,
                             max_input_tokens=args.max_input_tokens,
                             keep_recent=args.keep_recent,
                             per_turn_chars=args.per_turn_chars,
                             summary_chars=args.summary_chars)

    run_audit(
        records, auditor, args.output_dir,
        backend=args.backend,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        resume=not args.no_resume,
        snapshot_every=args.snapshot_every,
        extra_config={
            "model": args.model,
            "data_format": args.data_format,
            "data_dir": str(Path(args.data_dir).resolve()),
            "domains": domains,
            "exclude_splits": args.exclude_splits,
            "max_input_tokens": args.max_input_tokens,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
    )


if __name__ == "__main__":
    main()

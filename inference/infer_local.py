"""Local-model (or oracle/mock) online auditing inference.

Runs the paper's online protocol over AFTraj-2K parquets or the
sample100_by_benchmark JSON corpus (train/validation excluded by default).
Every per-sample row carries the step/agent verdicts plus time and token
statistics; the summary report prints Step-Acc / Agent-Acc / ASS / Exact-F1
(Appendix B.2) and the cost aggregates.

Examples::

    python -m inference.infer_local --backend oracle \\
        --data-format sample100 --data-dir ./sample100_by_benchmark \\
        --output-dir ./outputs/oracle

    python -m inference.infer_local \\
        --model-path <hf_repo_or_local_path> \\
        --data-dir ./data \\
        --output-dir ./outputs/af7b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inference.audit import LocalAuditor, MockAuditor, OracleAuditor, run_audit
from inference.data import load_aftraj, load_sample100


def load_model(model_path: str, device: str = "auto") -> tuple:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading auditor from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"  loaded; device={model.device}")
    return model, tokenizer


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
    p.add_argument("--backend", default="local", choices=["local", "oracle", "mock"],
                   help="auditor backend (local needs --model-path)")
    p.add_argument("--mock-mode", default="safe", choices=["safe", "last", "random"])
    p.add_argument("--model-path", default=None, help="HF repo id or local path of the auditor.")
    p.add_argument("--data-dir", default="./data",
                   help="Directory: aftraj parquets, or the sample100_by_benchmark JSON corpus.")
    p.add_argument("--data-format", default="auto", choices=["auto", "aftraj", "sample100"])
    p.add_argument("--output-dir", default="./outputs", help="Where to write per_sample.jsonl + results.json.")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-trajs", type=int, default=None, help="Optional cap (smoke test).")
    p.add_argument("--domains", type=str, default=None, help="Comma-separated domain whitelist.")
    p.add_argument("--paper-test-split", action="store_true",
                   help="Restrict to the held-out test split used in the paper's main table.")
    p.add_argument("--macro-domain", action="store_true",
                   help="Aggregate metrics by the paper's 3-way macro buckets (Math/Coding/Agentic).")
    # sample100 / protocol knobs
    p.add_argument("--exclude-splits", default="train,validation",
                   help="Comma-separated split markers to exclude from the test set (sample100).")
    p.add_argument("--max-input-tokens", type=int, default=30720,
                   help="Input budget B (default = 32k context minus 2k generation margin).")
    p.add_argument("--keep-recent", type=int, default=24,
                   help="Compression V1: number of recent turns kept verbatim.")
    p.add_argument("--per-turn-chars", type=int, default=12000,
                   help="Compression V1: per-turn content cap for verbatim turns.")
    p.add_argument("--summary-chars", type=int, default=600,
                   help="Compression V1: L1 summary length per compressed turn.")
    p.add_argument("--no-resume", action="store_true",
                   help="Restart from scratch instead of skipping already-audited conv_ids.")
    p.add_argument("--snapshot-every", type=int, default=25,
                   help="Write an intermediate results.json every N samples.")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1,
                   help=">1 splits samples round-robin across processes (3x A800 sharding).")
    args = p.parse_args()

    if args.backend == "local" and not args.model_path:
        p.error("--model-path is required for --backend local")

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
        model, tokenizer = load_model(args.model_path, args.device)
        auditor = LocalAuditor(model, tokenizer,
                               max_input_tokens=args.max_input_tokens,
                               keep_recent=args.keep_recent,
                               per_turn_chars=args.per_turn_chars,
                               summary_chars=args.summary_chars)

    run_audit(
        records, auditor, args.output_dir,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        resume=not args.no_resume,
        snapshot_every=args.snapshot_every,
        extra_config={
            "data_format": args.data_format,
            "data_dir": str(Path(args.data_dir).resolve()),
            "domains": domains,
            "exclude_splits": args.exclude_splits,
            "max_input_tokens": args.max_input_tokens,
            "keep_recent": args.keep_recent,
            "per_turn_chars": args.per_turn_chars,
            "summary_chars": args.summary_chars,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
    )


if __name__ == "__main__":
    main()

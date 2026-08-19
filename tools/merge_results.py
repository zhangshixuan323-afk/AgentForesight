# -*- coding: utf-8 -*-
"""Merge per-shard per_sample.jsonl outputs into a single results.json.

    python -m tools.merge_results --dirs outputs/shard0,outputs/shard1,outputs/shard2 \
        --out outputs/final

Rows are deduplicated by conv_id (first occurrence wins) and re-aggregated
(by_domain + cost), producing the final report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.metrics import aggregate_by_domain, aggregate_cost, format_cost_report, format_report


def collect_rows(dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for d in dirs:
        for jl in sorted(d.glob("per_sample.jsonl")):
            for line in jl.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_record_type") == "config" or not obj.get("conv_id"):
                    continue
                if obj["conv_id"] in seen:
                    continue
                seen.add(obj["conv_id"])
                rows.append(obj)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dirs", required=True, help="Comma-separated shard output dirs.")
    p.add_argument("--out", default="outputs/final", help="Output directory for results.json.")
    args = p.parse_args()

    dirs = [Path(x) for x in args.dirs.split(",")]
    rows = collect_rows(dirs)
    if not rows:
        raise SystemExit("no per-sample rows found in the given dirs")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_domain = aggregate_by_domain(rows)
    cost = aggregate_cost(rows)
    (out / "results.json").write_text(
        json.dumps({"by_domain": by_domain, "cost": cost, "per_sample": rows},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"merged {len(rows)} unique samples from {len(dirs)} shards -> {out / 'results.json'}")
    print("\n" + format_report(by_domain))
    print("\n" + format_cost_report(cost))


if __name__ == "__main__":
    main()

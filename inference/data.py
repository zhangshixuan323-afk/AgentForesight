"""Loaders for the AFTraj-2K parquet artifacts and the sample100 JSON corpus.

- ``load_aftraj`` reads ``aftraj_safe.parquet`` / ``aftraj_unsafe.parquet``.
- ``load_sample100`` reads the ``sample100_by_benchmark/**/*.json`` corpus
  (top-level keys: ``question_ID``, ``history``, ``mistake_agent``,
  ``mistake_step``, ``mistake_reason``; each ``history`` entry carries
  ``step``/``content``/``role``/``name``). Every sample is unsafe.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TrajectoryRecord:
    conv_id: str
    domain: str
    label: str                               
    task: str
    gold_answer: str
    num_turns: int
    turns: list[dict]
    tools: list[dict] = field(default_factory=list)
    mistake_step: int = -1                         
    mistake_agent: str = ""
    mistake_reason: str = ""
    unsafe_source: str = ""
    split: str = ""            # benchmark split marker (train/validation/test/""), sample100 only
    src_path: str = ""         # source file path, sample100 only

def _as_list(value) -> list:
    if value is None:
        return []
    return [v for v in value]

def _row_to_record(row: dict, label: str) -> TrajectoryRecord:
                                                                           
    turns = [dict(t) for t in _as_list(row.get("turns"))]
    tools = [dict(t) for t in _as_list(row.get("tools"))]
    return TrajectoryRecord(
        conv_id=str(row["conv_id"]),
        domain=str(row["domain"]),
        label=label,
        task=str(row.get("task", "")),
        gold_answer=str(row.get("gold_answer", "")),
        num_turns=int(row.get("num_turns", len(turns))),
        turns=turns,
        tools=tools,
        mistake_step=int(row.get("mistake_step", -1)),
        mistake_agent=str(row.get("mistake_agent", "")),
        mistake_reason=str(row.get("mistake_reason", "")),
        unsafe_source=str(row.get("unsafe_source", "")),
    )

def load_aftraj(data_dir: str | Path,
                domains: list[str] | None = None,
                splits: tuple[str, ...] = ("safe", "unsafe"),
                limit: int | None = None,
                paper_test_split: bool = False) -> list[TrajectoryRecord]:
    import pandas as pd  # lazy: only the parquet loader needs pandas

    data_dir = Path(data_dir)
    files = {
        "safe":   data_dir / "aftraj_safe.parquet",
        "unsafe": data_dir / "aftraj_unsafe.parquet",
    }
    test_ids: dict[str, set[str]] | None = None
    if paper_test_split:
        sp = data_dir / "splits_test.json"
        if not sp.exists():
            raise FileNotFoundError(f"paper_test_split=True but missing: {sp}")
        sj = json.load(open(sp))
        test_ids = {"safe": set(sj["test_safe"]), "unsafe": set(sj["test_unsafe"])}

    records: list[TrajectoryRecord] = []
    for split in splits:
        if split not in files:
            raise ValueError(f"unknown split: {split!r}")
        path = files[split]
        if not path.exists():
            raise FileNotFoundError(f"parquet not found: {path}")
        df = pd.read_parquet(path)
        if domains is not None:
            df = df[df["domain"].isin(domains)]
        if test_ids is not None:
            df = df[df["conv_id"].isin(test_ids[split])]
        for _, row in df.iterrows():
            records.append(_row_to_record(row.to_dict(), split))
    if limit is not None:
        records = records[:limit]
    return records


# --- sample100_by_benchmark JSON corpus -------------------------------------

_SPLIT_RE = re.compile(r"_(train|validation|test)__")

def detect_split(filename: str) -> str:
    """Return the benchmark split marker embedded in a sample file name.

    TravelPlanner file names carry ``groupchat_train/validation/test__<id>``;
    other benchmarks have no marker and return "".
    """
    m = _SPLIT_RE.search(filename)
    return m.group(1) if m else ""

def _sample100_to_record(obj: dict, path: Path, split: str) -> TrajectoryRecord:
    history = obj.get("history") or []
    turns = [dict(t) for t in history]
    task = ""
    if turns and turns[0].get("role") == "user":
        task = str(turns[0].get("content") or "")
    return TrajectoryRecord(
        conv_id=str(obj.get("question_ID") or path.stem),
        domain=str(path.parent.name),
        label="unsafe",  # every sample in this corpus has a mistake annotation
        task=task,
        gold_answer="",
        num_turns=len(turns),
        turns=turns,
        tools=[],
        mistake_step=int(obj.get("mistake_step", -1)),
        mistake_agent=str(obj.get("mistake_agent", "")),
        mistake_reason=str(obj.get("mistake_reason", "")),
        unsafe_source="",
        split=split,
        src_path=str(path),
    )

def load_sample100(data_dir: str | Path,
                   exclude_splits: tuple[str, ...] = ("train", "validation"),
                   domains: list[str] | None = None,
                   limit: int | None = None) -> tuple[list[TrajectoryRecord], list[Path]]:
    """Load the sample100 JSON corpus.

    Returns ``(records, excluded)`` where ``excluded`` lists the files whose
    filename split marker is in ``exclude_splits`` (default: train/validation
    are not part of the test set). JSON is read with ``errors="replace"`` so
    mojibake content never crashes the loader.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob("*.json"))
    records: list[TrajectoryRecord] = []
    excluded: list[Path] = []
    for p in files:
        split = detect_split(p.stem)
        if split in exclude_splits:
            excluded.append(p)
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                obj = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"bad json: {p}: {e}") from e
        rec = _sample100_to_record(obj, p, split)
        if domains is not None and rec.domain not in domains:
            continue
        records.append(rec)
    if limit is not None:
        records = records[:limit]
    return records, excluded

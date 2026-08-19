"""Agent-identity normalization for Agent-Acc evaluation.

The corpus mixes two name spaces:

- **history ``name`` values** — what the auditor sees in the prompt and can
  output (e.g. ``transport_agent``, ``stay_agent``, ``manager_agent``,
  ``verifier_agent``, ``Task_Planner``, ...).
- **gt ``mistake_agent`` values** — the canonical annotation space
  (``Action_Expert``, ``Task_Planner``, ``Verification_Expert``,
  ``DiagnostAgent (-> ActionAgent)``, ...).

The four layers of the normalization pipeline:

- **L1** deterministic alias families, derived from co-occurrence statistics
  on the 83-sample test set: a history name that maps 100% to one canonical
  label is a family member (e.g. ``transport_agent`` -> ``Action_Expert``).
- **L2** semantic-role disambiguation for the two ambiguous travelplanner
  labels: ``manager_agent`` -> ``Task_Planner`` (orchestrator), and
  ``verifier_agent`` -> ``Verification_Expert`` (blackboard checker). The ~5
  samples whose annotator chose the other label are reported as a disclosed
  caveat, not silently absorbed.
- **L3** special identities (``Computer_terminal`` / ``human``) are the
  environment, not an agent: they are excluded from the Agent-Acc denominator.
- **L4** prompt-side transparency: the prompt only ever exposes history
  ``name`` values; normalization happens on the evaluation side only.

Matching semantics
------------------
``strict_match(pred, gt)``: both sides canonicalized, then exact equality.
``loose_match(pred, gt)``: pred equals gt, or pred is a *deterministic*
alias-family member of gt (ambiguous names are deliberately NOT in families,
so the loose metric does not degenerate into always-correct on noisy rows).
For compound gt labels ``A (-> B)`` the loose family additionally admits the
source/target ends ``A`` and ``B``.
"""

from __future__ import annotations

# --- L1: deterministic alias families (83-sample test set co-occurrence) -----
# canonical label -> raw history names that map to it 100%.
ALIAS_FAMILIES: dict[str, set[str]] = {
    "Action_Expert": {"Action_Expert", "transport_agent", "stay_agent", "food_attr_agent"},
    "Task_Planner": {"Task_Planner"},
    "Verification_Expert": {"Verification_Expert"},
    "ActionAgent": {"ActionAgent"},
    "JudgeAgent": {"JudgeAgent"},
    "DiagnostAgent": {"DiagnostAgent"},
    # compound label: L1 raw member is the exact string only; the delegation
    # source/target ends are admitted by the *loose* rule, not L1 (so strict
    # matching does not conflate DiagnostAgent with the compound label).
    "DiagnostAgent (-> ActionAgent)": {"DiagnostAgent (-> ActionAgent)"},
    "Computer_terminal": {"Computer_terminal"},   # L3 special, never scored
}

# --- L2: semantic-role disambiguation for ambiguous travelplanner labels -----
L2_MAP: dict[str, str] = {
    "manager_agent": "Task_Planner",       # orchestrator decides when to finish
    "verifier_agent": "Verification_Expert",  # blackboard checker
}

# --- L3: environment/user identities, not agents -----------------------------
SPECIAL_NAMES: frozenset[str] = frozenset({"Computer_terminal", "human"})

# reverse index: raw name -> canonical label (L1 only, ambiguous names absent;
# first occurrence wins so a plain name is never shadowed by a compound family)
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _raws in ALIAS_FAMILIES.items():
    for _raw in _raws:
        if _raw in L2_MAP or _raw in _ALIAS_TO_CANONICAL:
            continue
        _ALIAS_TO_CANONICAL[_raw] = _canonical

_AMBIGUOUS: frozenset[str] = frozenset(L2_MAP)


def is_special(name: str) -> bool:
    """L3: is this identity the environment/user rather than an agent?"""
    return _strip(name) in SPECIAL_NAMES


def canonical(name: str) -> str | None:
    """Map any raw name to its canonical label; None if unmappable."""
    n = _strip(name)
    if not n:
        return None
    if n in L2_MAP:
        return L2_MAP[n]
    return _ALIAS_TO_CANONICAL.get(n, n if n in ALIAS_FAMILIES else None)


def strict_match(pred_agent: str, gt_agent: str) -> bool:
    """Both sides canonicalized and equal (L2 applies to pred side)."""
    p = canonical(pred_agent)
    g = canonical(gt_agent)
    return p is not None and g is not None and p == g


def loose_family(gt_agent: str) -> set[str]:
    """Deterministic alias family of a canonical gt label (no ambiguous names)."""
    g = canonical(gt_agent) or _strip(gt_agent)
    fam = set(ALIAS_FAMILIES.get(g, {g}))
    # compound "A (-> B)": admit source/target ends
    if " (-> " in g:
        a, _, b = g.partition(" (-> ")
        fam |= {a.strip(), b.strip().rstrip(")").strip()}
    return fam


def loose_match(pred_agent: str, gt_agent: str) -> bool:
    """pred hits gt exactly, its canonical label, or a deterministic alias."""
    fam = loose_family(gt_agent)
    p_raw = _strip(pred_agent)
    return p_raw in fam or (canonical(pred_agent) or "") in fam


def ambiguous_names() -> frozenset[str]:
    """Names with noisy annotation (disclosed caveat, not family members)."""
    return _AMBIGUOUS


def _strip(name: str) -> str:
    return str(name or "").strip().strip('"').strip("'")

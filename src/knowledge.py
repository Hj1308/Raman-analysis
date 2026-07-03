"""
knowledge.py — Literature reference knowledge base for Raman-analysis.

Loads a curated set of literature reference values (positions, ratios, FWHM,
crystallite-size / defect-density formulas, defect-type thresholds) and exposes
a small query API so the rest of the app can:

  * look up published reference values with citation + measurement conditions
    (replacing hard-coded magic numbers -> Fix #5), and
  * compare a measured quantity against the literature range for validation
    (-> Fix #8 support).

Design principles
------------------
* The ACTIVE base (knowledge_base.json) contains ONLY sp2-carbon materials
  (graphene / GO / rGO / graphite / g-C3N4 / nanocrystalline carbon / doped
  graphene / GNP). SiC, MXene, SWNT, SERS substrates etc. live in
  knowledge_extended.json and are NOT loaded into the active base, so the
  app never accidentally compares a carbon spectrum against a SiC reference.
* Every entry keeps its citation, conditions (laser_nm!), and confidence.
  Numbers are never presented without that context.
* Entries are classified into three kinds so callers can pick what they need:
    - "numeric"     : has a concrete value or range (usable as a threshold)
    - "formula"     : encodes a formula in notes (no single value)
    - "qualitative" : direction/trend only (no value, no formula)

Data files are expected next to this module (or one dir up); paths are
resolved robustly so it works whether imported from src/ or the repo root.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

_CONF_RANK = {"strong": 3, "suggestive": 2, "weak": 1}


# --------------------------------------------------------------------------- #
# Entry model
# --------------------------------------------------------------------------- #
@dataclass
class RefEntry:
    id: str
    material: str
    metric: str
    measure: str
    value: Optional[float]
    range: list
    conditions: dict
    confidence_level: str
    notes: str
    quote: Optional[str]
    citation: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """'numeric' | 'formula' | 'qualitative'."""
        if self.value is not None or (self.range and self.range != [None, None]):
            return "numeric"
        n = (self.notes or "").lower()
        if "formula" in n or "=" in (self.notes or ""):
            return "formula"
        return "qualitative"

    @property
    def laser_nm(self):
        return (self.conditions or {}).get("laser_nm")

    def cite_short(self) -> str:
        c = self.citation or {}
        a = c.get("authors", "?")
        y = c.get("year", "")
        return f"{a} ({y})".strip()

    def describe(self) -> str:
        """One-line, citation-anchored, condition-aware description."""
        cond = []
        if self.laser_nm:
            cond.append(f"{self.laser_nm} nm")
        else:
            cond.append("laser-agnostic")
        for k in ("dopant_concentration", "substrate", "stage", "defect_type"):
            v = (self.conditions or {}).get(k)
            if v:
                cond.append(str(v))
        cond_s = ", ".join(cond)
        if self.value is not None:
            val = f"{self.value:g} {self.measure}"
        elif self.range and self.range != [None, None]:
            lo, hi = self.range
            val = f"{lo}–{hi} {self.measure}"
        else:
            val = self.notes or "(formula/qualitative)"
        return (f"{self.metric} = {val} [{self.cite_short()}; {cond_s}; "
                f"{self.confidence_level}]")


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
class KnowledgeBase:
    def __init__(self, entries: List[RefEntry]):
        self.entries = entries

    # ---- loading ----
    @classmethod
    def _resolve(cls, filename: str) -> Optional[str]:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.join(here, filename),
                     os.path.join(here, "..", filename),
                     os.path.join(os.getcwd(), filename)):
            if os.path.exists(cand):
                return cand
        return None

    @classmethod
    def load(cls, filename: str = "knowledge_base.json") -> "KnowledgeBase":
        path = cls._resolve(filename)
        if path is None:
            return cls([])  # empty base is valid (app still runs)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        entries = []
        for e in doc.get("entries", []):
            entries.append(RefEntry(
                id=e.get("id", ""),
                material=e.get("material", ""),
                metric=e.get("metric", ""),
                measure=e.get("measure", ""),
                value=e.get("value"),
                range=e.get("range", [None, None]),
                conditions=e.get("conditions", {}) or {},
                confidence_level=e.get("confidence_level", "weak"),
                notes=e.get("notes", "") or "",
                quote=e.get("quote"),
                citation=e.get("citation", {}) or {},
            ))
        return cls(entries)

    # ---- queries ----
    def query(self, metric=None, material=None, measure=None,
              laser_nm=None, kind=None, min_confidence=None) -> List[RefEntry]:
        """Filter entries. Substring match on metric/material (case-insensitive).
        laser_nm matches exact or laser-agnostic (None) entries."""
        out = []
        minrank = _CONF_RANK.get(min_confidence, 0) if min_confidence else 0
        for e in self.entries:
            if metric and metric.lower() not in e.metric.lower():
                continue
            if material and material.lower() not in e.material.lower():
                continue
            if measure and measure.lower() != e.measure.lower():
                continue
            if laser_nm is not None and e.laser_nm is not None \
                    and e.laser_nm != laser_nm:
                continue
            if kind and e.kind != kind:
                continue
            if _CONF_RANK.get(e.confidence_level, 0) < minrank:
                continue
            out.append(e)
        # strongest first
        out.sort(key=lambda e: _CONF_RANK.get(e.confidence_level, 0), reverse=True)
        return out

    def reference_range(self, metric, material=None, measure=None,
                        laser_nm=None):
        """Aggregate a plausible literature range for a metric.

        Returns (lo, hi, supporting_entries) using every numeric entry that
        matches. Useful for validation: is the measured value within what the
        literature reports? Returns (None, None, []) if nothing numeric matches.
        """
        matches = [e for e in self.query(metric=metric, material=material,
                                         measure=measure, laser_nm=laser_nm,
                                         kind="numeric")]
        vals = []
        for e in matches:
            if e.value is not None:
                vals.append(e.value)
            if e.range and e.range != [None, None]:
                lo, hi = e.range
                if lo is not None:
                    vals.append(lo)
                if hi is not None:
                    vals.append(hi)
        if not vals:
            return None, None, []
        return min(vals), max(vals), matches

    def defect_type_ladder(self, laser_nm=None):
        """Return the Eckmann I_D/I_D' defect-type reference points, sorted.
        [(value, defect_type, entry), ...] — for defect classification (Fix #5)."""
        out = []
        for e in self.query(metric="I_D/I_D'", kind="numeric", laser_nm=laser_nm):
            dt = (e.conditions or {}).get("defect_type")
            if e.value is not None and dt:
                out.append((e.value, dt, e))
        out.sort(key=lambda t: t[0])
        return out

    def classify_defect_ratio(self, ratio, laser_nm=None, tol=0.15):
        """Locate a measured I_D/I_D' on the Eckmann ladder as a RANGE.

        Rather than snapping to one defect type (misleading for borderline
        values), report where the value sits relative to the reference
        ladder. Returns a dict:
            {
              'ratio': float,
              'position': 'below' | 'between' | 'above' | 'at',
              'lower': (value, type, cite) or None,   # nearest point below
              'upper': (value, type, cite) or None,   # nearest point above
              'nearest': (value, type, cite),         # closest single point
              'ambiguous': bool,   # True when value sits ~midway between two
              'summary': str,      # human-readable, citation-anchored
            }
        Reference points sharing a defect type are collapsed to their mean so
        duplicate literature entries don't clutter the ladder.
        """
        raw = self.defect_type_ladder(laser_nm=laser_nm)
        if not raw or ratio is None:
            return None

        # collapse duplicate defect types to a mean value, keep one citation
        by_type = {}
        for v, dt, e in raw:
            by_type.setdefault(dt, []).append((v, e))
        pts = []
        for dt, items in by_type.items():
            mean_v = sum(v for v, _ in items) / len(items)
            pts.append((mean_v, dt, items[0][1]))
        pts.sort(key=lambda t: t[0])

        lower = None
        upper = None
        for p in pts:
            if p[0] <= ratio:
                lower = p
            if p[0] >= ratio and upper is None:
                upper = p
        nearest = min(pts, key=lambda p: abs(p[0] - ratio))

        if lower and upper and lower[1] != upper[1]:
            position = "between"
            span = upper[0] - lower[0]
            # ambiguous if roughly midway (not close to either endpoint)
            frac = (ratio - lower[0]) / span if span > 0 else 0.0
            ambiguous = 0.5 - tol <= frac <= 0.5 + tol or (0.2 < frac < 0.8)
        elif lower and not upper:
            position = "above"
            ambiguous = False
        elif upper and not lower:
            position = "below"
            ambiguous = False
        else:
            position = "at"
            ambiguous = False

        # build summary
        def _fmt(p):
            return f"{p[1]} (~{p[0]:.1f}, {p[2].cite_short()})"

        if position == "between":
            summary = (f"I_D/I_D' = {ratio:.1f} lies between "
                       f"{_fmt(lower)} and {_fmt(upper)}")
            if ambiguous:
                summary += " — borderline, assignment uncertain"
        elif position == "above":
            summary = (f"I_D/I_D' = {ratio:.1f} is at/above the highest "
                       f"reference point {_fmt(lower)}")
        elif position == "below":
            summary = (f"I_D/I_D' = {ratio:.1f} is below the lowest "
                       f"reference point {_fmt(upper)}")
        else:
            summary = f"I_D/I_D' = {ratio:.1f} matches {_fmt(nearest)}"

        return {
            "ratio": ratio,
            "position": position,
            "lower": lower,
            "upper": upper,
            "nearest": nearest,
            "ambiguous": ambiguous,
            "summary": summary,
        }

    def __len__(self):
        return len(self.entries)


# module-level singletons (lazy)
_ACTIVE = None
_EXTENDED = None


def active() -> KnowledgeBase:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = KnowledgeBase.load("knowledge_base.json")
    return _ACTIVE


def extended() -> KnowledgeBase:
    """Non-carbon references (SiC/MXene/SWNT...). Loaded on demand only."""
    global _EXTENDED
    if _EXTENDED is None:
        _EXTENDED = KnowledgeBase.load("knowledge_extended.json")
    return _EXTENDED

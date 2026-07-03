"""
knowledge.py  –  Raman literature knowledge-base access layer
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ── default file locations ────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_DEFAULT_KB   = _HERE / "knowledge_base.json"
_DEFAULT_EXT  = _HERE / "knowledge_extended.json"


# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class Entry:
    id: str
    citation: Dict[str, Any]
    material: str
    metric: str
    measure: str
    value: Optional[float]
    range: Tuple[Optional[float], Optional[float]]
    conditions: Dict[str, Any]
    confidence_level: str
    notes: str
    quote: Optional[str]
    stage: Optional[str] = None

    # derived
    @property
    def authors(self) -> str:
        return self.citation.get("authors", "")

    @property
    def year(self) -> Optional[int]:
        return self.citation.get("year")

    @property
    def doi(self) -> Optional[str]:
        return self.citation.get("doi")

    def is_numeric(self) -> bool:
        return self.value is not None

    def is_range(self) -> bool:
        return any(v is not None for v in self.range)

    def laser_nm(self) -> Optional[float]:
        return (self.conditions or {}).get("laser_nm")

    def defect_type(self) -> Optional[str]:
        return (self.conditions or {}).get("defect_type")


def _parse_entry(raw: dict) -> Entry:
    cit = raw.get("citation", {})
    rng = raw.get("range", [None, None]) or [None, None]
    conds = raw.get("conditions", {}) or {}
    return Entry(
        id=raw["id"],
        citation=cit,
        material=raw.get("material", ""),
        metric=raw.get("metric", ""),
        measure=raw.get("measure", ""),
        value=raw.get("value"),
        range=(rng[0], rng[1]),
        conditions=conds,
        confidence_level=raw.get("confidenceLevel", raw.get("confidence_level", "")),
        notes=raw.get("notes", "") or "",
        quote=raw.get("quote"),
        stage=raw.get("stage") or conds.get("stage"),
    )


# ── main class ────────────────────────────────────────────────────────────────
class KnowledgeBase:
    """
    Thin wrapper around the Raman literature JSON knowledge-base.

    Parameters
    ----------
    paths : list of path-like, optional
        JSON files to load (merged).  Defaults to the bundled
        ``knowledge_base.json`` only (carbon-only active base).
    """

    def __init__(self, paths: Optional[List[Union[str, Path]]] = None):
        if paths is None:
            paths = [p for p in [_DEFAULT_KB] if p.exists()]
        self._entries: List[Entry] = []
        for p in paths:
            self._load(Path(p))

    # ── loading ──────────────────────────────────────────────────────────────
    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Knowledge-base file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for raw in data.get("entries", []):
            self._entries.append(_parse_entry(raw))

    # ── querying ─────────────────────────────────────────────────────────────
    def query(
        self,
        metric: Optional[str] = None,
        material: Optional[str] = None,
        laser_nm: Optional[float] = None,
        kind: Optional[str] = None,
        confidence: Optional[str] = None,
        defect_type: Optional[str] = None,
    ) -> List[Entry]:
        out = []
        for e in self._entries:
            if metric and e.metric != metric:
                continue
            if material and material.lower() not in e.material.lower():
                continue
            if laser_nm and e.laser_nm() != laser_nm:
                continue
            if kind == "numeric" and not e.is_numeric():
                continue
            if kind == "range" and not e.is_range():
                continue
            if confidence and e.confidence_level != confidence:
                continue
            if defect_type and e.defect_type() != defect_type:
                continue
            out.append(e)
        return out

    def get(self, entry_id: str) -> Optional[Entry]:
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    @property
    def entries(self) -> List[Entry]:
        return list(self._entries)

    # ── convenience methods ──────────────────────────────────────────────────
    def idig_values(
        self,
        material_filter: Optional[str] = None,
        laser_nm: Optional[float] = None,
    ) -> List[Entry]:
        return self.query(metric="I_D/I_G", material=material_filter, laser_nm=laser_nm)

    def i2dig_values(
        self,
        material_filter: Optional[str] = None,
    ) -> List[Entry]:
        return self.query(metric="I_2D/I_G", material=material_filter)

    def defect_type_ladder(self) -> Dict[str, List[Entry]]:
        """
        Return a dict mapping defect_type -> list of I_D/I_D' entries.
        """
        ladder: Dict[str, List[Entry]] = {}
        for e in self.query(metric="I_D/I_D'", kind="numeric"):
            dt = e.defect_type() or "unknown"
            ladder.setdefault(dt, []).append(e)
        return ladder

    def ld_from_idig(
        self,
        idig: float,
        laser_nm: float = 514.0,
    ) -> Optional[float]:
        """
        Estimate defect inter-distance L_D (nm) from I_D/I_G using
        the Lucchese / Cancado formula.
        """
        cl_entries = self.query(metric="C_lambda", kind="numeric")
        if cl_entries:
            cl_entries.sort(key=lambda x: abs((x.laser_nm() or 514) - laser_nm))
            c_lambda = cl_entries[0].value
        else:
            c_lambda = 1.8e-9 * (laser_nm ** 4)
        try:
            ld = math.sqrt(c_lambda / idig)
            return ld
        except (ZeroDivisionError, ValueError):
            return None

    def la_from_idig(
        self,
        idig: float,
        laser_nm: float = 514.0,
    ) -> Optional[float]:
        """
        Estimate crystallite size L_a (nm) from I_D/I_G using Tuinstra-Koenig
        / Cancado formula.
        """
        cl_entries = self.query(metric="C_lambda", kind="numeric")
        if cl_entries:
            cl_entries.sort(key=lambda x: abs((x.laser_nm() or 514) - laser_nm))
            c_lambda = cl_entries[0].value
        else:
            c_lambda = 1.8e-9 * (laser_nm ** 4)
        try:
            la = (2.4e-10 * (laser_nm ** 4)) / idig
            return la
        except ZeroDivisionError:
            return None

    # ── summary helpers ──────────────────────────────────────────────────────
    def summary(self) -> str:
        metrics = sorted({e.metric for e in self._entries})
        materials = sorted({e.material for e in self._entries})
        lines = [
            f"KnowledgeBase  ({len(self._entries)} entries)",
            f"  metrics   : {', '.join(metrics)}",
            f"  materials : {', '.join(materials[:10])}"
            + (" ..." if len(materials) > 10 else ""),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"KnowledgeBase({len(self._entries)} entries)"

    def __len__(self) -> int:
        return len(self._entries)


# ── module-level singletons (lazy) ────────────────────────────────────────────
_ACTIVE: Optional[KnowledgeBase] = None
_EXTENDED: Optional[KnowledgeBase] = None


def active() -> KnowledgeBase:
    """Return the active (carbon-only) knowledge base singleton."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = KnowledgeBase(paths=[_DEFAULT_KB] if _DEFAULT_KB.exists() else [])
    return _ACTIVE


def extended() -> KnowledgeBase:
    """Return extended knowledge base (SiC/MXene/SWNT etc.), loaded on demand."""
    global _EXTENDED
    if _EXTENDED is None:
        _EXTENDED = KnowledgeBase(paths=[_DEFAULT_EXT] if _DEFAULT_EXT.exists() else [])
    return _EXTENDED


def reset_singletons() -> None:
    """Reset cached singletons (useful for testing)."""
    global _ACTIVE, _EXTENDED
    _ACTIVE = None
    _EXTENDED = None

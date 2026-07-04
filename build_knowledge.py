#!/usr/bin/env python3
"""
build_knowledge.py — Merge, deduplicate, and domain-split Raman reference entries.

Reads every *.json in kb_blocks/ (each a {"entries":[...]} document from the
DeepSeek extraction batches), then:

  1. Deduplicates. Two entries collide if they share the same `id` OR the same
     (citation.doi or authors+year) + metric + measure + material + value/range.
     On collision the "better" entry wins, ranked by:
        confidence (strong>suggestive>weak) > has laser_nm > has doi > longer notes
  2. Domain-splits into:
        carbon_sp2  -> graphene / GO / rGO / graphite / g-C3N4 / nanocrystalline
                       carbon / doped graphene / GNP / CNT-free sp2 carbons
        extended    -> everything else (SiC, MXene, SWNT, SERS substrates, glass…)
  3. Writes knowledge_base.json (carbon_sp2, the app's ACTIVE base) and
     knowledge_extended.json (everything else, preserved but inert).

Run:  python build_knowledge.py
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCKS = os.path.join(HERE, "kb_blocks")

_CONF_RANK = {"strong": 3, "suggestive": 2, "weak": 1, None: 0}

# --- domain classification -------------------------------------------------
_CARBON_KEYS = (
    "graphene", "graphite", "graphitic", "go", "rgo", "graphene oxide",
    "g-c3n4", "gc3n4", "c3n4", "carbon nitride", "nanographite",
    "nanocrystalline carbon", "amorphous carbon", "nanoporous carbon",
    "activated carbon", "gnp", "graphene nanoplatelet", "hopg", "carbon",
)
# things that look carbon-ish by substring but are NOT the app's domain
_NOT_CARBON = (
    "sic", "silicon carbide", "mxene", "swnt", "nanotube", "wsi2",
    "si nanoparticle", "b-doped si", "glass", "moo", "wo", "ti2c", "ti3c",
    "mo2c", "cupc", "r6g", "sers", "gers",
)


def is_carbon_sp2(material: str) -> bool:
    m = (material or "").lower()
    if any(k in m for k in _NOT_CARBON):
        # allow "graphene on SiC" (graphene IS the analyte) but reject bare SiC
        if "graphene" in m or "rgo" in m or "graphene oxide" in m:
            return True
        return False
    return any(k in m for k in _CARBON_KEYS)


# --- dedup key -------------------------------------------------------------
def dedup_key(e):
    c = e.get("citation", {}) or {}
    src = c.get("doi") or f"{c.get('authors')}|{c.get('year')}"
    val = e.get("value")
    rng = tuple(e.get("range") or [None, None])
    return (src, e.get("metric"), e.get("measure"),
            (e.get("material") or "").lower(), val, rng)


def better(a, b):
    """Return the entry to keep when a and b collide."""
    def score(e):
        c = e.get("citation", {}) or {}
        return (
            _CONF_RANK.get(e.get("confidence_level"), 0),
            1 if (e.get("conditions") or {}).get("laser_nm") else 0,
            1 if c.get("doi") else 0,
            len(e.get("notes") or ""),
        )
    return a if score(a) >= score(b) else b


def main():
    files = sorted(glob.glob(os.path.join(BLOCKS, "*.json")))
    if not files:
        raise SystemExit(f"No JSON blocks found in {BLOCKS}")

    raw = []
    for fp in files:
        with open(fp) as f:
            doc = json.load(f)
        raw.extend(doc.get("entries", []))
    print(f"read {len(raw)} raw entries from {len(files)} files")

    # dedupe: prefer id-collision first, then content-collision
    by_id = {}
    for e in raw:
        eid = e.get("id")
        if eid in by_id:
            by_id[eid] = better(by_id[eid], e)
        else:
            by_id[eid] = e
    print(f"after id-dedup: {len(by_id)}")

    by_content = {}
    for e in by_id.values():
        k = dedup_key(e)
        if k in by_content:
            by_content[k] = better(by_content[k], e)
        else:
            by_content[k] = e
    unique = list(by_content.values())
    print(f"after content-dedup: {len(unique)}")

    carbon = [e for e in unique if is_carbon_sp2(e.get("material", ""))]
    extended = [e for e in unique if not is_carbon_sp2(e.get("material", ""))]
    print(f"carbon_sp2 (active): {len(carbon)}  |  extended (stored): {len(extended)}")

    with open(os.path.join(HERE, "knowledge_base.json"), "w") as f:
        json.dump({"entries": carbon}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(HERE, "knowledge_extended.json"), "w") as f:
        json.dump({"entries": extended}, f, indent=2, ensure_ascii=False)

    # quick material breakdown for sanity
    from collections import Counter
    print("\ncarbon materials:",
          Counter(e["material"] for e in carbon).most_common(8))
    print("extended materials:",
          Counter(e["material"] for e in extended).most_common(8))


if __name__ == "__main__":
    main()

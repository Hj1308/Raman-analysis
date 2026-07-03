"""Tests for src/knowledge.py — literature reference knowledge base."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import knowledge as K


class TestLoad:
    def test_active_loads(self):
        kb = K.active()
        assert len(kb) > 0

    def test_active_is_carbon_only(self):
        # no SiC / MXene / SWNT should leak into the active base
        kb = K.active()
        bad = ("sic", "mxene", "swnt", "silicon carbide")
        for e in kb.entries:
            m = e.material.lower()
            # graphene-on-SiC is allowed (analyte is graphene); bare SiC is not
            if "graphene" in m or "rgo" in m or "graphene oxide" in m:
                continue
            assert not any(b in m for b in bad), f"leaked: {e.material}"

    def test_extended_separate(self):
        assert len(K.extended()) >= 0  # loads without error


class TestQuery:
    def test_metric_filter(self):
        kb = K.active()
        res = kb.query(metric="I_D/I_D'")
        assert all("I_D/I_D'" in e.metric for e in res)

    def test_confidence_sort(self):
        kb = K.active()
        res = kb.query(metric="L_D")
        ranks = [K._CONF_RANK.get(e.confidence_level, 0) for e in res]
        assert ranks == sorted(ranks, reverse=True)

    def test_kind_classification(self):
        kb = K.active()
        # a formula entry (Cancado L_D) should be kind 'formula'
        formula = [e for e in kb.query(metric="L_D") if "formula" in e.notes.lower()
                   or "=" in e.notes]
        assert any(e.kind == "formula" for e in formula)


class TestDefectLadder:
    def test_ladder_sorted_and_populated(self):
        kb = K.active()
        ladder = kb.defect_type_ladder()
        assert len(ladder) >= 3
        vals = [v for v, _, _ in ladder]
        assert vals == sorted(vals)

    def test_sp3_is_highest_eckmann(self):
        # Eckmann: sp3 defects give the highest I_D/I_D' (~13)
        kb = K.active()
        ladder = kb.defect_type_ladder()
        top_types = [dt for _, dt, _ in ladder[-3:]]
        assert any("sp3" in t for t in top_types)


class TestReferenceRange:
    def test_range_for_LD(self):
        kb = K.active()
        lo, hi, ms = kb.reference_range("L_D")
        assert lo is not None and hi is not None and lo <= hi
        assert len(ms) > 0

    def test_empty_for_unknown_metric(self):
        kb = K.active()
        lo, hi, ms = kb.reference_range("totally_unknown_metric_xyz")
        assert lo is None and hi is None and ms == []


class TestDescribe:
    def test_describe_has_citation_and_conditions(self):
        kb = K.active()
        e = kb.query(metric="I_D/I_D'")[0]
        s = e.describe()
        assert "[" in s and "]" in s  # citation block present

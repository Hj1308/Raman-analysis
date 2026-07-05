# Validation

Raw spectra from the literature cannot be redistributed, so this folder
validates the tool in three honest, clearly-separated tiers plus an
end-to-end script for your own data.

## Tier A — Independent-pair agreement (`validate_formulas.py`)

Only cases where a paper publishes **both** the measured I_D/I_G **and** the
derived length count as real validation: we feed the published ratio into the
same equation the tool uses and require agreement with the published length
(±10 %). Currently: Dierke et al. 2022 (532 nm), where both quantities are
reported for the same spots on SiO₂ and hBN. For the hBN linescan the paper
reports a ratio *interval* (0.51 ± 0.16), so the test is range-aware.

**What we deliberately do NOT do:** back-calculate I_D/I_G from a published
L_a using the same formula and then "validate" the formula against it — that
is circular and always passes by construction. Papers whose lengths come from
combined/other defect models (e.g. Anusuya 2022) are therefore not used as
agreement tests; their values live in the knowledge base as reference context
instead.

## Tier B — Negative control (regime limits)

The Cançado low-defect relation is valid only for L_D ≳ 10 nm (Stage 1).
Kim et al. 2012 report L_D ≈ 4.76 nm (from the full Lucchese curve) — outside
the regime — so our simple formula **must** disagree substantially, and the
test asserts that it does (observed ≈ 36 %). This demonstrates the regime
warnings issued by `src/validation.py` are scientifically necessary.

## Tier C — Implementation self-consistency

Round-trip and unit-conversion checks, honestly labelled as such: they verify
the equations are *coded* correctly (invertibility, E_L⁴ scaling,
λ→eV conversion), not that they agree with any experiment.

## Running

```bash
pytest validation/validate_formulas.py -v   # all three tiers
python validation/validate_formulas.py      # human-readable table
```

## Your own spectrum (`run_on_your_spectrum.py`)

```bash
python validation/run_on_your_spectrum.py --file my.txt --laser 532 --material rGO
```

Runs the full pipeline (despike → baseline → global fit → analysis →
validation flags) and prints every derived quantity next to literature
reference ranges pulled live from the tool's knowledge base
(`src/knowledge_base.json`, 113 cited entries), so you can judge whether the
results are physically reasonable for your material.

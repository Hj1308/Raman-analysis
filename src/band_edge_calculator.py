"""
Band Edge Calculator — Butler-Ginley formalism
==============================================
Calculates conduction band (CB) and valence band (VB) edge potentials
vs. NHE using the Butler-Ginley electronegativity method.

Reference:
    Butler & Ginley, J. Electrochem. Soc. 125, 228 (1978)
    DOI: 10.1149/1.2131419

Formulae
--------
    E_CB(NHE) = χ - E°(H⁺/H₂) - 0.5 · Eg
    E_VB(NHE) = E_CB(NHE) + Eg

Constants
---------
    _NHE_VS_VAC = 4.44 eV   (NHE vs. vacuum; IUPAC recommended)
"""

from __future__ import annotations

__all__ = ["BandEdgeCalculator", "calculate_band_edges"]

# IUPAC recommended value: NHE vs. vacuum (eV)
_NHE_VS_VAC: float = 4.44

# Common semiconductor electronegativities (Mulliken, eV) — lookup table
_CHI_DB: dict[str, float] = {
    "TiO2": 5.81,
    "g-C3N4": 4.73,
    "ZnO": 5.79,
    "BiVO4": 6.04,
    "Fe2O3": 5.88,
    "WO3": 6.59,
    "CdS": 4.88,
    "ZnS": 5.26,
    "In2O3": 5.79,
    "SnO2": 6.25,
}


class BandEdgeCalculator:
    """Calculate semiconductor band edge positions vs. NHE.

    Parameters
    ----------
    chi : float
        Geometric mean of atomic electronegativities (Mulliken scale, eV).
    Eg : float
        Optical band gap energy (eV). Must be positive.

    Raises
    ------
    ValueError
        If Eg <= 0.
    """

    def __init__(self, chi: float, Eg: float) -> None:
        if Eg <= 0:
            raise ValueError(f"Band gap Eg must be positive, got {Eg}")
        self.chi = float(chi)
        self.Eg = float(Eg)

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    @property
    def Ecb_NHE(self) -> float:
        """Conduction band minimum vs. NHE (eV).

        E_CB(NHE) = χ - 4.44 - 0.5·Eg
        """
        return self.chi - _NHE_VS_VAC - 0.5 * self.Eg

    @property
    def Evb_NHE(self) -> float:
        """Valence band maximum vs. NHE (eV).

        E_VB(NHE) = E_CB(NHE) + Eg
        """
        return self.Ecb_NHE + self.Eg

    @property
    def Ecb_vac(self) -> float:
        """Conduction band minimum vs. vacuum (eV).

        E_CB(vac) = χ - 0.5·Eg
        Note: no extra offset; vacuum reference is the zero of energy.
        """
        return self.chi - 0.5 * self.Eg

    @property
    def Evb_vac(self) -> float:
        """Valence band maximum vs. vacuum (eV)."""
        return self.Ecb_vac + self.Eg

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def overpotential_HER(self) -> float:
        """Thermodynamic driving force for H₂ evolution (eV).

        Positive value means CB is above H⁺/H₂ level — HER is feasible.
        """
        return -self.Ecb_NHE  # E_CB(NHE) < 0 → positive overpotential

    @property
    def overpotential_OER(self) -> float:
        """Thermodynamic driving force for O₂ evolution (eV).

        Positive value means VB is below H₂O/O₂ level (1.23 V) — OER feasible.
        """
        return self.Evb_NHE - 1.23

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, float]:
        """Return all band edge values as a dictionary."""
        return {
            "chi_eV": self.chi,
            "Eg_eV": self.Eg,
            "Ecb_NHE_eV": round(self.Ecb_NHE, 4),
            "Evb_NHE_eV": round(self.Evb_NHE, 4),
            "Ecb_vac_eV": round(self.Ecb_vac, 4),
            "Evb_vac_eV": round(self.Evb_vac, 4),
            "overpotential_HER_eV": round(self.overpotential_HER, 4),
            "overpotential_OER_eV": round(self.overpotential_OER, 4),
        }

    def __repr__(self) -> str:
        return (
            f"BandEdgeCalculator(chi={self.chi} eV, Eg={self.Eg} eV)\n"
            f"  E_CB(NHE) = {self.Ecb_NHE:.4f} eV\n"
            f"  E_VB(NHE) = {self.Evb_NHE:.4f} eV"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def calculate_band_edges(
    chi: float | None = None,
    Eg: float | None = None,
    material: str | None = None,
) -> dict[str, float]:
    """Convenience wrapper — calculate band edges from chi + Eg or material name.

    Parameters
    ----------
    chi : float, optional
        Electronegativity (eV). Required if *material* is not given.
    Eg : float
        Band gap (eV). Always required.
    material : str, optional
        Lookup chi from built-in database (e.g. 'TiO2', 'g-C3N4').

    Returns
    -------
    dict
        Band edge summary dictionary.

    Examples
    --------
    >>> calculate_band_edges(chi=4.73, Eg=2.7)
    >>> calculate_band_edges(material='TiO2', Eg=3.2)
    """
    if Eg is None:
        raise ValueError("Eg (band gap) is required.")
    if material is not None:
        if material not in _CHI_DB:
            raise KeyError(
                f"Material '{material}' not in database. "
                f"Available: {list(_CHI_DB.keys())}"
            )
        chi = _CHI_DB[material]
    if chi is None:
        raise ValueError("Provide either chi or a known material name.")
    return BandEdgeCalculator(chi=chi, Eg=Eg).summary()

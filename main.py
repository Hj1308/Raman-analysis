"""
Raman Spectrum Analyzer — CLI Entry Point
Author: Hoda Jaafari
Usage:
    python main.py --file spectrum.txt --laser 532 --output results/
    python main.py --folder ./spectra/ --laser 633 --batch
    python main.py --file spectrum.txt --laser 532 --substrate SiC
    python main.py --folder ./spectra/ --laser 532 --batch --substrate g-NBSiC
"""

import argparse
import sys
from pathlib import Path
from src.loader      import load_spectrum, load_batch
from src.baseline    import correct_baseline
from src.peak_fitter import fit_all_peaks
from src.analyzer    import analyze, format_report
from src.plotter     import (plot_full_spectrum, plot_fitted_peaks,
                              plot_individual_peaks, plot_ratios)
from src.exporter    import append_csv, save_text_report


def process_one(filepath: str, laser_nm: float, output_dir: str,
                substrate: str = "unknown"):
    filename = Path(filepath).name
    print(f"\n{'\u2550'*50}")
    print(f"  Processing: {filename}")
    print(f"  Laser: {laser_nm} nm")
    print(f"  Substrate: {substrate}")
    print(f"{'\u2550'*50}")

    # 1. Load
    wavenumber, intensity = load_spectrum(filepath)
    print(f"  Loaded {len(wavenumber)} data points  "
          f"[{wavenumber[0]:.0f} \u2013 {wavenumber[-1]:.0f} cm\u207b\u00b9]")

    # 2. Baseline correction
    corrected, baseline = correct_baseline(wavenumber, intensity, method="als")
    print("  Baseline correction (ALS): done")

    # 3. Peak fitting
    peaks = fit_all_peaks(wavenumber, corrected, laser_nm=laser_nm)
    for key, p in peaks.items():
        status = f"{p.center:.1f} cm\u207b\u00b9  FWHM={p.fwhm:.1f}  R\u00b2={p.r_squared:.3f}" if p.found else "not detected"
        print(f"  {p.name:<6}: {status}")

    # 4. Analysis  — substrate passed to suppress invalid doping labels
    #    (Fix 1.3: SiC / g-NBSiC suppress G-shift doping; other non-graphene
    #    substrates suppress I2D/IG n/p classification [Faugeras 2008])
    analysis = analyze(peaks, laser_nm=laser_nm, substrate=substrate)
    print(f"\n  ID/IG   = {analysis.ID_IG_height:.4f}" if analysis.D_found and analysis.G_found else "  ID/IG   = N/A")
    print(f"  I2D/IG  = {analysis.I2D_IG_height:.4f}" if analysis.twoD_found and analysis.G_found else "  I2D/IG  = N/A")
    if analysis.L_D_nm and analysis.L_D_nm == analysis.L_D_nm:
        print(f"  L_D     = {analysis.L_D_nm:.2f} nm")
    print(f"  Layers  : {analysis.estimated_layers}")
    print(f"  Stage   : {analysis.disorder_stage}")
    print(f"  Doping  : {analysis.doping_type}")
    if analysis.doping_note:
        print(f"  Doping note: {analysis.doping_note}")

    # 5. Plots
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    p1 = plot_full_spectrum(wavenumber, intensity, baseline, corrected, filename, output_dir)
    p2 = plot_fitted_peaks(wavenumber, corrected, peaks, filename, output_dir)
    p3 = plot_individual_peaks(wavenumber, corrected, peaks, filename, output_dir)
    p4 = plot_ratios(analysis, filename, output_dir)
    print(f"\n  Plots saved to: {output_dir}")

    # 6. Export
    csv_path    = str(Path(output_dir) / "raman_results.csv")
    report_path = str(Path(output_dir) / f"{Path(filename).stem}_report.txt")
    append_csv(csv_path, filename, laser_nm, peaks, analysis)
    report_text = format_report(filename, peaks, analysis, laser_nm)
    save_text_report(report_path, report_text)
    print(f"  Report : {report_path}")
    print(f"  CSV    : {csv_path}")
    print(report_text)


def main():
    parser = argparse.ArgumentParser(
        description="Raman Spectrum Analyzer for graphene and graphene-like materials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file spectrum.txt --laser 532
  python main.py --folder ./spectra/ --laser 633 --batch
  python main.py --file sample.csv --laser 785 --output ./my_results/
  python main.py --file sic_sample.txt --laser 532 --substrate SiC
  python main.py --folder ./spectra/ --laser 532 --batch --substrate g-NBSiC
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",   type=str, help="Single spectrum file (.txt or .csv)")
    group.add_argument("--folder", type=str, help="Folder with multiple spectrum files (batch mode)")

    parser.add_argument("--laser",  type=float, default=532.0,
                        help="Laser excitation wavelength in nm (default: 532)")
    parser.add_argument("--output", type=str,   default="results",
                        help="Output directory for plots and reports (default: results/)")
    parser.add_argument("--batch",  action="store_true",
                        help="Enable batch mode (required with --folder)")
    parser.add_argument(
        "--substrate", type=str, default="unknown",
        help=(
            "Substrate on which the carbon layer is deposited. "
            "Controls doping classification: SiC and g-NBSiC suppress G-shift "
            "estimation (Fuchs-Kliewer phonon overlap); hBN, SiO2, Cu, Ni etc. "
            "suppress I2D/IG n/p labelling. "
            "Examples: SiC, g-NBSiC, hBN, SiO2, Cu, unknown (default)."
        ),
    )

    args = parser.parse_args()

    if args.file:
        process_one(args.file, args.laser, args.output,
                    substrate=args.substrate)

    elif args.folder:
        print(f"\nBatch mode \u2014 scanning: {args.folder}")
        spectra = load_batch(args.folder)
        print(f"Found {len(spectra)} spectrum file(s)\n")
        for sp in spectra:
            try:
                process_one(sp["filepath"], args.laser, args.output,
                            substrate=args.substrate)
            except Exception as e:
                print(f"  ERROR processing {sp['filename']}: {e}")
        print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()

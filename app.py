"""
Raman Analyzer — Graphical User Interface
Author: Hoda Jaafari
Run: python app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os, sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from src.loader      import load_spectrum
from src.baseline    import correct_baseline
from src.peak_fitter import fit_all_peaks
from src.analyzer    import analyze, format_report
from src.exporter    import append_csv, save_text_report

# ── Colour palette ────────────────────────────────
BG        = "#1e1e2e"
SURFACE   = "#2a2a3e"
SURFACE2  = "#313145"
ACCENT    = "#4fc3f7"
ACCENT2   = "#81d4fa"
GREEN     = "#69db7c"
ORANGE    = "#ffa94d"
RED       = "#ff6b6b"
TEXT      = "#cdd6f4"
MUTED     = "#7f849c"
BORDER    = "#45475a"

PEAK_COLORS = {
    "D": "#ff6b6b", "G": "#69db7c", "D_prime": "#ffa94d",
    "2D": "#4fc3f7", "DG": "#cc99ff"
}

class RamanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Raman Spectrum Analyzer  •  Graphene & sp² Carbon Materials")
        self.geometry("1280x820")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        # State
        self.filepath   = tk.StringVar()
        self.laser_nm   = tk.DoubleVar(value=532.0)
        self.baseline_m = tk.StringVar(value="als")
        self.als_lam    = tk.DoubleVar(value=1e5)
        self.als_p      = tk.DoubleVar(value=0.001)
        self.output_dir = tk.StringVar(value=os.path.join(os.path.dirname(__file__), "results"))
        self.strict_mode = tk.BooleanVar(value=True)

        self._wn = None; self._intensity = None
        self._corrected = None; self._baseline = None
        self._peaks = None; self._analysis = None

        self._build_ui()
        self._apply_styles()

    # ─────────────────────────────────────────────
    #  UI BUILD
    # ─────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──────────────────────────────
        topbar = tk.Frame(self, bg=SURFACE, height=52)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="🔬  Raman Spectrum Analyzer",
                 bg=SURFACE, fg=ACCENT,
                 font=("Helvetica", 16, "bold")).pack(side="left", padx=20, pady=12)
        tk.Label(topbar, text="Graphene / sp² Carbon Materials",
                 bg=SURFACE, fg=MUTED,
                 font=("Helvetica", 10)).pack(side="left", pady=12)

        # ── Main layout ───────────────────────────
        main = tk.PanedWindow(self, orient="horizontal",
                              bg=BG, sashwidth=6, sashpad=0, relief="flat")
        main.pack(fill="both", expand=True, padx=8, pady=6)

        # Left panel
        left = tk.Frame(main, bg=BG, width=300)
        main.add(left, minsize=260)
        self._build_left(left)

        # Right panel (notebook)
        right = tk.Frame(main, bg=BG)
        main.add(right, minsize=700)
        self._build_right(right)

        # ── Status bar ────────────────────────────
        self.status_var = tk.StringVar(value="Ready — load a spectrum file to begin")
        statusbar = tk.Frame(self, bg=SURFACE2, height=26)
        statusbar.pack(fill="x", side="bottom")
        statusbar.pack_propagate(False)
        tk.Label(statusbar, textvariable=self.status_var,
                 bg=SURFACE2, fg=MUTED,
                 font=("Helvetica", 9)).pack(side="left", padx=12, pady=4)

    def _build_left(self, parent):
        def section(title):
            f = tk.LabelFrame(parent, text=f"  {title}  ",
                              bg=BG, fg=ACCENT2,
                              font=("Helvetica", 10, "bold"),
                              bd=1, relief="groove",
                              labelanchor="nw")
            f.pack(fill="x", padx=8, pady=(6,2))
            return f

        # ── File ─────────────────────────────────
        fs = section("📂  Input File")
        tk.Button(fs, text="Browse…", command=self._browse_file,
                  bg=ACCENT, fg="#000", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=12, cursor="hand2").pack(fill="x", padx=8, pady=(6,2))
        self._file_label = tk.Label(fs, text="No file selected",
                                    bg=BG, fg=MUTED,
                                    font=("Helvetica", 9),
                                    wraplength=240, justify="left")
        self._file_label.pack(padx=8, pady=(0,6), anchor="w")

        # ── Laser ─────────────────────────────────
        ls = section("🔴  Laser Wavelength")
        laser_frame = tk.Frame(ls, bg=BG)
        laser_frame.pack(fill="x", padx=8, pady=6)
        tk.Label(laser_frame, text="Wavelength (nm):",
                 bg=BG, fg=TEXT, font=("Helvetica", 10)).grid(row=0, column=0, sticky="w")
        laser_entry = tk.Entry(laser_frame, textvariable=self.laser_nm,
                               bg=SURFACE2, fg=ACCENT,
                               font=("Helvetica", 12, "bold"),
                               insertbackground=ACCENT,
                               relief="flat", width=8, justify="center")
        laser_entry.grid(row=0, column=1, padx=(8,0), sticky="w")

        preset_frame = tk.Frame(ls, bg=BG)
        preset_frame.pack(fill="x", padx=8, pady=(0,6))
        tk.Label(preset_frame, text="Presets:", bg=BG, fg=MUTED,
                 font=("Helvetica", 9)).pack(side="left")
        for nm, color in [("488", "#4fc3f7"), ("532", "#69db7c"),
                           ("633", "#ffa94d"), ("785", "#ff6b6b")]:
            tk.Button(preset_frame, text=f"{nm} nm",
                      bg=SURFACE2, fg=color,
                      font=("Helvetica", 9, "bold"),
                      relief="flat", padx=6, cursor="hand2",
                      command=lambda v=nm: self.laser_nm.set(float(v))
                      ).pack(side="left", padx=2)

        # ── Baseline ──────────────────────────────
        bs = section("📉  Baseline Correction")
        bf = tk.Frame(bs, bg=BG); bf.pack(fill="x", padx=8, pady=6)
        tk.Label(bf, text="Method:", bg=BG, fg=TEXT,
                 font=("Helvetica", 10)).grid(row=0, column=0, sticky="w")
        ttk.Combobox(bf, textvariable=self.baseline_m,
                     values=["als", "linear"],
                     state="readonly", width=10
                     ).grid(row=0, column=1, padx=(8,0), sticky="w")
        tk.Label(bf, text="ALS λ (smoothness):", bg=BG, fg=TEXT,
                 font=("Helvetica", 9)).grid(row=1, column=0, sticky="w", pady=(4,0))
        tk.Entry(bf, textvariable=self.als_lam,
                 bg=SURFACE2, fg=TEXT, relief="flat",
                 font=("Helvetica", 9), width=12
                 ).grid(row=1, column=1, padx=(8,0), sticky="w", pady=(4,0))
        tk.Label(bf, text="ALS p (asymmetry):", bg=BG, fg=TEXT,
                 font=("Helvetica", 9)).grid(row=2, column=0, sticky="w", pady=(4,0))
        tk.Entry(bf, textvariable=self.als_p,
                 bg=SURFACE2, fg=TEXT, relief="flat",
                 font=("Helvetica", 9), width=12
                 ).grid(row=2, column=1, padx=(8,0), sticky="w", pady=(4,0))

        # ── Output ────────────────────────────────
        os_frame = section("💾  Output Directory")
        tk.Button(os_frame, text="Choose folder…",
                  command=self._browse_output,
                  bg=SURFACE2, fg=TEXT,
                  font=("Helvetica", 9), relief="flat",
                  padx=8, cursor="hand2").pack(fill="x", padx=8, pady=(6,2))
        self._out_label = tk.Label(os_frame,
                                   text=self.output_dir.get(),
                                   bg=BG, fg=MUTED,
                                   font=("Helvetica", 8),
                                   wraplength=240, justify="left")
        self._out_label.pack(padx=8, pady=(0,6), anchor="w")

        # ── Run button ────────────────────────────
        tk.Frame(parent, bg=BG, height=8).pack()
        self._run_btn = tk.Button(parent,
                                  text="▶   RUN ANALYSIS",
                                  command=self._run,
                                  bg=GREEN, fg="#000",
                                  font=("Helvetica", 13, "bold"),
                                  relief="flat", padx=16, pady=12,
                                  cursor="hand2", activebackground="#51cf66")
        self._run_btn.pack(fill="x", padx=8)

        self._export_btn = tk.Button(parent,
                                     text="📊  Export Excel",
                                     command=self._export_excel,
                                     bg=SURFACE2, fg=ACCENT2,
                                     font=("Helvetica", 11, "bold"),
                                     relief="flat", padx=12, pady=8,
                                     cursor="hand2", state="disabled")
        self._export_btn.pack(fill="x", padx=8, pady=(6,0))

    def _build_right(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        self._nb = nb

        # Tab 1: Spectrum
        tab1 = tk.Frame(nb, bg=BG)
        nb.add(tab1, text="  📈 Spectrum  ")
        self._fig1 = Figure(figsize=(9,5), facecolor="#1a1a2e")
        self._ax1a = self._fig1.add_subplot(211)
        self._ax1b = self._fig1.add_subplot(212)
        canvas1 = FigureCanvasTkAgg(self._fig1, tab1)
        canvas1.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas1, tab1).pack(fill="x")
        self._canvas1 = canvas1

        # Tab 2: Peak Fits
        tab2 = tk.Frame(nb, bg=BG)
        nb.add(tab2, text="  🔍 Peak Fits  ")
        self._fig2 = Figure(figsize=(9,5), facecolor="#1a1a2e")
        canvas2 = FigureCanvasTkAgg(self._fig2, tab2)
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas2, tab2).pack(fill="x")
        self._canvas2 = canvas2

        # Tab 3: Results table
        tab3 = tk.Frame(nb, bg=BG)
        nb.add(tab3, text="  📋 Results  ")
        self._build_results_tab(tab3)

        # Tab 4: Full report
        tab4 = tk.Frame(nb, bg=BG)
        nb.add(tab4, text="  📄 Report  ")
        self._report_text = scrolledtext.ScrolledText(
            tab4, bg="#0d1117", fg="#c9d1d9",
            font=("Courier", 10), relief="flat",
            insertbackground=ACCENT, state="disabled")
        self._report_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_results_tab(self, parent):
        # Ratios frame
        rf = tk.LabelFrame(parent, text="  Intensity Ratios  ",
                           bg=BG, fg=ACCENT2,
                           font=("Helvetica", 10, "bold"), bd=1)
        rf.pack(fill="x", padx=10, pady=8)

        ratio_headers = ["Ratio", "Height-based", "Area-based"]
        for ci, h in enumerate(ratio_headers):
            tk.Label(rf, text=h, bg=SURFACE, fg=ACCENT,
                     font=("Helvetica", 10, "bold"),
                     width=20, relief="flat", pady=4
                     ).grid(row=0, column=ci, padx=1, pady=1, sticky="ew")

        self._ratio_vars = {}
        for ri, lbl in enumerate(["ID/IG", "I2D/IG", "ID'/IG", "ID/ID'"]):
            tk.Label(rf, text=lbl, bg=SURFACE2, fg=TEXT,
                     font=("Helvetica", 11, "bold"),
                     width=20, pady=4).grid(row=ri+1, column=0, padx=1, pady=1, sticky="ew")
            h_var = tk.StringVar(value="—")
            a_var = tk.StringVar(value="—")
            self._ratio_vars[lbl] = (h_var, a_var)
            tk.Label(rf, textvariable=h_var, bg=SURFACE2, fg=GREEN,
                     font=("Courier", 11, "bold"),
                     width=20, pady=4).grid(row=ri+1, column=1, padx=1, pady=1, sticky="ew")
            tk.Label(rf, textvariable=a_var, bg=SURFACE2, fg=ACCENT2,
                     font=("Courier", 11, "bold"),
                     width=20, pady=4).grid(row=ri+1, column=2, padx=1, pady=1, sticky="ew")

        # Peaks table
        pf = tk.LabelFrame(parent, text="  Fitted Peaks  ",
                           bg=BG, fg=ACCENT2,
                           font=("Helvetica", 10, "bold"), bd=1)
        pf.pack(fill="x", padx=10, pady=(0,8))

        pk_headers = ["Peak", "Center (cm⁻¹)", "FWHM (cm⁻¹)", "Height", "Area", "R²", "Status"]
        for ci, h in enumerate(pk_headers):
            tk.Label(pf, text=h, bg=SURFACE, fg=ACCENT,
                     font=("Helvetica", 9, "bold"),
                     pady=4, relief="flat"
                     ).grid(row=0, column=ci, padx=1, pady=1, sticky="ew")

        self._peak_vars = {}
        for ri, name in enumerate(["D", "G", "D'", "2D", "D+G"]):
            color = list(PEAK_COLORS.values())[ri]
            tk.Label(pf, text=name, bg=SURFACE2, fg=color,
                     font=("Helvetica", 11, "bold"), pady=3
                     ).grid(row=ri+1, column=0, padx=1, pady=1, sticky="ew")
            row_vars = []
            for ci in range(1, 7):
                v = tk.StringVar(value="—")
                row_vars.append(v)
                fg_color = TEXT if ci < 6 else MUTED
                tk.Label(pf, textvariable=v, bg=SURFACE2, fg=fg_color,
                         font=("Courier", 10), pady=3
                         ).grid(row=ri+1, column=ci, padx=1, pady=1, sticky="ew")
            self._peak_vars[name] = row_vars

        # Structural analysis
        sf = tk.LabelFrame(parent, text="  Structural Analysis  ",
                           bg=BG, fg=ACCENT2,
                           font=("Helvetica", 10, "bold"), bd=1)
        sf.pack(fill="x", padx=10, pady=(0,8))
        self._struct_vars = {}
        struct_labels = ["L_D (nm)", "Disorder Stage", "Defect Type", "Estimated Layers"]
        for ri, lbl in enumerate(struct_labels):
            tk.Label(sf, text=lbl, bg=SURFACE2, fg=TEXT,
                     font=("Helvetica", 10, "bold"),
                     width=22, pady=4, anchor="w"
                     ).grid(row=ri, column=0, padx=(8,1), pady=1, sticky="ew")
            v = tk.StringVar(value="—")
            self._struct_vars[lbl] = v
            tk.Label(sf, textvariable=v, bg=SURFACE2, fg=ORANGE,
                     font=("Helvetica", 10), pady=4, anchor="w"
                     ).grid(row=ri, column=1, padx=(1,8), pady=1, sticky="ew")

    # ─────────────────────────────────────────────
    #  ACTIONS
    # ─────────────────────────────────────────────
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Raman spectrum file",
            filetypes=[("Text/CSV files", "*.txt *.csv"), ("All files", "*.*")]
        )
        if path:
            self.filepath.set(path)
            self._file_label.config(text=os.path.basename(path), fg=ACCENT2)
            self._status(f"File loaded: {os.path.basename(path)}")

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.output_dir.set(d)
            self._out_label.config(text=d)

    def _status(self, msg, color=None):
        self.status_var.set(msg)
        if color:
            pass

    def _run(self):
        if not self.filepath.get():
            messagebox.showwarning("No file", "Please select a Raman spectrum file first.")
            return
        self._run_btn.config(state="disabled", text="⏳  Analysing…", bg=MUTED)
        self._status("Running analysis…")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            # 1. Load
            wn, intensity = load_spectrum(self.filepath.get())
            self._wn = wn; self._intensity = intensity
            self._status(f"Loaded {len(wn)} points | Running baseline correction…")

            # 2. Baseline
            lam = float(self.als_lam.get())
            p   = float(self.als_p.get())
            corrected, baseline = correct_baseline(
                wn, intensity, method=self.baseline_m.get(), lam=lam, p=p)
            self._corrected = corrected
            self._baseline  = baseline
            self._status("Baseline done | Fitting peaks…")

            # 3. Peaks
            laser = float(self.laser_nm.get())
            peaks = fit_all_peaks(wn, corrected, laser_nm=laser)
            self._peaks = peaks

            # 4. Analysis
            analysis = analyze(peaks, laser_nm=laser)
            self._analysis = analysis
            self._status("Peak fitting done | Rendering plots…")

            # Update UI on main thread
            self.after(0, lambda: self._update_ui(wn, intensity, baseline,
                                                   corrected, peaks, analysis, laser))
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _update_ui(self, wn, intensity, baseline, corrected, peaks, analysis, laser):
        fname = os.path.basename(self.filepath.get())

        # ── Plot 1: Spectrum ──────────────────────
        self._fig1.clf()
        ax1 = self._fig1.add_subplot(211)
        ax2 = self._fig1.add_subplot(212)

        for ax in [ax1, ax2]:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.spines['bottom'].set_color(BORDER)
            ax.spines['left'].set_color(BORDER)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax1.plot(wn, intensity,  color="#4fc3f7", lw=1.0, label="Raw spectrum", alpha=0.9)
        ax1.plot(wn, baseline,   color="#ff6b6b", lw=1.5, ls="--", label="ALS baseline", alpha=0.85)
        ax1.fill_between(wn, baseline, intensity, alpha=0.08, color="#4fc3f7")
        ax1.set_ylabel("Intensity (a.u.)", color=TEXT, fontsize=9)
        ax1.set_title(f"{fname}  |  λ = {laser:.0f} nm", color=ACCENT2, fontsize=10, pad=6)
        ax1.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

        ax2.plot(wn, corrected, color="#69db7c", lw=1.2, label="Corrected", alpha=0.9)
        ax2.axhline(0, color=BORDER, lw=0.8, ls="--")
        for key, p in peaks.items():
            if p.found:
                ax2.axvline(p.center, color=PEAK_COLORS.get(key,"gray"),
                            lw=0.8, ls=":", alpha=0.7)
                ax2.text(p.center, corrected.max()*0.92,
                         p.name, color=PEAK_COLORS.get(key,"gray"),
                         fontsize=8, ha="center", fontweight="bold")
        ax2.set_xlabel("Raman Shift (cm⁻¹)", color=TEXT, fontsize=9)
        ax2.set_ylabel("Intensity (a.u.)", color=TEXT, fontsize=9)
        ax2.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

        self._fig1.tight_layout(pad=1.5)
        self._canvas1.draw()

        # ── Plot 2: Individual peaks ──────────────
        self._fig2.clf()
        found_peaks = [(k,p) for k,p in peaks.items() if p.found and len(p.model_x)>0]
        n = len(found_peaks)
        if n > 0:
            for i, (key, p) in enumerate(found_peaks):
                ax = self._fig2.add_subplot(1, n, i+1)
                ax.set_facecolor("#0d1117")
                ax.tick_params(colors=MUTED, labelsize=7)
                for sp in ['top','right']: ax.spines[sp].set_visible(False)
                for sp in ['bottom','left']: ax.spines[sp].set_color(BORDER)

                color = PEAK_COLORS.get(key, "gray")
                mask  = (wn >= p.model_x[0]) & (wn <= p.model_x[-1])
                xd, yd = wn[mask], corrected[mask]
                yfit   = np.interp(xd, p.model_x, p.model_y)

                ax.scatter(xd, yd, s=4, color="#cdd6f4", alpha=0.5, zorder=3)
                ax.plot(p.model_x, p.model_y, color=color, lw=2.0)
                ax.fill_between(p.model_x, p.model_y, alpha=0.3, color=color)
                ax.set_title(
                    f"{p.name}\n{p.center:.1f} cm⁻¹\nFWHM={p.fwhm:.1f}\nR²={p.r_squared:.3f}",
                    color=color, fontsize=8, pad=4)
                ax.set_xlabel("Raman Shift (cm⁻¹)", color=MUTED, fontsize=7)
                if i == 0:
                    ax.set_ylabel("Intensity", color=MUTED, fontsize=7)

        self._fig2.tight_layout(pad=1.2)
        self._canvas2.draw()

        # ── Results tab ───────────────────────────
        import math
        def fmt(v): return f"{v:.4f}" if not math.isnan(v) else "N/A"
        def fmti(v): return f"{v:.2f}"  if not math.isnan(v) else "N/A"

        self._ratio_vars["ID/IG"][0].set(fmt(analysis.ID_IG_height))
        self._ratio_vars["ID/IG"][1].set(fmt(analysis.ID_IG_area))
        self._ratio_vars["I2D/IG"][0].set(fmt(analysis.I2D_IG_height))
        self._ratio_vars["I2D/IG"][1].set(fmt(analysis.I2D_IG_area))
        self._ratio_vars["ID'/IG"][0].set(fmt(analysis.IDp_IG_height))
        self._ratio_vars["ID'/IG"][1].set("—")
        self._ratio_vars["ID/ID'"][0].set(fmt(analysis.ID_IDp_height))
        self._ratio_vars["ID/ID'"][1].set("—")

        peak_map = {"D":"D","G":"G","D'":"D_prime","2D":"2D","D+G":"DG"}
        for name, row_vars in self._peak_vars.items():
            key = peak_map[name]
            p   = peaks.get(key)
            if p and p.found:
                row_vars[0].set(f"{p.center:.1f}")
                row_vars[1].set(f"{p.fwhm:.1f}")
                row_vars[2].set(f"{p.amplitude:.1f}")
                row_vars[3].set(f"{p.area:.1f}")
                row_vars[4].set(f"{p.r_squared:.3f}")
                row_vars[5].set("✓ Detected")
            else:
                for v in row_vars: v.set("—")
                row_vars[5].set("Not found")

        self._struct_vars["L_D (nm)"].set(fmti(analysis.L_D_nm))
        self._struct_vars["Disorder Stage"].set(analysis.disorder_stage)
        self._struct_vars["Defect Type"].set(analysis.defect_type)
        self._struct_vars["Estimated Layers"].set(analysis.estimated_layers)

        # ── Report tab ────────────────────────────
        report = format_report(fname, peaks, analysis, laser)
        self._report_text.config(state="normal")
        self._report_text.delete("1.0", "end")
        self._report_text.insert("1.0", report)
        self._report_text.config(state="disabled")

        # ── Enable export ─────────────────────────
        self._export_btn.config(state="normal", bg=ACCENT, fg="#000")
        self._run_btn.config(state="normal", text="▶   RUN ANALYSIS", bg=GREEN)
        self._nb.select(2)  # jump to Results tab
        self._status(
            f"✓  Done  |  ID/IG = {analysis.ID_IG_height:.4f}  |  "
            f"I2D/IG = {analysis.I2D_IG_height:.4f}  |  "
            f"L_D = {analysis.L_D_nm:.2f} nm  |  {analysis.estimated_layers}")

    def _on_error(self, msg):
        self._run_btn.config(state="normal", text="▶   RUN ANALYSIS", bg=GREEN)
        self._status(f"Error: {msg}")
        messagebox.showerror("Analysis Error", msg)

    def _export_excel(self):
        if self._analysis is None:
            return
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            title="Save Excel report",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="raman_analysis.xlsx"
        )
        if not path:
            return
        try:
            self._status("Exporting Excel…")
            self._do_excel_export(path)
            self._status(f"✓  Excel saved: {path}")
            messagebox.showinfo("Exported", f"Excel file saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _do_excel_export(self, path):
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        from datetime import datetime
        import math

        wb  = Workbook()
        ws  = wb.active
        ws.title = "Raman Analysis"
        ws.sheet_view.showGridLines = False
        ws.column_dimensions['A'].width = 3

        H  = Font(name='Calibri', bold=True, color='FFFFFF', size=11)
        HF = PatternFill('solid', fgColor='1F4E79')
        SF = PatternFill('solid', fgColor='2E75B6')
        N  = Font(name='Calibri', size=11)
        B  = Font(name='Calibri', bold=True, size=11, color='1F4E79')
        C  = Alignment(horizontal='center', vertical='center')
        L  = Alignment(horizontal='left',   vertical='center', indent=1)
        R  = Alignment(horizontal='right',  vertical='center')
        fname  = os.path.basename(self.filepath.get())
        laser  = float(self.laser_nm.get())

        def brd():
            s = Side(style='thin', color='BDD7EE')
            return Border(left=s, right=s, top=s, bottom=s)

        # Title
        ws.merge_cells('B2:H2')
        ws['B2'] = 'Raman Spectroscopy Analysis Report'
        ws['B2'].font = Font(name='Calibri', bold=True, color='1F4E79', size=16)
        ws['B2'].alignment = C
        ws.row_dimensions[2].height = 32
        ws.merge_cells('B3:H3')
        ws['B3'] = f"File: {fname}  |  Laser: {laser:.0f} nm  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ws['B3'].font = Font(name='Calibri', size=10, color='7F7F7F', italic=True)
        ws['B3'].alignment = C
        ws.row_dimensions[4].height = 8

        # Intensity Ratios
        ws.merge_cells('B5:G5')
        ws['B5'] = 'Intensity Ratios'; ws['B5'].font = Font(name='Calibri', bold=True, color='1F4E79', size=13)
        ws.row_dimensions[5].height = 22
        for ci, h in enumerate(['Ratio','Height-based','Area-based','Interpretation']):
            c = ws.cell(row=6, column=2+ci, value=h)
            c.font=H; c.fill=HF; c.alignment=C; c.border=brd()
        ws.row_dimensions[6].height = 22

        an = self._analysis
        def fv(v): return round(v,4) if not math.isnan(v) else 'N/A'
        layer_interp = an.estimated_layers
        ratios = [
            ('ID/IG',  fv(an.ID_IG_height),  fv(an.ID_IG_area),
             'Low defect density' if an.ID_IG_height < 0.5 else 'Moderate defects' if an.ID_IG_height < 1.0 else 'High defect density'),
            ('I2D/IG', fv(an.I2D_IG_height), fv(an.I2D_IG_area), layer_interp),
            ("ID'/IG", fv(an.IDp_IG_height), 'N/A', 'Intravalley defect indicator'),
            ("ID/ID'", fv(an.ID_IDp_height), 'N/A', an.defect_type[:40] if an.defect_type != 'N/A' else 'N/A'),
        ]
        for ri,(param,hv,av,interp) in enumerate(ratios):
            r = 7+ri
            f2 = PatternFill('solid', fgColor='D6E4F0' if ri%2==0 else 'EBF3FB')
            for c in range(2,7): ws.cell(row=r,column=c).fill=f2; ws.cell(row=r,column=c).border=brd()
            ws.cell(row=r,column=2,value=param).font=B; ws.cell(row=r,column=2).alignment=L
            ws.cell(row=r,column=3,value=hv).font=N;   ws.cell(row=r,column=3).alignment=C; ws.cell(row=r,column=3).number_format='0.0000'
            ws.cell(row=r,column=4,value=av).font=N;   ws.cell(row=r,column=4).alignment=C; ws.cell(row=r,column=4).number_format='0.0000'
            ws.cell(row=r,column=5,value=interp).font=N; ws.cell(row=r,column=5).alignment=L
            ws.row_dimensions[r].height=20

        ws.row_dimensions[11].height=8

        # Peak parameters
        ws.merge_cells('B12:H12')
        ws['B12']='Fitted Peak Parameters'; ws['B12'].font=Font(name='Calibri',bold=True,color='1F4E79',size=13)
        ws.row_dimensions[12].height=22
        pk_h=['Peak','Center (cm⁻¹)','FWHM (cm⁻¹)','Height (a.u.)','Area (a.u.)','R²','Status']
        for ci,h in enumerate(pk_h):
            c=ws.cell(row=13,column=2+ci,value=h); c.font=H; c.fill=SF; c.alignment=C; c.border=brd()
        ws.row_dimensions[13].height=22
        peaks=self._peaks
        for ri,(name,key) in enumerate([('D','D'),('G','G'),("D'",'D_prime'),('2D','2D'),('D+G','DG')]):
            r=14+ri; p=peaks.get(key)
            f2=PatternFill('solid',fgColor='D6E4F0' if ri%2==0 else 'EBF3FB')
            for c in range(2,9): ws.cell(row=r,column=c).fill=f2; ws.cell(row=r,column=c).border=brd()
            ws.cell(row=r,column=2,value=name).font=B; ws.cell(row=r,column=2).alignment=L
            if p and p.found:
                for ci,v in enumerate([round(p.center,2),round(p.fwhm,2),round(p.amplitude,1),round(p.area,1),round(p.r_squared,4)]):
                    ws.cell(row=r,column=3+ci,value=v).font=N; ws.cell(row=r,column=3+ci).alignment=C
                ws.cell(row=r,column=8,value='Detected ✓').font=Font(name='Calibri',bold=True,size=10,color='375623')
                ws.cell(row=r,column=8).fill=PatternFill('solid',fgColor='E2EFDA')
            else:
                ws.cell(row=r,column=8,value='Not detected').font=Font(name='Calibri',size=10,color='C00000')
                ws.cell(row=r,column=8).fill=PatternFill('solid',fgColor='FCE4D6')
            ws.cell(row=r,column=8).alignment=C; ws.cell(row=r,column=8).border=brd()
            ws.row_dimensions[r].height=20

        ws.row_dimensions[19].height=8

        # Structural
        ws.merge_cells('B20:H20')
        ws['B20']='Structural Analysis'; ws['B20'].font=Font(name='Calibri',bold=True,color='1F4E79',size=13)
        ws.row_dimensions[20].height=22
        struct=[('L_D (nm)', f"{an.L_D_nm:.2f}" if not math.isnan(an.L_D_nm) else 'N/A'),
                ('Disorder Stage',an.disorder_stage),('Defect Type',an.defect_type),
                ('Estimated Layers',an.estimated_layers)]
        for ri,(lbl,val) in enumerate(struct):
            r=21+ri; f2=PatternFill('solid',fgColor='D6E4F0' if ri%2==0 else 'EBF3FB')
            ws.merge_cells(f'D{r}:H{r}')
            for c in [2,3,4]: ws.cell(row=r,column=c).fill=f2; ws.cell(row=r,column=c).border=brd()
            ws.cell(row=r,column=2,value=lbl).font=B; ws.cell(row=r,column=2).alignment=L
            ws.cell(row=r,column=4,value=val).font=N; ws.cell(row=r,column=4).alignment=L
            ws.cell(row=r,column=4).border=brd(); ws.row_dimensions[r].height=20

        # Spectrum Data sheet
        ws2 = wb.create_sheet("Spectrum Data")
        ws2.sheet_view.showGridLines=False
        ws2.column_dimensions['A'].width=3
        ws2.merge_cells('B2:F2')
        ws2['B2']='Baseline-Corrected Spectrum Data'
        ws2['B2'].font=Font(name='Calibri',bold=True,color='1F4E79',size=14)
        ws2['B2'].alignment=C; ws2.row_dimensions[2].height=28
        for ci,h in enumerate(['Wavenumber (cm⁻¹)','Raw Intensity','ALS Baseline','Corrected Intensity','Normalised (0-1)']):
            c=ws2.cell(row=4,column=2+ci,value=h); c.font=H; c.fill=HF; c.alignment=C
        ws2.freeze_panes='B5'
        norm=self._corrected/self._corrected.max()
        for i,(w,r,b,co,n) in enumerate(zip(self._wn,self._intensity,self._baseline,self._corrected,norm)):
            row=5+i
            for ci,v in enumerate([round(float(w),3),round(float(r),3),round(float(b),3),round(float(co),3),round(float(n),5)]):
                ws2.cell(row=row,column=2+ci,value=v).number_format='0.000' if ci<4 else '0.00000'
        for c,w in [(2,22),(3,20),(4,20),(5,24),(6,16)]:
            ws2.column_dimensions[get_column_letter(c)].width=w

        for c,w in [(2,22),(3,16),(4,16),(5,36),(6,14),(7,14),(8,16)]:
            ws.column_dimensions[get_column_letter(c)].width=w

        ws.sheet_properties.tabColor='1F4E79'
        ws2.sheet_properties.tabColor='2E75B6'
        wb.save(path)

    def _apply_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",        background=BG,      borderwidth=0)
        style.configure("TNotebook.Tab",    background=SURFACE, foreground=MUTED,
                        padding=[12,6],     font=("Helvetica",10,"bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", SURFACE2)],
                  foreground=[("selected", ACCENT)])
        style.configure("TCombobox",
                        fieldbackground=SURFACE2, background=SURFACE2,
                        foreground=TEXT, selectbackground=ACCENT)


if __name__ == "__main__":
    app = RamanApp()
    app.mainloop()

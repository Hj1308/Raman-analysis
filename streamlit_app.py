"""
Raman Spectrum Analyzer — Streamlit Web Application
Author: Hoda Jaafari
Run:    streamlit run streamlit_app.py
"""

import io
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from src.loader      import load_spectrum
from src.baseline    import correct_baseline
from src.peak_fitter import fit_all_peaks
from src.analyzer    import analyze, format_report

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Raman Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette (matches Tkinter app) ─────────────
PEAK_COLORS = {
    "D": "#ff6b6b", "G": "#69db7c", "D_prime": "#ffa94d",
    "2D": "#4fc3f7", "DG": "#cc99ff"
}

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Raman Analyzer")
    st.caption("Graphene / sp² Carbon Materials")
    st.divider()

    uploaded = st.file_uploader(
        "Upload spectrum file",
        type=["txt", "csv"],
        help="Two-column file: wavenumber (cm⁻¹), intensity"
    )

    st.subheader("🔴 Laser Wavelength")
    laser_nm = st.number_input("Wavelength (nm)", min_value=400.0,
                                max_value=1100.0, value=532.0, step=1.0)
    cols = st.columns(4)
    for col, nm in zip(cols, [488, 532, 633, 785]):
        if col.button(f"{nm}", use_container_width=True):
            laser_nm = float(nm)

    st.subheader("📉 Baseline Correction")
    baseline_method = st.selectbox("Method", ["als", "linear"])
    if baseline_method == "als":
        als_lam = st.number_input("ALS λ (smoothness)", value=1e5,
                                   min_value=1e2, max_value=1e9,
                                   format="%.0e")
        als_p   = st.number_input("ALS p (asymmetry)",  value=0.001,
                                   min_value=1e-4, max_value=0.5,
                                   format="%.4f")
    else:
        als_lam, als_p = 1e5, 0.001

    st.divider()
    run_btn = st.button("▶ RUN ANALYSIS", type="primary",
                        use_container_width=True, disabled=(uploaded is None))

# ── Main area ─────────────────────────────────────────
st.title("Raman Spectrum Analyzer")
st.caption("Graphene and sp² Carbon Materials | Scientific Edition")

if uploaded is None:
    st.info("👈  Upload a spectrum file in the sidebar to begin.")
    st.stop()

if run_btn or "analysis_done" in st.session_state:
    if run_btn:
        # ── Load ─────────────────────────────────────
        try:
            content = uploaded.read().decode("utf-8", errors="ignore")
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, "w") as f:
                f.write(content)
            wn, intensity = load_spectrum(tmp_path)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()

        # ── Baseline ─────────────────────────────────
        try:
            corrected, baseline = correct_baseline(
                wn, intensity,
                method=baseline_method,
                lam=als_lam, p=als_p
            )
        except Exception as e:
            st.error(f"Baseline correction failed: {e}")
            st.stop()

        # ── Peak fitting ─────────────────────────────
        try:
            peaks    = fit_all_peaks(wn, corrected, laser_nm=laser_nm)
            analysis = analyze(peaks, laser_nm=laser_nm)
        except Exception as e:
            st.error(f"Peak fitting failed: {e}")
            st.stop()

        # Store in session
        st.session_state.update({
            "analysis_done": True,
            "wn": wn, "intensity": intensity,
            "baseline": baseline, "corrected": corrected,
            "peaks": peaks, "analysis": analysis,
            "filename": uploaded.name, "laser_nm": laser_nm,
        })

    # ── Retrieve from session ─────────────────────────
    s        = st.session_state
    wn       = s["wn"]
    intensity= s["intensity"]
    baseline = s["baseline"]
    corrected= s["corrected"]
    peaks    = s["peaks"]
    analysis = s["analysis"]
    fname    = s["filename"]
    laser_nm = s["laser_nm"]

    st.success(
        f"✅  Analysis complete  |  "
        f"ID/IG = {analysis.ID_IG_height:.4f}  |  "
        f"I2D/IG = {analysis.I2D_IG_height:.4f}  |  "
        f"L_D = {analysis.L_D_nm:.2f} nm"
        if not np.isnan(analysis.ID_IG_height) else "✅  Analysis complete"
    )

    # ─────────────────────────────────────────────────
    #  TABS
    # ─────────────────────────────────────────────────
    tab_spec, tab_peaks, tab_results, tab_report = st.tabs(
        ["📈 Spectrum", "🔍 Peak Fits", "📋 Results", "📄 Report"]
    )

    # ── Tab 1: Spectrum ───────────────────────────────
    with tab_spec:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Raw spectrum + ALS baseline",
                            "Baseline-corrected spectrum"),
            vertical_spacing=0.12
        )
        fig.add_trace(go.Scatter(x=wn, y=intensity, name="Raw",
                                  line=dict(color="#4fc3f7", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=wn, y=baseline, name="Baseline",
                                  line=dict(color="#ff6b6b", width=1.5, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=wn, y=corrected, name="Corrected",
                                  line=dict(color="#69db7c", width=1.4),
                                  fill="tozeroy", fillcolor="rgba(105,219,124,0.07)"), row=2, col=1)

        for key, p in peaks.items():
            if p.found:
                color = PEAK_COLORS.get(key, "gray")
                for row in [1, 2]:
                    fig.add_vline(x=p.center, line_width=0.8, line_dash="dot",
                                   line_color=color, row=row, col=1)
                fig.add_annotation(
                    x=p.center, y=corrected.max() * 0.92,
                    text=f"<b>{p.name}</b>",
                    font=dict(color=color, size=11),
                    showarrow=False, row=2, col=1
                )

        fig.update_layout(
            height=600, template="plotly_dark",
            title=dict(text=f"{fname}  |  λ = {laser_nm:.0f} nm", font=dict(size=14)),
            legend=dict(orientation="h", y=1.06),
        )
        fig.update_xaxes(title_text="Raman Shift (cm⁻¹)", row=2, col=1)
        fig.update_yaxes(title_text="Intensity (a.u.)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Peak Fits ──────────────────────────────
    with tab_peaks:
        found_peaks = [(k, p) for k, p in peaks.items() if p.found and len(p.model_x) > 0]
        if not found_peaks:
            st.warning("No peaks detected.")
        else:
            n_cols = min(len(found_peaks), 3)
            cols_list = st.columns(n_cols)
            for i, (key, p) in enumerate(found_peaks):
                color = PEAK_COLORS.get(key, "gray")
                mask  = (wn >= p.model_x[0]) & (wn <= p.model_x[-1])
                xd, yd = wn[mask], corrected[mask]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=xd, y=yd, mode="markers",
                    marker=dict(size=4, color="#cdd6f4", opacity=0.6),
                    name="Data"
                ))
                fig2.add_trace(go.Scatter(
                    x=p.model_x, y=p.model_y,
                    line=dict(color=color, width=2.5),
                    fill="tozeroy", fillcolor=color.replace("#", "rgba(").rstrip(")")+",0.2)",
                    name="Fit"
                ))
                split_note = " [dual-L]" if getattr(p, "is_split_2D", False) else ""
                fig2.update_layout(
                    title=dict(
                        text=(
                            f"<b style='color:{color}'>{p.name}{split_note}</b><br>"
                            f"{p.center:.1f} cm⁻¹ | FWHM={p.fwhm:.1f} | R²={p.r_squared:.3f}"
                        ),
                        font=dict(size=12)
                    ),
                    template="plotly_dark",
                    height=300,
                    showlegend=False,
                    margin=dict(t=80, b=40, l=40, r=10),
                    xaxis_title="Raman Shift (cm⁻¹)",
                    yaxis_title="Intensity",
                )
                cols_list[i % n_cols].plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Results ────────────────────────────────
    with tab_results:
        st.subheader("Intensity Ratios")
        import math
        def _f(v): return f"{v:.4f}" if not math.isnan(v) else "N/A"

        ratio_data = {
            "Ratio":         ["ID/IG",   "I2D/IG",  "ID'/IG",              "ID/ID'"],
            "Height-based":  [_f(analysis.ID_IG_height), _f(analysis.I2D_IG_height),
                              _f(analysis.IDp_IG_height), _f(analysis.ID_IDp_height)],
            "Area-based":    [_f(analysis.ID_IG_area),   _f(analysis.I2D_IG_area),
                              "—",                        "—"],
        }
        st.dataframe(pd.DataFrame(ratio_data), hide_index=True, use_container_width=True)

        st.subheader("Fitted Peak Parameters")
        peak_map = {"D":"D","G":"G","D'":"D_prime","2D":"2D","D+G":"DG"}
        rows = []
        for name, key in peak_map.items():
            p = peaks.get(key)
            if p and p.found:
                split = " ✦" if getattr(p, "is_split_2D", False) else ""
                rows.append({
                    "Peak": name + split,
                    "Center (cm⁻¹)": f"{p.center:.1f}",
                    "FWHM (cm⁻¹)":   f"{p.fwhm:.1f}",
                    "Height (a.u.)": f"{p.amplitude:.1f}",
                    "Area (a.u.)":   f"{p.area:.1f}",
                    "R²":            f"{p.r_squared:.3f}",
                    "Status":        "✅ Detected",
                })
            else:
                rows.append({"Peak": name, "Center (cm⁻¹)": "—",
                             "FWHM (cm⁻¹)": "—", "Height (a.u.)": "—",
                             "Area (a.u.)": "—", "R²": "—", "Status": "❌ Not found"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("✦ = dual-Lorentzian fit used for 2D band (bilayer candidate)")

        st.subheader("Structural Analysis")
        col1, col2 = st.columns(2)
        col1.metric("L_D (nm)",       f"{analysis.L_D_nm:.2f}" if not math.isnan(analysis.L_D_nm) else "N/A")
        col2.metric("Disorder Stage",  analysis.disorder_stage)
        col1.metric("Defect Type",     analysis.defect_type)
        col2.metric("Estimated Layers",analysis.estimated_layers)

    # ── Tab 4: Report ─────────────────────────────────
    with tab_report:
        report_text = format_report(fname, peaks, analysis, laser_nm)
        st.code(report_text, language="", wrap_lines=True)

        # ── Export buttons ────────────────────────────
        st.divider()
        dl1, dl2 = st.columns(2)

        # CSV export
        csv_df = pd.DataFrame({
            "Wavenumber (cm-1)": np.round(wn, 3),
            "Raw Intensity":     np.round(intensity, 3),
            "ALS Baseline":      np.round(baseline, 3),
            "Corrected":         np.round(corrected, 3),
            "Normalised (0-1)": np.round(corrected / corrected.max(), 5)
        })
        dl1.download_button(
            label="⬇️  Download CSV",
            data=csv_df.to_csv(index=False).encode(),
            file_name=f"{os.path.splitext(fname)[0]}_raman.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Text report
        dl2.download_button(
            label="⬇️  Download Report (.txt)",
            data=report_text.encode(),
            file_name=f"{os.path.splitext(fname)[0]}_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

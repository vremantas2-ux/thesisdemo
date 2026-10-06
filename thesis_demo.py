"""
thesis_demo.py - Αυτόνομο MVP: Heatmap Παχομετρήσεων (UT) για υπάρχοντα πλοία.

Εξαρτήσεις: streamlit, pandas, numpy, plotly (τίποτα άλλο).

Ροή:
  1. Ορίζεις το grid (strakes × frame bays).
  2. Συμπληρώνεις ανά strake το απαιτούμενο πάχος κανονισμού t_req (χειροκίνητα, dummy τιμή).
  3. Εισάγεις τις μετρήσεις πάχους UT (mm) ανά κελί.
  4. Βλέπεις το Heatmap του περιθωρίου πάχους πάνω από το t_req (%).

Εκτέλεση:  streamlit run thesis_demo.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Ship Thickness Survey Heatmap (MVP)", page_icon="🚢", layout="wide")

# ----------------------------------------------------------------------------
# Σταθερές
# ----------------------------------------------------------------------------
T_REQ_COL = "Απαιτούμενο Πάχος Κανονισμού (t_req) [mm]"
LOCATIONS = ["bottom", "side", "deck", "bulkhead", "superstructure"]
DEFAULT_T_REQ = 6.0


# ----------------------------------------------------------------------------
# Βοηθητικές συναρτήσεις
# ----------------------------------------------------------------------------
def make_default_strakes(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "strake_name": [f"Strake-{i + 1}" for i in range(n)],
        "location": ["bottom"] * n,
        T_REQ_COL: [DEFAULT_T_REQ] * n,
    })


def resize_strakes(prev: pd.DataFrame | None, n: int) -> pd.DataFrame:
    """Κρατά τα υπάρχοντα strakes όταν αλλάζει ο αριθμός τους (κόβει ή προσθέτει στο τέλος)."""
    if prev is None:
        return make_default_strakes(n)
    prev = prev.reset_index(drop=True)
    if len(prev) >= n:
        return prev.iloc[:n].copy()
    extra = make_default_strakes(n)
    extra["strake_name"] = [f"Strake-{i + 1}" for i in range(n)]
    extra = extra.iloc[len(prev):]
    return pd.concat([prev, extra], ignore_index=True)


def resize_measurements(prev: np.ndarray | None, n_rows: int, n_cols: int) -> np.ndarray:
    """Κρατά τις υπάρχουσες μετρήσεις (κατά θέση) όταν αλλάζει το μέγεθος του grid."""
    out = np.full((n_rows, n_cols), np.nan)
    if prev is not None:
        r, c = min(n_rows, prev.shape[0]), min(n_cols, prev.shape[1])
        out[:r, :c] = prev[:r, :c]
    return out


def synthetic_measurements(t_req: np.ndarray, n_bays: int, seed: int = 42) -> np.ndarray:
    """Συνθετικές παχομετρήσεις για demo: φθορά που αυξάνει προς το κέντρο + θόρυβος + τοπικό pitting."""
    rng = np.random.default_rng(seed)
    n = len(t_req)
    bays = np.linspace(-1, 1, n_bays)
    base_loss = 0.05 + 0.20 * (1 - bays ** 2)                       # 5%..25% απώλεια, μέγιστη στο μέσο
    strake_factor = rng.uniform(0.6, 1.4, size=(n, 1))
    loss = base_loss[None, :] * strake_factor + rng.normal(0, 0.03, size=(n, n_bays))
    meas = (t_req.reshape(-1, 1) * 1.40) * (1 - np.clip(loss, 0, 0.6))  # ονομαστικό ≈ 1.40·t_req
    pits = rng.random((n, n_bays)) < 0.06                           # τοπικό pitting
    meas[pits] *= rng.uniform(0.6, 0.85, size=pits.sum())
    meas[rng.random((n, n_bays)) < 0.05] = np.nan                   # μερικά κελιά χωρίς μέτρηση
    return np.round(meas, 2)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("n_strakes", 5)
ss.setdefault("n_bays", 8)
ss.setdefault("strakes_df", None)
ss.setdefault("meas_arr", None)
ss.setdefault("editor_version", 0)   # αλλάζει όταν αλλάζει η δομή, ώστε να μηδενίζονται οι παλιές επεξεργασίες των editors

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🚢 Heatmap Παχομετρήσεων (UT) — MVP")
st.caption(
    "Χαρτογράφηση φθοράς ελασμάτων από παχομετρήσεις υπηρεσιακής κατάστασης. "
    "Grid ανά strake (σειρά πλάτης) × frame bay (διαμέρισμα μήκους). "
    "Το απαιτούμενο πάχος t_req δίνεται χειροκίνητα ανά strake."
)

with st.sidebar:
    st.header("Σκάφος / Survey")
    vessel_name = st.text_input("Όνομα σκάφους", value="Demo Vessel")
    survey_note = st.text_input("Σημείωση survey", value="")
    st.divider()
    st.caption("Το demo δεν αποθηκεύει δεδομένα σε βάση· εξάγεις τα αποτελέσματα σε CSV (τέλος σελίδας).")

# ---- 1. Grid ---------------------------------------------------------------
st.subheader("1. Ορισμός grid")
gc1, gc2 = st.columns(2)
n_strakes = int(gc1.number_input("Αριθμός strakes (σειρές πλάτης)", min_value=1, max_value=20, step=1, key="n_strakes"))
n_bays = int(gc2.number_input("Αριθμός frame bays (διαμερίσματα μήκους)", min_value=1, max_value=40, step=1, key="n_bays"))

# Προσαρμογή των αποθηκευμένων δεδομένων στο τρέχον μέγεθος grid
prev_strakes = ss["strakes_df"]
if prev_strakes is None or len(prev_strakes) != n_strakes:
    ss["strakes_df"] = resize_strakes(prev_strakes, n_strakes)
    ss["editor_version"] += 1
prev_meas = ss["meas_arr"]
if prev_meas is None or prev_meas.shape != (n_strakes, n_bays):
    ss["meas_arr"] = resize_measurements(prev_meas, n_strakes, n_bays)
    ss["editor_version"] += 1

ver = ss["editor_version"]

# ---- 2. Strakes + t_req ----------------------------------------------------
st.subheader("2. Strakes και απαιτούμενο πάχος κανονισμού")
st.caption(
    f"Πληκτρολόγησε στη στήλη «{T_REQ_COL}» το t_req κάθε strake (dummy τιμή για το demo). "
    "Το όνομα και η θέση (location) είναι περιγραφικά."
)
strakes_df = st.data_editor(
    ss["strakes_df"],
    column_config={
        "strake_name": st.column_config.TextColumn("strake_name", required=True),
        "location": st.column_config.SelectboxColumn("location", options=LOCATIONS, required=True),
        T_REQ_COL: st.column_config.NumberColumn(
            T_REQ_COL, min_value=0.1, max_value=200.0, step=0.1, format="%.2f", required=True),
    },
    num_rows="fixed", hide_index=True, key=f"strakes_editor_{ver}", width="stretch",
)
ss["strakes_df"] = strakes_df

# Επικύρωση
names = [str(n).strip() for n in strakes_df["strake_name"]]
t_req_series = pd.to_numeric(strakes_df[T_REQ_COL], errors="coerce")
problems = []
if any(n == "" or n.lower() == "nan" for n in names):
    problems.append("Υπάρχουν strakes χωρίς όνομα.")
if len(set(names)) != len(names):
    problems.append("Τα ονόματα των strakes πρέπει να είναι μοναδικά.")
if t_req_series.isna().any() or (t_req_series <= 0).any():
    problems.append(f"Η στήλη «{T_REQ_COL}» πρέπει να έχει θετικό αριθμό σε κάθε γραμμή.")
if problems:
    for p in problems:
        st.error("❌ " + p)
    st.stop()

t_req = t_req_series.to_numpy(dtype=float)
bay_cols = [f"FR{j + 1}" for j in range(n_bays)]

# ---- 3. Μετρήσεις ----------------------------------------------------------
st.subheader("3. Μετρήσεις πάχους (UT, mm)")
bc1, bc2, _ = st.columns([1, 1, 3])
if bc1.button("🎲 Συνθετικά δεδομένα (demo)"):
    ss["meas_arr"] = synthetic_measurements(t_req, n_bays)
    ss["editor_version"] += 1
    st.rerun()
if bc2.button("🧹 Καθαρισμός μετρήσεων"):
    ss["meas_arr"] = np.full((n_strakes, n_bays), np.nan)
    ss["editor_version"] += 1
    st.rerun()

meas_input = pd.DataFrame(ss["meas_arr"], index=names, columns=bay_cols)
meas_df = st.data_editor(
    meas_input,
    column_config={c: st.column_config.NumberColumn(c, min_value=0.0, step=0.1, format="%.2f") for c in bay_cols},
    key=f"meas_editor_{ver}", width="stretch",
)
measured = meas_df.to_numpy(dtype=float)
ss["meas_arr"] = measured

# ---- 4. Heatmap ------------------------------------------------------------
st.subheader("4. Heatmap")
t_req_col = t_req.reshape(-1, 1)
margin_pct = (measured - t_req_col) / t_req_col * 100.0

hover_text = [[
    (f"{names[i]} / {bay_cols[j]}<br>Μέτρηση: {measured[i, j]:.2f} mm<br>"
     f"t_req: {t_req[i]:.2f} mm<br>Margin: {margin_pct[i, j]:.1f}%")
    if not np.isnan(measured[i, j]) else f"{names[i]} / {bay_cols[j]}<br>(καμία μέτρηση)"
    for j in range(n_bays)] for i in range(n_strakes)]

if np.isnan(measured).all():
    st.info("Δώσε μετρήσεις (ή πάτα «Συνθετικά δεδομένα») για να εμφανιστεί το heatmap.")
else:
    fig = go.Figure(data=go.Heatmap(
        z=margin_pct, x=bay_cols, y=names,
        colorscale=[[0.0, "#b30000"], [0.5, "#f5e642"], [1.0, "#0a7d2e"]],
        zmid=0, zmin=-30, zmax=30,
        text=hover_text, hoverinfo="text",
        colorbar=dict(title="Margin %"),
    ))
    fig.update_layout(
        title=f"{vessel_name} — Margin πάχους πάνω από t_req (%)",
        xaxis_title="Frame bay", yaxis_title="Strake",
        yaxis=dict(autorange="reversed"),
        height=160 + 45 * n_strakes,
    )
    st.plotly_chart(fig, width="stretch")

    # Συνοπτικά στατιστικά
    n_meas = int(np.count_nonzero(~np.isnan(measured)))
    n_below = int(np.nansum(measured < t_req_col))
    worst = np.nanmin(margin_pct)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Κελιά με μέτρηση", f"{n_meas} / {measured.size}")
    m2.metric("Κάτω από t_req", n_below)
    m3.metric("Χειρότερο margin", f"{worst:.1f} %")
    m4.metric("Μέσο margin", f"{np.nanmean(margin_pct):.1f} %")

    # ---- 5. Εκτίμηση εναπομείναντος χρόνου ζωής (απλό γραμμικό μοντέλο) ----
    with st.expander("⏳ Εκτίμηση εναπομεινάντων ετών ζωής (απλό γραμμικό μοντέλο φθοράς)"):
        st.caption(
            "years_left = (μέτρηση − t_req) / u, με ΕΝΙΑΙΟ ρυθμό διάβρωσης u (mm/έτος) που ορίζεις εσύ. "
            "Βασικό σημείο εκκίνησης· εδώ θα μπορούσε να μπει στοχαστικό μοντέλο φθοράς."
        )
        u = st.number_input("Ρυθμός φθοράς u [mm/yr]", min_value=0.001, value=0.10, step=0.01, format="%.3f")
        rows = []
        for i, name in enumerate(names):
            for j, bay in enumerate(bay_cols):
                if not np.isnan(measured[i, j]):
                    rows.append({
                        "Strake": name, "Frame bay": bay,
                        "Μέτρηση [mm]": round(float(measured[i, j]), 2),
                        "t_req [mm]": round(float(t_req[i]), 2),
                        "Έτη μέχρι t_req": round(float((measured[i, j] - t_req[i]) / u), 1),
                    })
        years_df = pd.DataFrame(rows).sort_values("Έτη μέχρι t_req").reset_index(drop=True)
        st.caption("Ταξινόμηση από το πιο κρίσιμο κελί. Αρνητική τιμή = ήδη κάτω από t_req.")
        st.dataframe(years_df, width="stretch", hide_index=True)

    # ---- 6. Εξαγωγή ------------------------------------------------------
    st.subheader("5. Εξαγωγή αποτελεσμάτων")
    long_rows = []
    for i, name in enumerate(names):
        for j, bay in enumerate(bay_cols):
            long_rows.append({
                "vessel": vessel_name, "note": survey_note,
                "strake": name, "location": strakes_df["location"].iloc[i], "frame_bay": bay,
                "t_req_mm": t_req[i],
                "measured_mm": measured[i, j],
                "margin_pct": margin_pct[i, j],
            })
    csv_bytes = pd.DataFrame(long_rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Λήψη CSV (long format)", csv_bytes, file_name="thickness_survey.csv", mime="text/csv")

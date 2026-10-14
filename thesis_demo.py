"""
thesis_demo.py - Αυτόνομο MVP: Χαρτογράφηση Φθοράς από Παχομετρήσεις (UT) υπάρχοντος πλοίου.

Εξαρτήσεις: streamlit, pandas, numpy, plotly (τίποτα άλλο).

Ροή:
  0. Βασικά χαρακτηριστικά πλοίου.
  1. Κατεβάζεις κενό CSV draft -> το συμπληρώνεις (strakes, t_req, μετρήσεις) -> το ανεβάζεις.
     (ή πληκτρολογείς απευθείας στους πίνακες της σελίδας)
  2. Heatmap περιθωρίου πάχους (2D grid ή σχηματική 3D όψη γάστρας).
  3. Εκτίμηση εναπομεινάντων ετών ζωής + εξαγωγή αποτελεσμάτων.

Εκτέλεση:  streamlit run thesis_demo.py
"""

import io
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Σταθερές
# ----------------------------------------------------------------------------
T_REQ_COL = "Απαιτούμενο Πάχος Κανονισμού (t_req) [mm]"
T_NOM_COL = "Ονομαστικό πάχος κατασκευής (t_nom) [mm] (προαιρετικό)"
LOCATIONS = ["bottom", "side", "deck", "bulkhead", "superstructure"]
DRAWABLE = ("bottom", "side", "deck")
DEFAULT_T_REQ = 6.0
MAX_STRAKES, MAX_BAYS = 20, 40
COLORSCALE = [[0.0, "#b30000"], [0.5, "#f5e642"], [1.0, "#0a7d2e"]]
CMIN, CMAX = -30, 30

VESSEL_TYPES = ["Φορτηγό / Bulk carrier", "Δεξαμενόπλοιο", "Επιβατηγό / Ro-Ro", "Αλιευτικό",
                "Ρυμουλκό / Εργασίας", "Ταχύπλοο / Yacht", "Άλλο"]
MATERIALS = ["Χάλυβας (steel)", "Αλουμίνιο", "FRP (πολυεστέρας)", "Ξύλο / Κόντρα πλακέ"]

# Αποδεκτά ονόματα στηλών στο CSV (κεφαλίδα -> εσωτερικό όνομα)
CSV_ALIASES = {
    "strake_name": "name", "strake": "name",
    "location": "location", "θέση": "location",
    "t_req_mm": "t_req", "t_req": "t_req", T_REQ_COL.lower(): "t_req",
    "t_nom_mm": "t_nom", "t_nom": "t_nom", T_NOM_COL.lower(): "t_nom",
}


# ----------------------------------------------------------------------------
# Βοηθητικές συναρτήσεις (χωρίς streamlit calls)
# ----------------------------------------------------------------------------
def make_default_strakes(n: int, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame({
        "strake_name": [f"Strake-{i + 1}" for i in range(start, n)],
        "location": ["bottom"] * (n - start),
        T_REQ_COL: [DEFAULT_T_REQ] * (n - start),
        T_NOM_COL: [np.nan] * (n - start),
    })


def resize_strakes(prev, n: int) -> pd.DataFrame:
    """Κρατά τα υπάρχοντα strakes όταν αλλάζει ο αριθμός τους."""
    if prev is None:
        return make_default_strakes(n)
    prev = prev.reset_index(drop=True)
    if len(prev) >= n:
        return prev.iloc[:n].copy()
    return pd.concat([prev, make_default_strakes(n, start=len(prev))], ignore_index=True)


def resize_measurements(prev, n_rows: int, n_cols: int) -> np.ndarray:
    out = np.full((n_rows, n_cols), np.nan)
    if prev is not None:
        r, c = min(n_rows, prev.shape[0]), min(n_cols, prev.shape[1])
        out[:r, :c] = prev[:r, :c]
    return out


def synthetic_measurements(t_req: np.ndarray, n_bays: int, seed: int = 42) -> np.ndarray:
    """Συνθετικές παχομετρήσεις για demo: φθορά μέγιστη στο μέσο + θόρυβος + τοπικό pitting."""
    rng = np.random.default_rng(seed)
    n = len(t_req)
    bays = np.linspace(-1, 1, n_bays)
    base_loss = 0.05 + 0.20 * (1 - bays ** 2)
    strake_factor = rng.uniform(0.6, 1.4, size=(n, 1))
    loss = base_loss[None, :] * strake_factor + rng.normal(0, 0.03, size=(n, n_bays))
    meas = (t_req.reshape(-1, 1) * 1.40) * (1 - np.clip(loss, 0, 0.6))
    pits = rng.random((n, n_bays)) < 0.06
    meas[pits] *= rng.uniform(0.6, 0.85, size=pits.sum())
    meas[rng.random((n, n_bays)) < 0.05] = np.nan
    return np.round(meas, 2)


def survey_to_csv_bytes(strakes_df: pd.DataFrame | None, meas: np.ndarray | None,
                        n_strakes: int, n_bays: int) -> bytes:
    """Wide CSV: strake_name;location;t_req_mm;t_nom_mm;FR1;...;FRn.
    strakes_df=None -> κενό draft. Διαχωριστικό ';' (ανοίγει σωστά σε Excel με ελληνικές ρυθμίσεις)."""
    bay_cols = [f"FR{j + 1}" for j in range(n_bays)]
    if strakes_df is None:
        base = pd.DataFrame({
            "strake_name": [f"Strake-{i + 1}" for i in range(n_strakes)],
            "location": ["bottom"] * n_strakes,
            "t_req_mm": [np.nan] * n_strakes,
            "t_nom_mm": [np.nan] * n_strakes,
        })
        grid = pd.DataFrame(np.full((n_strakes, n_bays), np.nan), columns=bay_cols)
    else:
        base = pd.DataFrame({
            "strake_name": strakes_df["strake_name"].to_numpy(),
            "location": strakes_df["location"].to_numpy(),
            "t_req_mm": strakes_df[T_REQ_COL].to_numpy(),
            "t_nom_mm": strakes_df[T_NOM_COL].to_numpy(),
        })
        grid = pd.DataFrame(meas, columns=bay_cols)
    out = pd.concat([base.reset_index(drop=True), grid.reset_index(drop=True)], axis=1)
    return out.to_csv(index=False, sep=";", na_rep="").encode("utf-8-sig")


def parse_survey_csv(raw: bytes):
    """Επιστρέφει (data|None, errors, warnings). data = dict(strakes_df, meas_arr, n_strakes, n_bays)."""
    errors, warnings = [], []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1253", errors="replace")   # Excel (ελληνικά Windows) «CSV» χωρίς UTF-8
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, ["Το αρχείο είναι κενό."], warnings
    head = lines[0]
    sep = max([";", ",", "\t"], key=head.count)
    if head.count(sep) == 0:
        return None, ["Δεν βρέθηκαν στήλες· έλεγξε ότι το αρχείο είναι CSV με διαχωριστικό ; ή ,"], warnings
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, skip_blank_lines=True)
    except Exception as e:  # noqa: BLE001 - μήνυμα προς τον χρήστη
        return None, [f"Αδυναμία ανάγνωσης CSV: {e}"], warnings

    colmap, fr_cols = {}, {}
    for c in df.columns:
        key = c.strip().lower()
        m = re.fullmatch(r"fr\s*(\d+)", key)
        if m:
            fr_cols[int(m.group(1))] = c
        elif key in CSV_ALIASES:
            colmap[CSV_ALIASES[key]] = c
    for need in ("name", "location", "t_req"):
        if need not in colmap:
            errors.append(f"Λείπει η στήλη «{ {'name': 'strake_name', 'location': 'location', 't_req': 't_req_mm'}[need] }».")
    if not fr_cols:
        errors.append("Δεν βρέθηκαν στήλες μετρήσεων FR1, FR2, ...")
    if errors:
        return None, errors, warnings

    def num(series: pd.Series, label: str):
        s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
        s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
        v = pd.to_numeric(s, errors="coerce")
        bad = v.isna() & s.notna()
        for i in np.flatnonzero(bad.to_numpy()):
            errors.append(f"Γραμμή {i + 2}, «{label}»: μη αριθμητική τιμή «{series.iloc[i]}».")
        return v

    # Αφαίρεση τελείως κενών γραμμών
    nonempty = df.apply(lambda r: any(str(x).strip() for x in r), axis=1)
    df = df[nonempty].reset_index(drop=True)
    if df.empty:
        return None, ["Δεν υπάρχουν γραμμές δεδομένων."], warnings
    if len(df) > MAX_STRAKES:
        errors.append(f"Έως {MAX_STRAKES} strakes (βρέθηκαν {len(df)}).")

    names = df[colmap["name"]].astype(str).str.strip()
    if (names == "").any():
        errors.append("Υπάρχουν γραμμές χωρίς strake_name.")
    if names.duplicated().any():
        errors.append("Διπλότυπα strake_name: " + ", ".join(sorted(set(names[names.duplicated()]))))
    loc = df[colmap["location"]].astype(str).str.strip().str.lower()
    bad_loc = sorted(set(loc[~loc.isin(LOCATIONS)]))
    if bad_loc:
        errors.append(f"Μη έγκυρο location: {', '.join(bad_loc)} (επιτρέπονται: {', '.join(LOCATIONS)}).")
    t_req = num(df[colmap["t_req"]], "t_req_mm")
    if t_req.isna().any() or (t_req <= 0).any():
        errors.append("Η στήλη t_req_mm πρέπει να έχει θετικό αριθμό σε κάθε γραμμή (συμπλήρωσέ την στο draft).")
    t_nom = num(df[colmap["t_nom"]], "t_nom_mm") if "t_nom" in colmap else pd.Series(np.nan, index=df.index)

    order = sorted(fr_cols)
    if order != list(range(1, len(order) + 1)):
        warnings.append("Οι στήλες FR δεν είναι συνεχόμενες· ξαναδιατάχθηκαν ως FR1..FRn με τη σειρά τους.")
    if len(order) > MAX_BAYS:
        errors.append(f"Έως {MAX_BAYS} frame bays (βρέθηκαν {len(order)}).")
    meas = np.column_stack([num(df[fr_cols[k]], f"FR{k}").to_numpy(dtype=float) for k in order])
    if np.nanmin(np.where(np.isnan(meas), np.inf, meas), initial=np.inf) <= 0:
        errors.append("Οι μετρήσεις πάχους πρέπει να είναι > 0 (άφησε κενό όπου δεν υπάρχει μέτρηση).")
    if np.nanmax(np.where(np.isnan(meas), -np.inf, meas), initial=-np.inf) > 200:
        warnings.append("Υπάρχουν μετρήσεις > 200 mm — έλεγξε τις μονάδες (mm).")
    if errors:
        return None, errors, warnings

    strakes_df = pd.DataFrame({
        "strake_name": names.to_numpy(), "location": loc.to_numpy(),
        T_REQ_COL: t_req.to_numpy(dtype=float), T_NOM_COL: t_nom.to_numpy(dtype=float),
    })
    return dict(strakes_df=strakes_df, meas_arr=meas, n_strakes=len(df), n_bays=len(order)), errors, warnings


def hull_outline_and_zones(loa: float, beam: float, depth: float, deadrise_deg: float):
    """Σχηματική τομή (μισή, δεξιά): κορυφές keel/chine/gunwale/deck-CL ως (y, z)."""
    hb = beam / 2.0
    zc = min(hb * np.tan(np.radians(deadrise_deg)), 0.9 * depth)
    pts = {"keel": (0.0, 0.0), "chine": (hb, zc), "gunwale": (hb, depth), "centre": (0.0, depth)}
    zones = {"bottom": (pts["keel"], pts["chine"]),
             "side": (pts["chine"], pts["gunwale"]),
             "deck": (pts["gunwale"], pts["centre"])}
    return pts, zones


def width_scale(x, loa: float):
    """Μόνο για εμφάνιση: πρισματική γάστρα με στένεμα στην πλώρη (σχηματικό)."""
    xf = np.clip(np.asarray(x, dtype=float) / loa, 0, 1)
    s = np.where(xf <= 0.6, 1.0, np.sqrt(np.clip(1 - ((xf - 0.6) / 0.4) ** 2, 0, 1)))
    return np.maximum(s, 0.02)


def build_hull_3d_figure(loa, beam, depth, deadrise, names, locations, bay_cols,
                         measured, t_req_col, mirror=False):
    """Σχηματική 3D γάστρα (όχι σχέδιο γραμμών) με χρωματισμό περιθωρίου πάχους ανά κελί.
    Strakes ανά location και σειρά καταχώρισης (ίσες αποστάσεις), FR1 στην πρύμνη."""
    pts, zones = hull_outline_and_zones(loa, beam, depth, deadrise)
    nb = len(bay_cols)
    margin = (measured - t_req_col) / t_req_col * 100.0
    sides = (1.0, -1.0) if mirror else (1.0,)
    NX = 2  # υποδιαιρέσεις κατά μήκος ανά κελί, ώστε να ακολουθείται το στένεμα

    members = {z: [i for i, n in enumerate(names) if locations[i] == z] for z in DRAWABLE}
    not_shown = [n for i, n in enumerate(names) if locations[i] not in DRAWABLE]

    def zpoint(zone, u, x, sgn):
        (y0, z0), (y1, z1) = zones[zone]
        y = (y0 + (y1 - y0) * u) * float(width_scale(x, loa))
        return sgn * y, z0 + (z1 - z0) * u

    meshes = {"meas": dict(x=[], y=[], z=[], i=[], j=[], k=[], inten=[], text=[]),
              "miss": dict(x=[], y=[], z=[], i=[], j=[], k=[], inten=[], text=[])}
    for zone in DRAWABLE:
        k_n = len(members[zone])
        for pos, row in enumerate(members[zone]):
            u0, u1 = pos / k_n, (pos + 1) / k_n
            for j in range(nb):
                xa, xb = j * loa / nb, (j + 1) * loa / nb
                has = not np.isnan(measured[row, j])
                d = meshes["meas" if has else "miss"]
                txt = (f"{names[row]} / {bay_cols[j]}<br>Μέτρηση: {measured[row, j]:.2f} mm<br>"
                       f"t_req: {t_req_col[row, 0]:.2f} mm<br>Margin: {margin[row, j]:.1f}%") if has \
                    else f"{names[row]} / {bay_cols[j]}<br>(καμία μέτρηση)"
                for sgn in sides:
                    base = len(d["x"])
                    for ix in range(NX + 1):
                        x = xa + (xb - xa) * ix / NX
                        for u in (u0, u1):
                            y, z = zpoint(zone, u, x, sgn)
                            d["x"].append(x); d["y"].append(y); d["z"].append(z); d["text"].append(txt)
                    for ix in range(NX):
                        a = base + ix * 2
                        b, c, e = a + 1, base + (ix + 1) * 2 + 1, base + (ix + 1) * 2
                        d["i"] += [a, a]; d["j"] += [b, c]; d["k"] += [c, e]
                        if has:
                            d["inten"] += [float(margin[row, j])] * 2

    fig = go.Figure()
    m = meshes["meas"]
    if m["x"]:
        fig.add_trace(go.Mesh3d(
            x=m["x"], y=m["y"], z=m["z"], i=m["i"], j=m["j"], k=m["k"],
            intensity=m["inten"], intensitymode="cell", colorscale=COLORSCALE, cmin=CMIN, cmax=CMAX,
            colorbar=dict(title="Margin %"), hovertext=m["text"], hoverinfo="text",
            flatshading=True, showscale=True, name="Μετρήσεις"))
    g = meshes["miss"]
    if g["x"]:
        fig.add_trace(go.Mesh3d(
            x=g["x"], y=g["y"], z=g["z"], i=g["i"], j=g["j"], k=g["k"], color="#bdbdbd", opacity=0.55,
            hovertext=g["text"], hoverinfo="text", flatshading=True, name="Χωρίς μέτρηση"))
    # περίγραμμα γάστρας (και των δύο πλευρών) για αναγνώριση σχήματος
    xs = np.linspace(0, loa, 41)
    for key in ("keel", "chine", "gunwale"):
        yy, zz = pts[key]
        for sgn in (1.0, -1.0):
            fig.add_trace(go.Scatter3d(
                x=xs, y=sgn * yy * width_scale(xs, loa), z=np.full_like(xs, zz), mode="lines",
                line=dict(color="#444", width=3), hoverinfo="skip", showlegend=False))
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="Μήκος (m) — FR1 στην πρύμνη", yaxis_title="Πλάτος (m)",
                   zaxis_title="Ύψος (m)", camera=dict(eye=dict(x=1.5, y=-1.7, z=0.9))),
        height=620, margin=dict(l=0, r=0, t=30, b=0), title="Σχηματική 3D όψη (όχι σχέδιο γραμμών)")
    return fig, not_shown


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Ship Thickness Survey Heatmap (MVP)", page_icon="🚢", layout="wide")

ss = st.session_state
ss.setdefault("n_strakes", 5)
ss.setdefault("n_bays", 8)
ss.setdefault("strakes_df", None)
ss.setdefault("meas_arr", None)
ss.setdefault("editor_version", 0)   # αλλάζει όταν αλλάζει η δομή, ώστε να μηδενίζονται οι παλιές επεξεργασίες των editors
ss.setdefault("upload_id", None)
ss.setdefault("upload_msg", None)

st.title("🚢 Χαρτογράφηση Φθοράς Πλοίου από Παχομετρήσεις (UT)")
st.caption(
    "Grid ανά strake (σειρά πλάτης) × frame bay (διαμέρισμα μήκους). Σύγκριση των μετρήσεων "
    "με το απαιτούμενο πάχος κανονισμού t_req και εκτίμηση της εναπομένουσας διάρκειας ζωής."
)

# ---- 0. Χαρακτηριστικά πλοίου ---------------------------------------------
st.subheader("0. Βασικά χαρακτηριστικά πλοίου")
v1, v2, v3 = st.columns(3)
with v1:
    vessel_name = v1.text_input("Όνομα πλοίου", value="Demo Vessel")
    vessel_type = v1.selectbox("Τύπος πλοίου", VESSEL_TYPES)
    material = v1.selectbox("Υλικό κατασκευής", MATERIALS)
    year_built = int(v1.number_input("Έτος ναυπήγησης", min_value=1900, max_value=date.today().year,
                                     value=2005, step=1))
with v2:
    loa = v2.number_input("LOA — Ολικό μήκος (m)", min_value=2.0, value=45.0, step=0.5)
    lwl = v2.number_input("LWL — Μήκος ίσαλου (m)", min_value=2.0, value=42.0, step=0.5)
    beam = v2.number_input("Beam — Πλάτος (m)", min_value=1.0, value=8.5, step=0.1)
    depth = v2.number_input("Depth — Κοίλο (m)", min_value=0.5, value=4.2, step=0.1)
with v3:
    draft = v3.number_input("Draft — Βύθισμα (m)", min_value=0.2, value=2.8, step=0.1)
    disp = v3.number_input("Εκτόπισμα (t)", min_value=1.0, value=650.0, step=10.0)
    deadrise = v3.number_input("Deadrise (°) — μόνο για το 3D σχήμα", min_value=0.0, max_value=45.0,
                               value=12.0, step=1.0)
    survey_date = v3.date_input("Ημερομηνία παχομέτρησης", value=date.today())
surveyor = st.text_input("Μελετητής / επιθεωρητής", value="")

geom_problems = []
if lwl > loa:
    geom_problems.append("Το LWL δεν μπορεί να είναι μεγαλύτερο από το LOA.")
if draft >= depth:
    geom_problems.append("Το βύθισμα πρέπει να είναι μικρότερο από το κοίλο.")
if survey_date.year < year_built:
    geom_problems.append("Η ημερομηνία παχομέτρησης είναι πριν από το έτος ναυπήγησης.")
for p in geom_problems:
    st.error("❌ " + p)
if geom_problems:
    st.stop()

age_years = (survey_date - date(year_built, 1, 1)).days / 365.25
k1, k2, k3, k4 = st.columns(4)
k1.metric("Ηλικία πλοίου", f"{age_years:.1f} έτη")
k2.metric("L/B", f"{loa / beam:.2f}")
k3.metric("B/D", f"{beam / depth:.2f}")
k4.metric("Cb (εκτ. ως ισοδύναμο)", f"{disp / (1.025 * lwl * beam * draft):.2f}")

st.divider()

# ---- 1. Δεδομένα survey: CSV draft / ανέβασμα -----------------------------
st.subheader("1. Δεδομένα survey")
st.markdown(
    "**Επιλογή Α (CSV):** κατέβασε το κενό draft → συμπλήρωσέ το (strakes, t_req, μετρήσεις) → ανέβασέ το. "
    "**Επιλογή Β:** πληκτρολόγησε απευθείας στους πίνακες παρακάτω."
)
with st.expander("ℹ️ Οδηγίες για το CSV"):
    st.markdown(
        "- Κάθε **γραμμή** = ένα strake. Στήλες: `strake_name`, `location`, `t_req_mm`, `t_nom_mm` (προαιρετικό), "
        "και μία στήλη μέτρησης ανά frame bay (`FR1`, `FR2`, ...).\n"
        f"- `location`: ένα από {', '.join('`' + x + '`' for x in LOCATIONS)}.\n"
        "- `t_req_mm`: το απαιτούμενο πάχος κανονισμού (υποχρεωτικό, > 0). "
        "`t_nom_mm`: το αρχικό πάχος κατασκευής (προαιρετικό — επιτρέπει εκτίμηση ρυθμού φθοράς από την ηλικία).\n"
        "- Άφησε **κενό** το κελί όπου δεν υπάρχει μέτρηση. Τιμές σε mm.\n"
        "- Διαχωριστικό `;` ή `,` και δεκαδικά με τελεία ή κόμμα γίνονται δεκτά (Excel ελληνικών ρυθμίσεων)."
    )

up1, up2 = st.columns(2)
n_tpl_s, n_tpl_b = int(ss["n_strakes"]), int(ss["n_bays"])
up1.download_button(
    f"📄 Κατέβασμα κενού CSV draft ({n_tpl_s} strakes × {n_tpl_b} bays)",
    survey_to_csv_bytes(None, None, n_tpl_s, n_tpl_b),
    file_name="thickness_survey_draft.csv", mime="text/csv",
    help="Οι διαστάσεις του draft ακολουθούν τον αριθμό strakes/bays που ορίζεις παρακάτω.")
uploaded = up2.file_uploader("📤 Ανέβασμα συμπληρωμένου CSV", type=["csv"], key="csv_upload")

if uploaded is not None and ss["upload_id"] != uploaded.file_id:
    ss["upload_id"] = uploaded.file_id
    data, errs, warns = parse_survey_csv(uploaded.getvalue())
    if data is None:
        ss["upload_msg"] = ("error", f"Το αρχείο «{uploaded.name}» δεν φορτώθηκε:", errs)
    else:
        ss["n_strakes"], ss["n_bays"] = data["n_strakes"], data["n_bays"]
        ss["strakes_df"], ss["meas_arr"] = data["strakes_df"], data["meas_arr"]
        ss["editor_version"] += 1
        ss["upload_msg"] = ("success",
                            f"Φορτώθηκε «{uploaded.name}»: {data['n_strakes']} strakes × {data['n_bays']} bays.", warns)
elif uploaded is None and ss["upload_id"] is not None:
    ss["upload_id"] = None   # ο χρήστης αφαίρεσε το αρχείο· επιτρέπεται νέο ανέβασμα του ίδιου

if ss["upload_msg"]:
    kind, head, items = ss["upload_msg"]
    if kind == "error":
        st.error(head + "\n\n" + "\n".join(f"- {e}" for e in items))
    else:
        st.success(head)
        for w in items:
            st.warning(w)

gc1, gc2 = st.columns(2)
n_strakes = int(gc1.number_input("Αριθμός strakes (σειρές πλάτης)", min_value=1, max_value=MAX_STRAKES,
                                 step=1, key="n_strakes"))
n_bays = int(gc2.number_input("Αριθμός frame bays (διαμερίσματα μήκους)", min_value=1, max_value=MAX_BAYS,
                              step=1, key="n_bays"))

if ss["strakes_df"] is None or len(ss["strakes_df"]) != n_strakes:
    ss["strakes_df"] = resize_strakes(ss["strakes_df"], n_strakes)
    ss["editor_version"] += 1
if ss["meas_arr"] is None or ss["meas_arr"].shape != (n_strakes, n_bays):
    ss["meas_arr"] = resize_measurements(ss["meas_arr"], n_strakes, n_bays)
    ss["editor_version"] += 1
ver = ss["editor_version"]

# ---- 2. Strakes + t_req ----------------------------------------------------
st.subheader("2. Strakes και απαιτούμενο πάχος κανονισμού")
st.caption(f"Πληκτρολόγησε στη στήλη «{T_REQ_COL}» το t_req κάθε strake. "
           "Το t_nom είναι προαιρετικό (χρησιμοποιείται μόνο στην εκτίμηση ρυθμού φθοράς).")
strakes_df = st.data_editor(
    ss["strakes_df"],
    column_config={
        "strake_name": st.column_config.TextColumn("strake_name", required=True),
        "location": st.column_config.SelectboxColumn("location", options=LOCATIONS, required=True),
        T_REQ_COL: st.column_config.NumberColumn(T_REQ_COL, min_value=0.1, max_value=200.0, step=0.1,
                                                 format="%.2f", required=False),
        T_NOM_COL: st.column_config.NumberColumn(T_NOM_COL, min_value=0.1, max_value=200.0, step=0.1,
                                                 format="%.2f", required=False),
    },
    num_rows="fixed", hide_index=True, key=f"strakes_editor_{ver}", width="stretch",
)
ss["strakes_df"] = strakes_df

names = [str(n).strip() for n in strakes_df["strake_name"]]
locations = [str(x) for x in strakes_df["location"]]
t_req_series = pd.to_numeric(strakes_df[T_REQ_COL], errors="coerce")
t_nom = pd.to_numeric(strakes_df[T_NOM_COL], errors="coerce").to_numpy(dtype=float)
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
t_req_col = t_req.reshape(-1, 1)
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

meas_df = st.data_editor(
    pd.DataFrame(ss["meas_arr"], index=names, columns=bay_cols),
    column_config={c: st.column_config.NumberColumn(c, min_value=0.0, step=0.1, format="%.2f") for c in bay_cols},
    key=f"meas_editor_{ver}", width="stretch",
)
measured = meas_df.to_numpy(dtype=float)
ss["meas_arr"] = measured

# ---- 4. Heatmap ------------------------------------------------------------
st.subheader("4. Heatmap")
margin_pct = (measured - t_req_col) / t_req_col * 100.0

if np.isnan(measured).all():
    st.info("Δώσε μετρήσεις (ή ανέβασε CSV / πάτα «Συνθετικά δεδομένα») για να εμφανιστεί το heatmap.")
else:
    view_mode = st.radio("Προβολή", ["2D (grid)", "3D (σχηματική γάστρα)"], horizontal=True)
    if view_mode.startswith("2D"):
        hover_text = [[
            (f"{names[i]} / {bay_cols[j]}<br>Μέτρηση: {measured[i, j]:.2f} mm<br>"
             f"t_req: {t_req[i]:.2f} mm<br>Margin: {margin_pct[i, j]:.1f}%")
            if not np.isnan(measured[i, j]) else f"{names[i]} / {bay_cols[j]}<br>(καμία μέτρηση)"
            for j in range(n_bays)] for i in range(n_strakes)]
        fig = go.Figure(data=go.Heatmap(
            z=margin_pct, x=bay_cols, y=names, colorscale=COLORSCALE, zmid=0, zmin=CMIN, zmax=CMAX,
            text=hover_text, hoverinfo="text", colorbar=dict(title="Margin %")))
        fig.update_layout(
            title=f"{vessel_name} — Margin πάχους πάνω από t_req (%)",
            xaxis_title="Frame bay", yaxis_title="Strake", yaxis=dict(autorange="reversed"),
            height=160 + 45 * n_strakes)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption(
            "⚠️ **Σχηματικό μοντέλο, όχι σχέδιο γραμμών:** πρισματική hard-chine γάστρα από LOA/Beam/Depth/Deadrise "
            "με στένεμα στην πλώρη. Τα strakes τοποθετούνται ανά location και σειρά καταχώρισης (ίσες αποστάσεις: "
            "bottom = καρίνα→chine, side = chine→κουπαστή, deck = κουπαστή→άξονας), FR1 στην πρύμνη. "
            "Bulkhead/superstructure δεν απεικονίζονται.")
        mirror = st.checkbox("Εμφάνιση και του άλλου πλευρού (ίδιες τιμές, συμμετρία — ΟΧΙ πραγματικές μετρήσεις)",
                             value=False)
        fig3d, not_shown = build_hull_3d_figure(loa, beam, depth, deadrise, names, locations, bay_cols,
                                                measured, t_req_col, mirror=mirror)
        st.plotly_chart(fig3d, width="stretch")
        if not_shown:
            st.info("Δεν απεικονίζονται στο 3D: " + ", ".join(not_shown))

    n_meas = int(np.count_nonzero(~np.isnan(measured)))
    n_below = int(np.nansum(measured < t_req_col))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Κελιά με μέτρηση", f"{n_meas} / {measured.size}")
    m2.metric("Κάτω από t_req", n_below)
    m3.metric("Χειρότερο margin", f"{np.nanmin(margin_pct):.1f} %")
    m4.metric("Μέσο margin", f"{np.nanmean(margin_pct):.1f} %")

    # ---- Εκτίμηση ετών ζωής -------------------------------------------------
    with st.expander("⏳ Εκτίμηση εναπομεινάντων ετών ζωής (γραμμικό μοντέλο φθοράς)"):
        have_nom = (~np.isnan(t_nom)).any() and age_years > 0
        mode = st.radio(
            "Ρυθμός φθοράς u",
            ["Ενιαίος ρυθμός (ορίζεται από εσένα)", "Ανά κελί από ηλικία: u = (t_nom − μέτρηση) / ηλικία"],
            horizontal=True, index=0,
            help="Η δεύτερη επιλογή απαιτεί συμπλήρωση του t_nom (αρχικό πάχος) στον πίνακα strakes.")
        by_age = mode.startswith("Ανά κελί")
        if by_age and not have_nom:
            st.warning("Συμπλήρωσε t_nom σε τουλάχιστον ένα strake (και ηλικία > 0) — προς το παρόν χρησιμοποιείται ενιαίος ρυθμός.")
            by_age = False
        u_uniform = st.number_input("Ενιαίος ρυθμός φθοράς u [mm/yr]", min_value=0.001, value=0.10,
                                    step=0.01, format="%.3f", disabled=by_age)
        rows = []
        for i, name in enumerate(names):
            for j, bay in enumerate(bay_cols):
                mv = measured[i, j]
                if np.isnan(mv):
                    continue
                if by_age:
                    if np.isnan(t_nom[i]):
                        continue
                    u = (t_nom[i] - mv) / age_years
                else:
                    u = u_uniform
                years = (mv - t_req[i]) / u if u > 0 else np.nan
                rows.append({"Strake": name, "Frame bay": bay, "Μέτρηση [mm]": round(float(mv), 2),
                             "t_req [mm]": round(float(t_req[i]), 2), "u [mm/yr]": round(float(u), 3),
                             "Έτη μέχρι t_req": round(float(years), 1) if not np.isnan(years) else np.nan})
        if rows:
            years_df = pd.DataFrame(rows).sort_values("Έτη μέχρι t_req", na_position="last").reset_index(drop=True)
            st.caption("Ταξινόμηση από το πιο κρίσιμο κελί. Αρνητική τιμή = ήδη κάτω από t_req· "
                       "κενό = δεν παρατηρείται φθορά (u ≤ 0).")
            st.dataframe(years_df, width="stretch", hide_index=True)
        else:
            st.info("Δεν υπάρχουν κελιά για υπολογισμό.")

    # ---- 5. Εξαγωγή ---------------------------------------------------------
    st.subheader("5. Εξαγωγή")
    long_rows = []
    for i, name in enumerate(names):
        for j, bay in enumerate(bay_cols):
            long_rows.append({
                "vessel": vessel_name, "type": vessel_type, "material": material, "year_built": year_built,
                "survey_date": survey_date.isoformat(), "surveyor": surveyor,
                "strake": name, "location": locations[i], "frame_bay": bay,
                "t_req_mm": t_req[i], "measured_mm": measured[i, j], "margin_pct": margin_pct[i, j]})
    e1, e2 = st.columns(2)
    e1.download_button("⬇️ Αποτελέσματα CSV (long format, για ανάλυση)",
                       pd.DataFrame(long_rows).to_csv(index=False).encode("utf-8-sig"),
                       file_name="thickness_survey_results.csv", mime="text/csv")
    e2.download_button("💾 Αποθήκευση τρέχοντος survey (CSV για επανεισαγωγή)",
                       survey_to_csv_bytes(strakes_df, measured, n_strakes, n_bays),
                       file_name="thickness_survey_current.csv", mime="text/csv")

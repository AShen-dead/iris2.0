import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles (v1 aesthetic, green palette) ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --primary:   #346739;
    --secondary: #79AE6F;
    --tertiary:  #9FCB98;
    --accent:    #F2EDC2;
    --bg:        #0b110c;
    --bg2:       #101a11;
    --bg3:       #151f16;
    --border:    #1e2e1f;
    --border2:   #263827;
    --muted:     #5a7a5d;
    --text:      #e8ede8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background-color: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f1f10 0%, #122014 50%, #1a3320 100%);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🌸';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.12;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: var(--accent);
    margin: 0 0 0.4rem;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1rem;
    color: var(--muted);
    margin: 0;
    font-weight: 300;
}

/* ── Cards ── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.section-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
    display: block;
}

/* ── Prediction result box ── */
.result-box {
    background: #0d2212;
    border: 1px solid var(--primary);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-box .species-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: var(--tertiary);
    display: block;
    margin-bottom: 0.3rem;
}
.result-box .result-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
}
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #8b9e8d;
    width: 90px;
    flex-shrink: 0;
    text-transform: capitalize;
}
.prob-bar-bg {
    flex: 1;
    background: var(--border);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
}
.prob-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Dataset / expander ── */
.dataset-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.dataset-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: var(--accent);
}
.dataset-badge {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 99px;
    padding: 0.25rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--secondary);
}

/* ── Sliders ── */
.stSlider > div > div > div { background: var(--secondary) !important; }
div[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    color: #8b9e8d !important;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Metric boxes ── */
[data-testid="stMetricValue"] {
    color: var(--tertiary) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Caption ── */
.stCaption { color: var(--muted) !important; font-size: 0.78rem !important; }

/* ── Subheader ── */
h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--accent) !important;
    letter-spacing: -0.5px !important;
}

/* ── DataFrame ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Hide sidebar ── */
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Load model & encoder ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("iris_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# ── Load dataset for visuals ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("iris.csv")

df = load_data()

# ── Species colours (green palette) ──────────────────────────────────────────
COLORS = {"setosa": "#346739", "versicolor": "#79AE6F", "virginica": "#9FCB98"}
SPECIES_EMOJI = {"setosa": "🌼", "versicolor": "🌿", "virginica": "🌺"}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>Iris Species Predictor</h1>
</div>
""", unsafe_allow_html=True)

# ── Inline sliders ───────────────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<span class='section-label'>🌿 Sepal Measurements</span>", unsafe_allow_html=True)
scol1, scol2 = st.columns(2)
with scol1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
with scol2:
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)

st.markdown("<br><span class='section-label'>🌸 Petal Measurements</span>", unsafe_allow_html=True)
scol3, scol4 = st.columns(2)
with scol3:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
with scol4:
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred_idx   = model.predict(input_data)[0]
pred_proba = model.predict_proba(input_data)[0]
pred_label = le.inverse_transform([pred_idx])[0]
color      = COLORS[pred_label]
emoji      = SPECIES_EMOJI.get(pred_label, "🌸")

# ── Top row: prediction card + probability bar ────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(
        f"""
        <div class='result-box' style='border-color:{color}; background:{color}18;'>
            <div style='font-size:48px; margin-bottom:4px;'>{emoji}</div>
            <span class='species-name' style='color:{color};'>Iris {pred_label.capitalize()}</span>
            <span class='result-label'>Predicted Species</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("<span class='section-label'>Prediction Confidence</span>", unsafe_allow_html=True)
    classes = le.classes_
    for sp, prob in zip(classes, pred_proba):
        pct = int(prob * 100)
        sp_color = COLORS[sp]
        sp_emoji = SPECIES_EMOJI.get(sp, "")
        st.markdown(f"""
        <div class='prob-row'>
            <span class='prob-label'>{sp_emoji} {sp}</span>
            <div class='prob-bar-bg'>
                <div class='prob-bar-fill' style='width:{pct}%; background:{sp_color};'></div>
            </div>
            <span class='prob-val'>{pct}%</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Dataset preview ───────────────────────────────────────────────────────────
st.markdown(f"""
<div class='dataset-header'>
    <span class='dataset-title'>📋 Iris Dataset Reference</span>
    <span class='dataset-badge'>150 samples · 3 species · 4 features</span>
</div>
""", unsafe_allow_html=True)

with st.expander("View Full Dataset"):
    st.dataframe(df, use_container_width=True, height=260)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total samples", len(df))
    c2.metric("Features", 4)
    c3.metric("Classes", df["species"].nunique())

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Model: Random Forest (100 trees) · Trained on Iris dataset · Accuracy: 100%")

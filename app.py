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
)

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

# ── Species colours ───────────────────────────────────────────────────────────
COLORS = {"setosa": "#e63946", "versicolor": "#2a9d8f", "virginica": "#e9c46a"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg", use_container_width=True)
st.sidebar.title("🌸 Iris Classifier")
st.sidebar.markdown("Adjust the sliders to predict the Iris species.")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

# ── Prediction ────────────────────────────────────────────────────────────────
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred_idx   = model.predict(input_data)[0]
pred_proba = model.predict_proba(input_data)[0]
pred_label = le.inverse_transform([pred_idx])[0]
color      = COLORS[pred_label]

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='text-align:center;'>🌸 Iris Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Top row: prediction card + probability bar
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(
        f"""
        <div style='background:{color}22; border:2px solid {color}; border-radius:16px;
                    padding:28px; text-align:center;'>
            <div style='font-size:56px;'>🌺</div>
            <div style='font-size:28px; font-weight:700; color:{color}; margin-top:8px;'>
                Iris {pred_label.capitalize()}
            </div>
            <div style='font-size:14px; color:#888; margin-top:4px;'>Predicted species</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.subheader("Prediction Confidence")
    classes = le.classes_
    fig, ax = plt.subplots(figsize=(6, 2.6))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    bars = ax.barh(classes, pred_proba, color=[COLORS[c] for c in classes], height=0.5, edgecolor="none")
    for bar, prob in zip(bars, pred_proba):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=11, color="#555")
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Probability", fontsize=10)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    st.pyplot(fig)

st.markdown("---")

# Second row: PCA scatter + feature importance
col3, col4 = st.columns(2)

with col3:
    st.subheader("Dataset — PCA Projection")
    pca = PCA(n_components=2)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    X_2d = pca.fit_transform(X)
    user_2d = pca.transform(input_data)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("none")
    ax2.set_facecolor("none")
    for species, grp in df.groupby("species"):
        idx = df.index[df["species"] == species]
        ax2.scatter(X_2d[idx, 0], X_2d[idx, 1],
                    color=COLORS[species], label=species.capitalize(),
                    alpha=0.7, edgecolors="none", s=50)
    ax2.scatter(user_2d[0, 0], user_2d[0, 1],
                color=color, s=220, marker="*", zorder=5,
                edgecolors="white", linewidths=0.8, label="Your input")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=9)
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=9)
    ax2.legend(fontsize=9, framealpha=0)
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)

with col4:
    st.subheader("Feature Importances")
    features  = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    importances = model.feature_importances_
    order = np.argsort(importances)

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    fig3.patch.set_facecolor("none")
    ax3.set_facecolor("none")
    ax3.barh([features[i] for i in order], importances[order],
             color="#457b9d", edgecolor="none", height=0.5)
    ax3.set_xlabel("Importance", fontsize=9)
    ax3.spines[["top", "right", "bottom"]].set_visible(False)
    ax3.tick_params(labelsize=10)
    st.pyplot(fig3)

st.markdown("---")

# Dataset preview
with st.expander("📋 View Dataset"):
    st.dataframe(df, use_container_width=True, height=260)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total samples", len(df))
    c2.metric("Features", 4)
    c3.metric("Classes", df["species"].nunique())

st.caption("Model: Random Forest (100 trees) · Trained on Iris dataset · Accuracy: 100%")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os


# 1. CONFIG GENERALE

st.set_page_config(
    page_title="Dashboard Examens Étudiants",
    layout="wide"
)

st.title("Dashboard des résultats d'examen — Student Exam Scores")
st.write("Données issues du fichier **Student_exam_nettoyees.csv**.")


# 2. CHARGEMENT DES DONNÉES

DEFAULT_PATH = "Student_exam_nettoyees.csv"

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df

uploaded = st.sidebar.file_uploader("📂 Importer un fichier CSV (optionnel)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("Données chargées depuis le fichier uploadé ")
elif os.path.exists(DEFAULT_PATH):
    df = load_data(DEFAULT_PATH)
    st.info(f"Données chargées depuis {DEFAULT_PATH} ")
else:
    st.error("Aucun fichier trouvé. Place `Student_exam_nettoyees.csv` dans le même dossier que ce script.")
    st.stop()


# 3. FILTRES DANS LA SIDEBAR

st.sidebar.header("Filtres")

# filtre tranche_heures (si la colonne existe)
if "tranche_heures" in df.columns:
    tranches = df["tranche_heures"].dropna().unique().tolist()
    tranches_sel = st.sidebar.multiselect(
        "Tranche d'heures d'étude",
        options=sorted(tranches),
        default=sorted(tranches)
    )
    df = df[df["tranche_heures"].isin(tranches_sel)]

# filtre sur les heures d'étude
if "hours_studied" in df.columns:
    min_h, max_h = float(df["hours_studied"].min()), float(df["hours_studied"].max())
    sel_min, sel_max = st.sidebar.slider(
        "Plage d'heures d'étude",
        min_value=min_h,
        max_value=max_h,
        value=(min_h, max_h),
        step=1.0
    )
    df = df[(df["hours_studied"] >= sel_min) & (df["hours_studied"] <= sel_max)]

st.sidebar.write(f"Nombre d'observations après filtres : **{len(df)}**")


# 4. INDICATEURS CLES (KPI)

st.subheader("Vue globale — Portefeuille filtré")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Nombre d'observations", len(df))

with col2:
    if "hours_studied" in df.columns:
        st.metric("Heures d'étude moyennes", f"{df['hours_studied'].mean():.2f}")
    else:
        st.metric("Heures d'étude moyennes", "N/A")

with col3:
    if "attendance_percent" in df.columns:
        st.metric("Assiduité moyenne (%)", f"{df['attendance_percent'].mean():.2f}")
    else:
        st.metric("Assiduité moyenne (%)", "N/A")

with col4:
    if "previous_scores" in df.columns:
        st.metric("Score précédent moyen", f"{df['previous_scores'].mean():.2f}")
    else:
        st.metric("Score précédent moyen", "N/A")

with col5:
    if "exam_score" in df.columns:
        st.metric("Score d'examen moyen", f"{df['exam_score'].mean():.2f}")
    else:
        st.metric("Score d'examen moyen", "N/A")

st.markdown("---")


# 5. GRAPHIQUES


tab1, tab2, tab3 = st.tabs(["Distribution", "Corrélations", "Relations"])

# ----- TAB 1 : Distribution -----
with tab1:
    st.subheader("Distribution des scores et des heures d'étude")

    col_a, col_b = st.columns(2)

    with col_a:
        if "exam_score" in df.columns:
            st.write("Histogramme du **score d'examen**")
            fig, ax = plt.subplots()
            ax.hist(df["exam_score"], bins=20)
            ax.set_xlabel("exam_score")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)

    with col_b:
        if "hours_studied" in df.columns:
            st.write("Histogramme des **heures d'étude**")
            fig, ax = plt.subplots()
            ax.hist(df["hours_studied"], bins=20)
            ax.set_xlabel("hours_studied")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)

# TAB 2 : Corrélations
with tab2:
    st.subheader("Matrice de corrélations (variables numériques)")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest",
                    vmin=-1, vmax=1, linewidths=0.7, linecolor="white", ax=ax)
        ax.set_title("Matrice de corrélations")
        st.pyplot(fig)
    else:
        st.info("Pas assez de variables numériques pour calculer une corrélation.")

# TAB 3 : Relations
with tab3:
    st.subheader("Relation entre heures d'étude et score d'examen")

    if {"hours_studied", "exam_score"}.issubset(df.columns):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="hours_studied", y="exam_score",
                        hue="tranche_heures" if "tranche_heures" in df.columns else None,
                        ax=ax)
        ax.set_xlabel("Heures d'étude")
        ax.set_ylabel("Score à l'examen")
        ax.set_title("Nuage de points : hours_studied vs exam_score")
        st.pyplot(fig)

        # petite régression linéaire
        clean = df[["hours_studied", "exam_score"]].dropna()
        if len(clean) > 2:
            coef = np.polyfit(clean["hours_studied"], clean["exam_score"], 1)
            a, b = coef
            x_line = np.linspace(clean["hours_studied"].min(), clean["hours_studied"].max(), 100)
            y_line = a * x_line + b

            fig2, ax2 = plt.subplots()
            ax2.scatter(clean["hours_studied"], clean["exam_score"], alpha=0.6, label="Données")
            ax2.plot(x_line, y_line, color="red", label=f"y = {a:.2f}x + {b:.2f}")
            ax2.set_xlabel("Heures d'étude")
            ax2.set_ylabel("Score à l'examen")
            ax2.set_title("Régression linéaire simple")
            ax2.legend()
            st.pyplot(fig2)

            # R²
            y_pred = a * clean["hours_studied"] + b
            ss_res = ((clean["exam_score"] - y_pred)**2).sum()
            ss_tot = ((clean["exam_score"] - clean["exam_score"].mean())**2).sum()
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else float("nan")
            st.write(f"**R² de la régression (hours_studied → exam_score) : {r2:.3f}**")
    else:
        st.info("Les colonnes `hours_studied` et `exam_score` sont nécessaires pour ce graphique.")

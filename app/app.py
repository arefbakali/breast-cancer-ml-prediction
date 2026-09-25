import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Permet d'importer le package "monitoring" (situé à la racine du projet)
# lorsque l'app est lancée avec `streamlit run app/app.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monitoring.logger import log_prediction  # noqa: E402

st.set_page_config(
    page_title="Détection Cancer du Sein — ML",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0a3d6b; }
  [data-testid="stSidebar"] * { color: #e8f4fd !important; }
  .kpi-card {
    background: #f0f7ff; border-left: 5px solid #1565C0;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
  }
  .kpi-label { font-size: 0.78rem; color: #555; font-weight: 600; text-transform: uppercase; }
  .kpi-value { font-size: 1.6rem; font-weight: 800; color: #1565C0; }
  .best-badge {
    background: #1565C0; color: white; padding: 4px 12px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 700;
  }
  .warning-box {
    background: #fff3e0; border-left: 5px solid #e65100;
    border-radius: 8px; padding: 12px 16px; margin: 12px 0;
  }
  .result-cancer {
    background: #ffebee; border: 2px solid #c62828;
    border-radius: 12px; padding: 20px; text-align: center;
  }
  .result-sain {
    background: #e8f5e9; border: 2px solid #2e7d32;
    border-radius: 12px; padding: 20px; text-align: center;
  }
  h1 { color: #1565C0; }
  .stButton>button { background:#1565C0; color:white; border-radius:8px; font-weight:700; }
  .stButton>button:hover { background:#0d47a1; }
</style>
""", unsafe_allow_html=True)



BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

@st.cache_data
def load_data():
    return pd.read_csv(PROJECT_DIR / "data" / "dataR2.csv")

@st.cache_resource
def load_model():
    with open(PROJECT_DIR / "models" / "pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

df = load_data()
pipeline = load_model()

FEATURES = ["Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1"]
TARGET = "Classification"

RESULTS = pd.DataFrame([
    {"Modèle": "Régression Logistique", "Accuracy": 0.5417, "Précision": 0.5417, "Recall": 1.0000, "F1": 0.7027, "AUC": 0.7692, "FN": 0,  "FP": 11},
    {"Modèle": "Naive Bayes",           "Accuracy": 0.6667, "Précision": 0.8571, "Recall": 0.4615, "F1": 0.6000, "AUC": 0.8671, "FN": 7,  "FP": 1},
    {"Modèle": "Ridge (L2)",            "Accuracy": 0.5417, "Précision": 0.5417, "Recall": 1.0000, "F1": 0.7027, "AUC": 0.7622, "FN": 0,  "FP": 11},
    {"Modèle": "Lasso (L1)",            "Accuracy": 0.5417, "Précision": 0.5417, "Recall": 1.0000, "F1": 0.7027, "AUC": 0.5000, "FN": 0,  "FP": 11},
    {"Modèle": "K-NN",                  "Accuracy": 0.7500, "Précision": 0.7692, "Recall": 0.7692, "F1": 0.7692, "AUC": 0.8252, "FN": 3,  "FP": 3},
    {"Modèle": "MLP (Réseau Neurones)", "Accuracy": 0.7500, "Précision": 0.8182, "Recall": 0.6923, "F1": 0.7500, "AUC": 0.7133, "FN": 4,  "FP": 2},
]).set_index("Modèle")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎗️ Navigation")
    page = st.radio("", ["🏠 Accueil", "📊 Données", "📈 Visualisations", "🤖 Modèles & Résultats", "🔬 Prédiction Patient", "🩺 Monitoring"])
    st.markdown("---")
    st.markdown("**Dataset :** Coimbra Breast Cancer")
    st.markdown(f"**Observations :** {len(df)}")
    st.markdown(f"**Variables :** {len(FEATURES)}")
    st.markdown("**Meilleur modèle :** K-NN ⭐")
    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;opacity:0.7">⚠️ Outil d\'aide à la décision uniquement</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🎗️ Détection du Cancer du Sein par Machine Learning")
        st.markdown("""
        Cette application compare **6 modèles de machine learning** entraînés sur le
        **Coimbra Breast Cancer Dataset** pour prédire la présence d'un cancer du sein
        à partir de biomarqueurs sanguins.

        Les biomarqueurs analysés incluent des indicateurs métaboliques (Glucose, Insuline,
        HOMA) et des adipokines (Leptine, Adiponectine, Résistine, MCP-1), dont les niveaux
        sont associés au risque de cancer du sein selon la littérature scientifique.
        """)
        st.markdown("""
        <div class="warning-box">
        ⚕️ <strong>Avertissement médical :</strong> Ce système est une <strong>aide à la décision</strong>
        à usage éducatif et de recherche. Il ne remplace en aucun cas l'avis d'un professionnel de santé.
        Tout résultat doit être confirmé par un médecin qualifié.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("### 📌 Résumé")
        for label, value in [("Patients", "116"), ("Sains (classe 0)", "52"), ("Cancer (classe 1)", "64"), ("Biomarqueurs", "9"), ("Modèles testés", "6"), ("Meilleur modèle", "K-NN ⭐")]:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Biomarqueurs utilisés")
    cols = st.columns(3)
    descriptions = {
        "Age": "Âge du patient (années)", "BMI": "Indice de masse corporelle",
        "Glucose": "Glycémie à jeun (mg/dL)", "Insulin": "Insulinémie (µU/mL)",
        "HOMA": "Résistance à l'insuline", "Leptin": "Leptine (ng/mL)",
        "Adiponectin": "Adiponectine (µg/mL)", "Resistin": "Résistine (ng/mL)",
        "MCP.1": "Protéine chimiotactique (pg/dL)",
    }
    for i, (feat, desc) in enumerate(descriptions.items()):
        with cols[i % 3]:
            st.info(f"**{feat}** — {desc}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Données":
    st.title("📊 Données — Coimbra Breast Cancer Dataset")
    tab1, tab2, tab3 = st.tabs(["Aperçu", "Statistiques", "Distribution cible"])

    with tab1:
        st.markdown(f"**{len(df)} observations × {len(df.columns)} colonnes**")
        display_df = df.copy()
        display_df[TARGET] = display_df[TARGET].map({0: "0 — Sain", 1: "1 — Cancer"})
        st.dataframe(display_df.head(20), use_container_width=True)

    with tab2:
        stats = df[FEATURES].describe().T.round(3)
        stats.columns = ["N", "Moyenne", "Écart-type", "Min", "Q1", "Médiane", "Q3", "Max"]
        st.dataframe(stats.style.background_gradient(cmap="Blues", subset=["Moyenne"]), use_container_width=True)

    with tab3:
        counts = df[TARGET].value_counts().sort_index()
        fig = px.pie(values=counts.values, names=["Sain (0)", "Cancer (1)"],
                     color_discrete_sequence=["#4CAF50", "#E53935"],
                     title="Répartition des classes", hole=0.4)
        fig.update_traces(textinfo="percent+label+value")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("Classe 0 — Sains", counts[0], f"{counts[0]/len(df)*100:.1f}%")
        col2.metric("Classe 1 — Cancer", counts[1], f"{counts[1]/len(df)*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualisations":
    st.title("📈 Visualisations Exploratoires")
    tab1, tab2, tab3 = st.tabs(["Distributions", "Corrélation", "Boxplots"])

    with tab1:
        feature_sel = st.selectbox("Variable à visualiser", FEATURES)
        fig = px.histogram(df, x=feature_sel, color=TARGET,
                           color_discrete_map={0: "#4CAF50", 1: "#E53935"},
                           barmode="overlay", nbins=25,
                           labels={TARGET: "Classe"},
                           title=f"Distribution de {feature_sel} par classe")
        fig.update_traces(opacity=0.7)
        fig.for_each_trace(lambda t: t.update(name="Sain" if t.name == "0" else "Cancer"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = df[FEATURES].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Matrice de corrélation des biomarqueurs", aspect="auto")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Forte corrélation entre **Insulin** et **HOMA** (HOMA est calculé à partir de l'insuline).")

    with tab3:
        key_vars = ["Glucose", "Insulin", "HOMA", "Leptin", "Resistin", "BMI"]
        fig = make_subplots(rows=2, cols=3, subplot_titles=key_vars)
        for i, var in enumerate(key_vars):
            r, c = divmod(i, 3)
            for cls, color, name in [(0, "#4CAF50", "Sain"), (1, "#E53935", "Cancer")]:
                fig.add_trace(go.Box(y=df[df[TARGET] == cls][var], name=name,
                                     marker_color=color, showlegend=(i == 0), boxmean=True),
                              row=r + 1, col=c + 1)
        fig.update_layout(height=550, title_text="Biomarqueurs : Sain vs Cancer", boxmode="group")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modèles & Résultats":
    st.title("🤖 Comparaison des 6 Modèles")

    cols = st.columns(6)
    for i, (name, row) in enumerate(RESULTS.iterrows()):
        short = name.split(" ")[0]
        badge = '<span class="best-badge">★ Meilleur</span>' if name == "K-NN" else ""
        cols[i].markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{short} {badge}</div>
          <div class="kpi-value">{row['Accuracy']:.0%}</div>
          <div style="font-size:0.75rem;color:#777">Acc | Recall: {row['Recall']:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Tableau comparatif complet")
    display_results = RESULTS.copy()
    display_results.index = [
        "K-NN ⭐ (Meilleur)" if i == "K-NN" else i for i in display_results.index
    ]
    st.dataframe(
        display_results.style
            .highlight_max(subset=["F1", "AUC", "Accuracy"], color="#c8e6c9")
            .highlight_min(subset=["FN"], color="#bbdefb")
            .format("{:.4f}", subset=["Accuracy", "Précision", "Recall", "F1", "AUC"])
            .format("{:.0f}", subset=["FN", "FP"]),
        use_container_width=True
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Métriques clés")
        fig = go.Figure()
        for metric, color in zip(["Accuracy", "Recall", "F1", "AUC"],
                                  ["#1565C0", "#E53935", "#388E3C", "#F57C00"]):
            fig.add_trace(go.Bar(name=metric, x=RESULTS.index, y=RESULTS[metric],
                                  marker_color=color, opacity=0.85))
        fig.update_layout(barmode="group", xaxis_tickangle=-30, height=380,
                          yaxis_range=[0, 1.1], legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Score global (Accuracy + F1 + AUC)")
        RESULTS["Score Global"] = (RESULTS["Accuracy"] + RESULTS["F1"] + RESULTS["AUC"]) / 3
        fig = px.bar(RESULTS.reset_index().sort_values("Score Global"),
                     x="Score Global", y="Modèle", orientation="h",
                     color="Score Global", color_continuous_scale="Blues",
                     title="Classement des modèles — score global équilibré")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("⭐ Meilleur Modèle : K-NN")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        **Logique de sélection — meilleur compromis global :**

        Le K-NN est sélectionné comme meilleur modèle car il offre le **meilleur équilibre**
        entre toutes les métriques, sans les défauts des autres modèles.

        | Critère | K-NN | Logreg / Ridge / Lasso |
        |---------|------|------------------------|
        | Accuracy | **75%** | 54% |
        | F1-score | **0.769** | 0.703 |
        | AUC-ROC | **0.825** | ≤ 0.769 |
        | FP (faux positifs) | **3** | 11 |
        | FN (faux négatifs) | 3 | 0 |

        ✅ La Régression Logistique / Ridge ont **FN = 0** mais **11 faux positifs** —
        elles classent presque tout le monde comme cancer. Le K-NN est plus fiable en pratique :
        **Accuracy 75%, F1 = 0.769, AUC = 0.825**, avec seulement 3 FN et 3 FP.
        """)
    with col2:
        for label, val, color in [("Accuracy", "75%", "#1565C0"), ("F1-score", "0.769", "#388E3C"),
                                   ("AUC-ROC", "0.825", "#F57C00"), ("FN | FP", "3 | 3", "#7B1FA2")]:
            st.markdown(f'<div class="kpi-card" style="border-left-color:{color}"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Prédiction Patient":
    st.title("🔬 Prédiction pour un Nouveau Patient")
    st.markdown("Saisissez les valeurs des biomarqueurs. Le **meilleur modèle (K-NN, Accuracy=75%, AUC=0.825)** sera appliqué automatiquement.")
    st.markdown('<div class="warning-box">⚕️ Ce résultat doit être confirmé par un professionnel de santé.</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.subheader("📝 Saisie des biomarqueurs")
        col1, col2, col3 = st.columns(3)
        defaults = df[FEATURES].median().round(2)

        with col1:
            age = st.number_input("Age (années)", 20.0, 90.0, float(defaults["Age"]), 0.5)
            bmi = st.number_input("BMI (kg/m²)", 15.0, 55.0, float(defaults["BMI"]), 0.1)
            glucose = st.number_input("Glucose (mg/dL)", 50.0, 250.0, float(defaults["Glucose"]), 0.5)
        with col2:
            insulin = st.number_input("Insulin (µU/mL)", 0.5, 80.0, float(defaults["Insulin"]), 0.1)
            homa = st.number_input("HOMA", 0.1, 20.0, float(defaults["HOMA"]), 0.1)
            leptin = st.number_input("Leptin (ng/mL)", 0.5, 120.0, float(defaults["Leptin"]), 0.5)
        with col3:
            adiponectin = st.number_input("Adiponectin (µg/mL)", 0.5, 60.0, float(defaults["Adiponectin"]), 0.1)
            resistin = st.number_input("Resistin (ng/mL)", 0.5, 60.0, float(defaults["Resistin"]), 0.1)
            mcp1 = st.number_input("MCP.1 (pg/dL)", 30.0, 800.0, float(defaults["MCP.1"]), 1.0)

        submitted = st.form_submit_button("🔍 Prédire", use_container_width=True)

    if submitted:
        patient = np.array([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp1]])
        prediction = pipeline.predict(patient)[0]

        try:
            proba = pipeline.predict_proba(patient)[0][1]
        except Exception:
            proba = None

        # Journalisation pour le monitoring de drift (Evidently) — ne bloque jamais
        # l'affichage du résultat même si l'écriture du log échoue.
        try:
            log_prediction(
                feature_values={
                    "Age": age, "BMI": bmi, "Glucose": glucose, "Insulin": insulin,
                    "HOMA": homa, "Leptin": leptin, "Adiponectin": adiponectin,
                    "Resistin": resistin, "MCP.1": mcp1,
                },
                prediction=int(prediction),
                probability=float(proba) if proba is not None else None,
            )
        except Exception as log_error:
            st.warning(f"⚠️ La prédiction n'a pas pu être journalisée pour le monitoring : {log_error}")

        st.markdown("---")
        st.subheader("🧬 Résultat de la Prédiction")
        col1, col2 = st.columns([3, 2])

        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="result-cancer">
                  <h2 style="color:#c62828">🔴 Cancer Probable</h2>
                  <p>Le modèle K-NN prédit une forte probabilité de cancer du sein.</p>
                  <p><strong>⚕️ Ce résultat doit être confirmé par un professionnel de santé.</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-sain">
                  <h2 style="color:#2e7d32">🟢 Pas de Cancer Probable</h2>
                  <p>Le modèle K-NN ne détecte pas de signe de cancer.</p>
                  <p><strong>⚕️ Ce résultat doit être confirmé par un professionnel de santé.</strong></p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            if proba is not None:
                if proba < 0.35:
                    risk_color, risk_label = "#4CAF50", "Faible"
                elif proba < 0.65:
                    risk_color, risk_label = "#FF9800", "Moyen"
                else:
                    risk_color, risk_label = "#E53935", "Élevé"

                st.markdown(f"""
                <div class="kpi-card" style="border-left-color:{risk_color}">
                  <div class="kpi-label">Probabilité de cancer</div>
                  <div class="kpi-value" style="color:{risk_color}">{proba:.1%}</div>
                </div>
                <div class="kpi-card" style="border-left-color:{risk_color}">
                  <div class="kpi-label">Niveau de risque</div>
                  <div class="kpi-value" style="color:{risk_color}">{risk_label}</div>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba * 100,
                    number={"suffix": "%"},
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Risque", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color},
                        "steps": [
                            {"range": [0, 35], "color": "#e8f5e9"},
                            {"range": [35, 65], "color": "#fff3e0"},
                            {"range": [65, 100], "color": "#ffebee"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3},
                                      "thickness": 0.75, "value": proba * 100},
                    },
                ))
                fig.update_layout(height=240, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Comparaison avec les classes du dataset")
        patient_series = pd.Series([age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp1], index=FEATURES)
        comparison = pd.DataFrame({
            "Patient": patient_series,
            "Moyenne Sain": df[df[TARGET] == 0][FEATURES].mean().round(2),
            "Moyenne Cancer": df[df[TARGET] == 1][FEATURES].mean().round(2),
        })
        st.dataframe(comparison.round(2), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MONITORING (MLOps : data drift)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🩺 Monitoring":
    st.title("🩺 Monitoring du Modèle en Production")
    st.markdown("""
    Cette page surveille la **dérive des données (data drift)** : elle compare la
    distribution des biomarqueurs saisis en production (page *Prédiction Patient*)
    à celle des données d'entraînement. Un drift important signale que le modèle
    voit des patients statistiquement différents de ceux sur lesquels il a été
    entraîné — un signal qu'un ré-entraînement pourrait être nécessaire.
    """)

    from monitoring.logger import load_predictions_log
    log_df = load_predictions_log()

    col1, col2, col3 = st.columns(3)
    col1.metric("Prédictions journalisées", len(log_df))
    col2.metric("Seuil minimal pour un rapport", 10)
    col3.metric("Cancer prédit", int(log_df["prediction"].sum()) if len(log_df) else 0)

    st.markdown("---")

    if len(log_df) == 0:
        st.info("Aucune prédiction journalisée pour le moment. Utilisez la page "
                "**🔬 Prédiction Patient** pour générer des données de production.")
    elif len(log_df) < 10:
        st.warning(f"Seulement **{len(log_df)}** prédiction(s) journalisée(s). "
                   "Au moins 10 sont recommandées pour un rapport de drift statistiquement fiable.")
        st.dataframe(log_df.tail(20), use_container_width=True)
    else:
        st.subheader("Historique des prédictions récentes")
        st.dataframe(log_df.tail(20), use_container_width=True)

        st.markdown("---")
        st.subheader("Rapport de Data Drift (Evidently)")

        if st.button("🔍 Générer / rafraîchir le rapport de drift", use_container_width=True):
            try:
                from monitoring.drift_report import build_drift_report, get_drift_summary

                with st.spinner("Génération du rapport Evidently en cours..."):
                    summary = get_drift_summary()
                    report_path = build_drift_report()

                if summary.get("dataset_drift_detected"):
                    st.error(f"🔴 Drift détecté sur {summary['n_drifted_features']}/"
                              f"{summary['n_features']} variables.")
                else:
                    st.success(f"🟢 Aucun drift significatif détecté "
                               f"({summary['n_drifted_features']}/{summary['n_features']} variables en drift).")

                with open(report_path, "r", encoding="utf-8") as f:
                    html_report = f.read()
                st.components.v1.html(html_report, height=800, scrolling=True)

                with open(report_path, "rb") as f:
                    st.download_button("⬇️ Télécharger le rapport HTML complet", f,
                                        file_name="data_drift_report.html", mime="text/html")
            except ImportError:
                st.error("Le package `evidently` n'est pas installé. "
                         "Lancez `pip install -r requirements.txt`.")
            except Exception as e:
                st.error(f"Erreur lors de la génération du rapport : {e}")

    st.markdown("---")
    st.subheader("ℹ️ Pipeline MLOps de ce projet")
    st.markdown("""
    | Étape | Outil | Où |
    |---|---|---|
    | Entraînement + tracking des expériences | **MLflow** | `train/train_model.py` |
    | Conteneurisation | **Docker** | `Dockerfile` |
    | Intégration / déploiement continu | **GitHub Actions** | `.github/workflows/ci-cd.yml` |
    | Détection de drift | **Evidently** | `monitoring/drift_report.py` |
    """)

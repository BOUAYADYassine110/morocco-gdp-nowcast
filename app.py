"""
Morocco GDP Nowcasting Platform
Master BI & Big Data Analytics - Yassine Bouayad - 2025/2026
All headline metrics are read dynamically from the CSV files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Morocco GDP Nowcasting",
    page_icon="MA",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DATA   = os.path.join(BASE, "data")

PRIMARY = "#1f4e79"
SECONDARY = "#2e75b6"
ACCENT = "#c55a11"

# ---- Custom CSS ----
st.markdown(f"""
<style>
  .main {{ background-color: #fafbfc; }}
  .block-container {{ padding-top: 2rem; }}
  h1, h2, h3 {{ color: {PRIMARY}; }}
  .metric-card {{
    background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
    color: white; padding: 20px; border-radius: 10px; text-align: center;
  }}
  .metric-value {{ font-size: 32px; font-weight: 700; }}
  .metric-label {{ font-size: 14px; opacity: 0.9; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL + DATA (cached)
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS, "svr_tuned_model.pkl"))
    sx = joblib.load(os.path.join(MODELS, "scaler_X.pkl"))
    sy = joblib.load(os.path.join(MODELS, "scaler_y.pkl"))
    with open(os.path.join(MODELS, "feature_names.json")) as f:
        feats = json.load(f)
    return model, sx, sy, feats

@st.cache_data
def load_data():
    clean = pd.read_csv(os.path.join(DATA, "Morocco_Annual_Clean.csv"))
    results = pd.read_csv(os.path.join(DATA, "model_results_tuned.csv"))
    resid = pd.read_csv(os.path.join(DATA, "residual_analysis.csv"))
    shap_i = pd.read_csv(os.path.join(DATA, "shap_interpretation.csv"))
    preds = pd.read_csv(os.path.join(DATA, "best_model_predictions.csv"))
    return clean, results, resid, shap_i, preds

MODEL, SCALER_X, SCALER_Y, FEATURES = load_model()
df_clean, df_results, df_resid, df_shap, df_preds = load_data()

SATELLITE = ["CO_mean","NDVI_mean","NightLights_std","Precip_mean","LST_mean","LST_std","LST_max"]

# ---- Dynamic headline metrics (read from CSVs, never hardcoded) ----
_best = df_results.sort_values("R2", ascending=False).iloc[0]
BEST_MODEL_NAME = _best["Model"]
BEST_R2   = float(_best["R2"])
BEST_RMSE = float(_best["RMSE_B$"])
N_FEATURES = len(FEATURES)
YEAR_MIN = int(df_clean["Year"].min())
YEAR_MAX = int(df_clean["Year"].max())

# ============================================================
# LIVE FETCH
# ============================================================
WB_MAP = {
    "GDP_Growth_Rate": "NY.GDP.MKTP.KD.ZG",
    "FDI_percent_GDP": "BX.KLT.DINV.WD.GD.ZS",
    "Gross_capital_formation_percent": "NE.GDI.TOTL.ZS",
    "Agriculture_percent_GDP": "NV.AGR.TOTL.ZS",
    "Cereal_yield_kg_per_ha": "AG.YLD.CREL.KG",
    "Inflation_rate": "FP.CPI.TOTL.ZG",
    "Manufacturing_percent_GDP": "NV.IND.MANF.ZS",
    "Food_production_index": "AG.PRD.FOOD.XD",
}

def fetch_wb(code):
    try:
        url = f"https://api.worldbank.org/v2/country/MA/indicator/{code}"
        r = requests.get(url, params={"format":"json","per_page":5,"mrv":5}, timeout=15)
        for e in r.json()[1]:
            if e["value"] is not None:
                return float(e["value"]), int(e["date"])
    except Exception:
        pass
    return None, None

def fetch_imf_current_account():
    try:
        r = requests.get("https://www.imf.org/external/datamapper/api/v1/BCA_NGDPD/MAR", timeout=15).json()
        vals = r["values"]["BCA_NGDPD"]["MAR"]
        # only use actual years (<= current year), not forecasts
        this_year = datetime.now().year
        valid = {int(y): v for y, v in vals.items() if int(y) <= this_year}
        latest = max(valid.keys())
        return float(valid[latest]), latest
    except Exception:
        return None, None

def build_live_row():
    latest = df_clean.iloc[-1].to_dict()
    row = {f: float(latest[f]) for f in FEATURES}
    src = {f: "Dernière valeur (CSV)" for f in FEATURES}
    for feat, code in WB_MAP.items():
        v, yr = fetch_wb(code)
        if v is not None:
            row[feat] = v
            src[feat] = f"Banque Mondiale ({yr})"
    v, yr = fetch_imf_current_account()
    if v is not None:
        row["Current_Account_GDP"] = v
        src["Current_Account_GDP"] = f"FMI ({yr})"
    return row, src

def predict_gdp(row):
    X = np.array([[row[f] for f in FEATURES]])
    Xs = SCALER_X.transform(X)
    ps = MODEL.predict(Xs)
    return SCALER_Y.inverse_transform(ps.reshape(-1,1)).ravel()[0]

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Morocco GDP Nowcasting")
st.sidebar.markdown("**Yassine Bouayad**")
st.sidebar.markdown("Master BI & Big Data Analytics")
st.sidebar.markdown("Faculté des Sciences El Jadida")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Vue d'ensemble",
    "Nowcast en direct",
    "Indicateurs satellitaires",
    "Résultats des modèles",
    "Interprétabilité (SHAP)",
    "Carte du Maroc"
])
st.sidebar.markdown("---")
st.sidebar.caption("Données : Google Earth Engine, Banque Mondiale, FMI, FAO, Google Trends")

# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================
if page == "Vue d'ensemble":
    st.title("Estimation du PIB du Maroc par Imagerie Satellitaire")
    st.markdown("Plateforme de nowcasting combinant télédétection et apprentissage automatique.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{BEST_R2:.2f}</div><div class="metric-label">R² ({BEST_MODEL_NAME})</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${BEST_RMSE:.1f}B</div><div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{N_FEATURES}</div><div class="metric-label">Variables</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{YEAR_MIN}-{str(YEAR_MAX)[2:]}</div><div class="metric-label">Période</div></div>', unsafe_allow_html=True)

    st.markdown("### PIB réel vs estimé")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_preds["Year"], y=df_preds["GDP_Actual_USD"]/1e9,
                             mode="lines+markers", name="PIB réel",
                             line=dict(color=PRIMARY, width=3)))
    fig.add_trace(go.Scatter(x=df_preds["Year"], y=df_preds["GDP_Predicted_USD"]/1e9,
                             mode="lines+markers", name=f"PIB estimé ({BEST_MODEL_NAME})",
                             line=dict(color=ACCENT, width=2, dash="dash")))
    fig.update_layout(height=450, xaxis_title="Année", yaxis_title="PIB (Milliards USD)",
                      plot_bgcolor="white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Le nowcasting estime le PIB présent en temps quasi réel, avant la publication "
            "officielle des statistiques par le HCP (délai habituel de plusieurs mois).")

# ============================================================
# PAGE 2 - LIVE NOWCAST
# ============================================================
elif page == "Nowcast en direct":
    st.title("Nowcast du PIB en direct")
    st.markdown("Estimation en temps réel à partir des dernières données économiques disponibles.")

    if st.button("Lancer le nowcast (données en direct)", type="primary"):
        with st.spinner("Récupération des données Banque Mondiale et FMI..."):
            row, src = build_live_row()
            gdp = predict_gdp(row)

        st.success(f"Estimation du PIB : **${gdp/1e9:.2f} Milliards USD**")

        st.markdown("### Variables utilisées")
        tbl = pd.DataFrame({
            "Variable": FEATURES,
            "Valeur": [round(row[f], 3) for f in FEATURES],
            "Source": [src[f] for f in FEATURES]
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.caption("Les indicateurs satellitaires utilisent les dernières valeurs extraites "
                   "(Google Earth Engine). Les indicateurs économiques sont récupérés en direct.")
    else:
        st.info("Cliquez sur le bouton pour lancer une estimation avec les données les plus récentes.")

# ============================================================
# PAGE 3 - SATELLITE INDICATORS
# ============================================================
elif page == "Indicateurs satellitaires":
    st.title("Indicateurs satellitaires")
    st.markdown("Signaux extraits via Google Earth Engine pour le Maroc.")

    descriptions = {
        "CO_mean": "Monoxyde de carbone (Sentinel-5P) - activité industrielle",
        "NDVI_mean": "Indice de végétation (MODIS) - activité agricole",
        "NightLights_std": "Variabilité des lumières nocturnes (VIIRS) - activité urbaine",
        "Precip_mean": "Précipitations (CHIRPS) - impact agricole",
        "LST_mean": "Température de surface moyenne (MODIS)",
        "LST_std": "Variabilité de la température de surface",
        "LST_max": "Température de surface maximale - stress de sécheresse"
    }
    indicator = st.selectbox("Choisir un indicateur", SATELLITE,
                             format_func=lambda x: descriptions.get(x, x))
    st.caption(descriptions.get(indicator, ""))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_clean["Year"], y=df_clean[indicator],
                             mode="lines+markers", line=dict(color=SECONDARY, width=2.5)))
    fig.update_layout(height=400, xaxis_title="Année", yaxis_title=indicator,
                      plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tous les indicateurs satellitaires")
    norm = df_clean[SATELLITE].apply(lambda c: (c-c.min())/(c.max()-c.min()))
    norm["Year"] = df_clean["Year"]
    fig2 = go.Figure()
    for col in SATELLITE:
        fig2.add_trace(go.Scatter(x=norm["Year"], y=norm[col], mode="lines", name=col))
    fig2.update_layout(height=400, xaxis_title="Année", yaxis_title="Valeur normalisée",
                       plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE 4 - MODEL RESULTS
# ============================================================
elif page == "Résultats des modèles":
    st.title("Résultats des modèles")

    st.markdown("### Comparaison des modèles (validation LOO-CV)")
    disp = df_results.copy().sort_values("R2", ascending=False)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    fig = px.bar(disp, x="Model", y="R2", color="R2",
                 color_continuous_scale="Blues", text="R2")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=400, plot_bgcolor="white", yaxis_title="R²")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Meilleur modèle")
    st.caption(f"{BEST_MODEL_NAME} — R² = {BEST_R2:.4f}, RMSE = ${BEST_RMSE:.2f}B")

# ============================================================
# PAGE 5 - SHAP
# ============================================================
elif page == "Interprétabilité (SHAP)":
    st.title("Interprétabilité du modèle (SHAP)")
    st.markdown("Contribution de chaque variable aux prédictions du PIB.")

    sh = df_shap.copy()
    val_col = "Mean_SHAP_B$" if "Mean_SHAP_B$" in sh.columns else sh.columns[1]
    sh = sh.sort_values(val_col, ascending=True)
    fig = px.bar(sh, x=val_col, y="Feature", orientation="h", color="Type",
                 color_discrete_map={"Satellite": ACCENT, "Economic": SECONDARY})
    fig.update_layout(height=600, plot_bgcolor="white",
                      xaxis_title="Impact moyen |SHAP| (Milliards USD)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Principaux constats")
    st.markdown("""
    - Plusieurs indicateurs satellitaires figurent parmi les variables les plus influentes.
    - Les émissions de CO et les lumières nocturnes sont des indicateurs satellitaires importants.
    - Le signal de sécheresse (température LST, NDVI) contribue fortement aux prédictions.
    - La fusion multi-sources améliore nettement la précision par rapport aux données satellitaires seules.
    """)

# ============================================================
# PAGE 6 - MAP
# ============================================================
elif page == "Carte du Maroc":
    st.title("Carte du Maroc")
    st.markdown("Visualisation géographique (niveau national).")

    fig = go.Figure(go.Choropleth(
        locations=["MAR"], z=[df_preds["GDP_Predicted_USD"].iloc[-1]/1e9],
        locationmode="ISO-3", colorscale="Blues",
        colorbar_title="PIB (Md$)"
    ))
    fig.update_layout(height=500,
                      geo=dict(scope="africa", projection_type="mercator",
                               center=dict(lat=31.8, lon=-7), showframe=False))
    st.plotly_chart(fig, use_container_width=True)

    st.info("L'analyse actuelle est au niveau national. Une extension régionale "
            "(12 régions du Maroc) constitue une perspective d'évolution du projet.")

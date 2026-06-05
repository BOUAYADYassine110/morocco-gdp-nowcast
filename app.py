"""
Morocco GDP Nowcasting Platform
Master BI & Big Data Analytics - Yassine Bouayad - 2025/2026
Modern light UI, animations, real interactive map.
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
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DATA   = os.path.join(BASE, "data")

PRIMARY = "#1f4e79"
SECONDARY = "#2e75b6"
ACCENT = "#c55a11"

# ============================================================
# MODERN UI CSS
# ============================================================
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
  .stApp {{ background: linear-gradient(180deg, #f8fafc 0%, #eef3f9 100%); }}
  .main .block-container {{ padding-top: 2.5rem; max-width: 1200px; }}

  h1 {{ color: {PRIMARY}; font-weight: 800; letter-spacing: -0.5px; }}
  h2, h3 {{ color: {PRIMARY}; font-weight: 700; }}
  p, li, label {{ color: #2d3748; }}

  section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PRIMARY} 0%, #16385a 100%);
  }}
  section[data-testid="stSidebar"] * {{ color: #eaf2fb !important; }}

  /* Animated metric cards */
  .metric-card {{
    background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
    color: white; padding: 24px 20px; border-radius: 16px; text-align: center;
    box-shadow: 0 8px 24px rgba(31,78,121,0.18);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fadeUp 0.6s ease both;
  }}
  .metric-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 14px 34px rgba(31,78,121,0.30);
  }}
  .metric-value {{ font-size: 36px; font-weight: 800; color: white; line-height: 1; }}
  .metric-label {{ font-size: 13px; opacity: 0.92; color: white; margin-top: 8px;
                   text-transform: uppercase; letter-spacing: 1px; }}

  /* Card delays for staggered entrance */
  .delay1 {{ animation-delay: 0.05s; }}
  .delay2 {{ animation-delay: 0.15s; }}
  .delay3 {{ animation-delay: 0.25s; }}
  .delay4 {{ animation-delay: 0.35s; }}

  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}

  /* Hero banner */
  .hero {{
    background: linear-gradient(120deg, {PRIMARY} 0%, {SECONDARY} 60%, {ACCENT} 140%);
    color: white; padding: 38px 40px; border-radius: 20px; margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(31,78,121,0.25);
    animation: fadeUp 0.6s ease both;
  }}
  .hero h1 {{ color: white; margin: 0 0 8px 0; font-size: 32px; }}
  .hero p {{ color: #e8f0fa; margin: 0; font-size: 16px; }}

  /* Content cards */
  .panel {{
    background: white; padding: 22px 26px; border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06); margin-bottom: 18px;
    animation: fadeUp 0.5s ease both;
  }}

  .stButton>button {{
    background: linear-gradient(135deg, {ACCENT}, #e08a3c);
    color: white; border: none; border-radius: 10px; padding: 12px 28px;
    font-weight: 600; font-size: 15px; transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 14px rgba(197,90,17,0.3);
  }}
  .stButton>button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(197,90,17,0.45); }}

  [data-testid="stMetricValue"] {{ color: {PRIMARY}; font-weight: 800; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD
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
st.sidebar.markdown("## 🇲🇦 GDP Nowcasting")
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
st.sidebar.caption("Sources : Google Earth Engine, Banque Mondiale, FMI, FAO, Google Trends")

# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================
if page == "Vue d'ensemble":
    st.markdown('<div class="hero"><h1>Estimation du PIB du Maroc par Imagerie Satellitaire</h1>'
                '<p>Plateforme de nowcasting combinant télédétection et apprentissage automatique</p></div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card delay1"><div class="metric-value">{BEST_R2:.2f}</div><div class="metric-label">R² ({BEST_MODEL_NAME})</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card delay2"><div class="metric-value">${BEST_RMSE:.1f}B</div><div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card delay3"><div class="metric-value">{N_FEATURES}</div><div class="metric-label">Variables</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card delay4"><div class="metric-value">{YEAR_MIN}-{str(YEAR_MAX)[2:]}</div><div class="metric-label">Période</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### PIB réel vs estimé")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_preds["Year"], y=df_preds["GDP_Actual_USD"]/1e9,
                             mode="lines+markers", name="PIB réel",
                             line=dict(color=PRIMARY, width=3.5),
                             marker=dict(size=8),
                             fill="tozeroy", fillcolor="rgba(31,78,121,0.06)"))
    fig.add_trace(go.Scatter(x=df_preds["Year"], y=df_preds["GDP_Predicted_USD"]/1e9,
                             mode="lines+markers", name=f"PIB estimé ({BEST_MODEL_NAME})",
                             line=dict(color=ACCENT, width=2.5, dash="dash"),
                             marker=dict(size=7)))
    fig.update_layout(height=440, xaxis_title="Année", yaxis_title="PIB (Milliards USD)",
                      plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
                      font=dict(color="#2d3748", family="Inter"),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.info("Le nowcasting estime le PIB présent en temps quasi réel, avant la publication "
            "officielle des statistiques par le HCP (délai habituel de plusieurs mois).")

# ============================================================
# PAGE 2 - LIVE NOWCAST
# ============================================================
elif page == "Nowcast en direct":
    st.markdown('<div class="hero"><h1>Nowcast du PIB en direct</h1>'
                '<p>Estimation en temps réel à partir des dernières données économiques</p></div>',
                unsafe_allow_html=True)

    if st.button("Lancer le nowcast (données en direct)", type="primary"):
        with st.spinner("Récupération des données Banque Mondiale et FMI..."):
            row, src = build_live_row()
            gdp = predict_gdp(row)

        st.markdown(f'<div class="metric-card" style="max-width:420px;margin:10px auto;">'
                    f'<div class="metric-value">${gdp/1e9:.2f}B</div>'
                    f'<div class="metric-label">PIB estimé (nowcast)</div></div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Variables utilisées")
        tbl = pd.DataFrame({
            "Variable": FEATURES,
            "Valeur": [round(row[f], 3) for f in FEATURES],
            "Source": [src[f] for f in FEATURES]
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.caption("Indicateurs satellitaires : dernières valeurs (Google Earth Engine). "
                   "Indicateurs économiques : récupérés en direct.")
    else:
        st.info("Cliquez sur le bouton pour lancer une estimation avec les données les plus récentes.")

# ============================================================
# PAGE 3 - SATELLITE
# ============================================================
elif page == "Indicateurs satellitaires":
    st.markdown('<div class="hero"><h1>Indicateurs satellitaires</h1>'
                '<p>Signaux extraits via Google Earth Engine pour le Maroc</p></div>',
                unsafe_allow_html=True)

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
                             mode="lines+markers", line=dict(color=SECONDARY, width=3),
                             marker=dict(size=7), fill="tozeroy",
                             fillcolor="rgba(46,117,182,0.08)"))
    fig.update_layout(height=400, xaxis_title="Année", yaxis_title=indicator,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(color="#2d3748", family="Inter"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tous les indicateurs (normalisés)")
    norm = df_clean[SATELLITE].apply(lambda c: (c-c.min())/(c.max()-c.min()))
    norm["Year"] = df_clean["Year"]
    fig2 = go.Figure()
    for col in SATELLITE:
        fig2.add_trace(go.Scatter(x=norm["Year"], y=norm[col], mode="lines",
                                  name=col, line=dict(width=2)))
    fig2.update_layout(height=400, xaxis_title="Année", yaxis_title="Valeur normalisée",
                       plot_bgcolor="white", paper_bgcolor="white",
                       font=dict(color="#2d3748", family="Inter"))
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE 4 - MODEL RESULTS
# ============================================================
elif page == "Résultats des modèles":
    st.markdown('<div class="hero"><h1>Résultats des modèles</h1>'
                '<p>Comparaison de six modèles par validation croisée Leave-One-Out</p></div>',
                unsafe_allow_html=True)

    disp = df_results.copy().sort_values("R2", ascending=False)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    fig = px.bar(disp, x="Model", y="R2", color="R2",
                 color_continuous_scale="Blues", text="R2")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                      yaxis_title="R²", font=dict(color="#2d3748", family="Inter"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Meilleur modèle : {BEST_MODEL_NAME} — R² = {BEST_R2:.4f}, RMSE = ${BEST_RMSE:.2f}B")

# ============================================================
# PAGE 5 - SHAP
# ============================================================
elif page == "Interprétabilité (SHAP)":
    st.markdown('<div class="hero"><h1>Interprétabilité (SHAP)</h1>'
                '<p>Contribution de chaque variable aux prédictions du PIB</p></div>',
                unsafe_allow_html=True)

    sh = df_shap.copy()
    val_col = "Mean_SHAP_B$" if "Mean_SHAP_B$" in sh.columns else sh.columns[1]
    sh = sh.sort_values(val_col, ascending=True)
    fig = px.bar(sh, x=val_col, y="Feature", orientation="h", color="Type",
                 color_discrete_map={"Satellite": ACCENT, "Economic": SECONDARY})
    fig.update_layout(height=600, plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_title="Impact moyen |SHAP| (Milliards USD)",
                      font=dict(color="#2d3748", family="Inter"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Principaux constats**
    - Plusieurs indicateurs satellitaires figurent parmi les variables les plus influentes.
    - Les émissions de CO et les lumières nocturnes sont des indicateurs satellitaires importants.
    - Le signal de sécheresse (température LST, NDVI) contribue fortement aux prédictions.
    - La fusion multi-sources améliore nettement la précision par rapport au satellite seul.
    """)

# ============================================================
# PAGE 6 - MAP (real interactive mapbox)
# ============================================================
elif page == "Carte du Maroc":
    st.markdown('<div class="hero"><h1>Carte économique du Maroc</h1>'
                '<p>Répartition régionale estimée du PIB (parts régionales HCP)</p></div>',
                unsafe_allow_html=True)

    national_gdp = df_preds["GDP_Predicted_USD"].iloc[-1] / 1e9

    regions = pd.DataFrame({
        "Région": ["Casablanca-Settat","Rabat-Salé-Kénitra","Tanger-Tétouan-Al Hoceïma",
                   "Marrakech-Safi","Fès-Meknès","Souss-Massa","Béni Mellal-Khénifra",
                   "Oriental","Drâa-Tafilalet","Laâyoune-Sakia El Hamra",
                   "Guelmim-Oued Noun","Dakhla-Oued Ed-Dahab"],
        "Part": [0.322,0.159,0.094,0.087,0.086,0.069,0.050,0.047,0.026,0.021,0.014,0.011],
        "lat": [33.57,34.02,35.76,31.63,34.04,30.42,32.34,34.68,31.93,27.15,28.99,23.68],
        "lon": [-7.59,-6.83,-5.83,-7.99,-5.00,-9.60,-6.36,-1.91,-4.42,-13.20,-10.06,-15.96],
    })
    regions["PIB (Md$)"] = (regions["Part"] * national_gdp).round(2)
    regions["Part (%)"] = (regions["Part"] * 100).round(1)

    col_map, col_kpi = st.columns([2.3, 1])

    with col_map:
        fig = px.scatter_mapbox(
            regions, lat="lat", lon="lon",
            size="Part", color="PIB (Md$)",
            color_continuous_scale="YlOrRd",
            size_max=45, zoom=4.3,
            hover_name="Région",
            hover_data={"PIB (Md$)": True, "Part (%)": True, "lat": False, "lon": False},
            center=dict(lat=29.5, lon=-8)
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            height=580, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_kpi:
        st.metric("PIB national estimé", f"${national_gdp:.1f}B")
        st.metric("Région la plus active", "Casablanca-Settat")
        st.metric("Part Casablanca-Settat", "32.2%")
        st.caption(f"Modèle : {BEST_MODEL_NAME} (R² = {BEST_R2:.2f})")

    st.markdown("### Répartition régionale du PIB estimé")
    st.dataframe(
        regions[["Région","Part (%)","PIB (Md$)"]].sort_values("PIB (Md$)", ascending=False),
        use_container_width=True, hide_index=True
    )
    st.info("Le modèle estime le PIB au niveau national. La répartition régionale applique "
            "les parts régionales publiées par le HCP. Une modélisation régionale directe "
            "constitue une perspective d'évolution du projet.")

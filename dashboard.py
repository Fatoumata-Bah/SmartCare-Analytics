"""
Dashboard PSL-CFX : Analyse COVID-19
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Smart Care Dashboard",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STYLE CSS - FOND BLANC + DESIGN PROPRE
# =============================================================================
st.markdown("""
<style>
    /* Fond blanc général */
    .stApp {
        background-color: #ffffff;
    }

    /* Réduire espace en haut de la page principale */
    .block-container {
        padding-top: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Aligner sidebar avec le contenu principal */
    section[data-testid="stSidebar"] > div {
        padding-top: 0;
        margin-top: -3rem;
    }

    /* Bordure sous le logo alignée avec le titre */
    section[data-testid="stSidebar"] [data-testid="stImage"] {
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 15px;
    }

    /* Masquer le header Streamlit par défaut */
    header[data-testid="stHeader"] {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    # Logo pleine largeur, aligné avec la ligne du titre
    st.image("logo.png", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Boutons État Normal / État d'Urgence / Recommandation
    col1, col2 = st.columns(2)

    with col1:
        btn_normal = st.button("État Normal", use_container_width=True, key="btn_normal")

    with col2:
        btn_urgence = st.button("État d'Urgence", use_container_width=True, key="btn_urgence")

    btn_reco = st.button("Recommandation", use_container_width=True, key="btn_reco")

    # Gestion de l'état des boutons
    if "mode" not in st.session_state:
        st.session_state.mode = "normal"

    if btn_normal:
        st.session_state.mode = "normal"
    if btn_urgence:
        st.session_state.mode = "urgence"
    if btn_reco:
        st.session_state.mode = "recommandation"

    # Style des boutons selon le mode actif
    mode_colors = {
        "normal": ("#3498db", "#2980b9"),
        "urgence": ("#dc3545", "#c82333"),
        "recommandation": ("#27ae60", "#219a52")
    }
    active_color, active_hover = mode_colors.get(st.session_state.mode, ("#3498db", "#2980b9"))

    st.markdown(f"""
    <style>
        [data-testid="stSidebar"] button[kind="secondary"] {{
            border: 1px solid #ccc;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filtres selon le mode
    annee = 2020
    mois = "Janvier"
    vague_select = "Vague 1 : mars 2020 - juillet 2020"
    if st.session_state.mode == "normal":
        st.markdown("<p style='color: black; font-weight: bold; font-size: 16px; margin-bottom: 10px;'>Filtres</p>", unsafe_allow_html=True)

        st.markdown("<p style='color: black; margin-bottom: 5px;'>Année</p>", unsafe_allow_html=True)
        annee = st.selectbox(
            "Année",
            options=list(range(2013, 2022)),
            index=6,
            label_visibility="collapsed"
        )

    elif st.session_state.mode == "urgence":
        st.markdown("<p style='color: black; font-weight: bold; font-size: 16px; margin-bottom: 10px;'>Filtres</p>", unsafe_allow_html=True)

        st.markdown("<p style='color: black; margin-bottom: 5px;'>Période</p>", unsafe_allow_html=True)
        vague_select = st.selectbox(
            "Vague",
            options=[
                "Vague 1 : mars 2020 - juillet 2020",
                "Vague 2 : juillet 2020 - janvier 2021",
                "Vague 3 : janvier 2021 - juillet 2021",
                "Vague 4 : juillet 2021 - septembre 2021"
            ],
            index=0,
            label_visibility="collapsed"
        )


# =============================================================================
# ZONE PRINCIPALE - HEADER
# =============================================================================

# Titre complet et visible
st.markdown(
    "<h2 style='text-align: center; color: #222222; font-weight: 600; "
    "border-bottom: 1px solid #e0e0e0; padding-bottom: 15px;'>"
    "Dashboard PSL-CFX : Analyse COVID-19</h2>",
    unsafe_allow_html=True
)

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================
@st.cache_data
def load_data():
    # Données locales PSL-CFX (capacité, séjours)
    df_local = pd.read_excel("PS_CF_data.xlsx")
    df_local['Sejours_MCO_Total'] = df_local['Sejours_MCO_Total'].fillna(
        df_local['Sejours_HC'] + df_local['Sejours_Amb']
    )

    # Tendances nationales
    df_nation = pd.read_excel("hospitalisation_capacity.xlsx")
    df_nation = df_nation.iloc[:, :5]
    df_nation.columns = ["Annee", "MCO_Complet", "MCO_Partiel", "Lits_Total", "Places_Partiel"]

    # Passages urgences (données journalières)
    df_urgences = pd.read_excel("Passages_aux_urgence_data.xlsx")
    df_urgences['Date'] = pd.to_datetime(df_urgences['Date'])
    col_passages = [c for c in df_urgences.columns if 'passages' in c.lower()][0]
    df_urgences['Passages'] = df_urgences[col_passages]
    df_urgences['Annee'] = df_urgences['Date'].dt.year
    df_urgences['Mois'] = df_urgences['Date'].dt.month

    # Données COVID
    df_covid = pd.read_excel("hospitalisation_covid.xlsx")
    df_covid['Date'] = pd.to_datetime(df_covid['Jour du début de la semaine'], unit='D', origin='1899-12-30')
    df_covid = df_covid.rename(columns={
        'Nouvelles hospitalisations': 'Hospitalisations',
        'Nouvelles entrées en soins critiques': 'SoinsCritiques',
        'Décès': 'Deces'
    })
    df_covid['Annee'] = df_covid['Date'].dt.year
    df_covid['Mois'] = df_covid['Date'].dt.month

    # Données passages période hivernale (Stat_régionale.xlsx)
    import glob
    stat_files = glob.glob('Stat*.xlsx')
    df_passages_hiver = None
    df_vagues = None
    for f in stat_files:
        if not f.startswith('~'):
            df_raw = pd.read_excel(f, sheet_name=1, skiprows=7)
            df_raw.columns = ['A', 'Periode', 'Mois', 'Passages']
            df_passages_hiver = df_raw[['Periode', 'Mois', 'Passages']].dropna()

            # Données passages par vague COVID (sheet 0)
            df_vag = pd.read_excel(f, sheet_name=0, skiprows=13)
            df_vag.columns = ['A', 'Periode', 'Departement', 'Passages', 'Passages_Ref', 'Variation']
            df_vag = df_vag[['Periode', 'Departement', 'Passages', 'Passages_Ref', 'Variation']].dropna()
            df_vag['Departement'] = df_vag['Departement'].astype(str)
            df_vagues_all = df_vag.copy()  # Tous les départements pour la carte
            df_vagues = df_vag[df_vag['Departement'].isin(['75', '94'])]
            break

    # Données impact vagues COVID
    df_covid_impact = pd.read_excel("passage_urg_vague_covid.xlsx")

    return df_local, df_nation, df_urgences, df_covid, df_passages_hiver, df_vagues, df_vagues_all, df_covid_impact

@st.cache_data
def get_capacite_reconstituee(df_nation):
    """Utilise directement les données nationales de capacité"""
    # Les données vont de 2013 à 2024
    df = df_nation.copy()
    df.columns = ['Annee', 'MCO_Complet', 'MCO_Partiel', 'Lits_Complet', 'Places_Partiel']
    return df

@st.cache_data
def get_sejours_reconstitues(df_local, df_nation):
    """Reconstitue les séjours MCO de 2012 à 2019 + prédiction SARIMA 2020-2021"""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Préparation des données nationales
    df_nat = df_nation.copy()
    df_nat.columns = ["Annee", "MCO_Complet", "MCO_Partiel", "Lits_Total", "Places_Partiel"]
    df_nat = df_nat.sort_values("Annee")
    df_nat["Var_MCO"] = df_nat["MCO_Complet"].pct_change()

    # Années de reconstitution
    years = list(range(2012, 2020))

    # Valeurs réelles connues
    known_values = {row['Annee']: row['Sejours_MCO_Total'] for _, row in df_local.iterrows() if pd.notna(row.get('Sejours_MCO_Total'))}

    # Reconstitution
    reconstructed = []
    current_val = known_values.get(2012, 10000)

    for y in years:
        if y in known_values:
            current_val = known_values[y]
        else:
            var = df_nat[df_nat['Annee'] == y]['Var_MCO'].values
            rate = var[0] if len(var) > 0 and not np.isnan(var[0]) else 0.0
            current_val = current_val * (1 + rate)
        reconstructed.append(current_val)

    # Modèle SARIMA pour prédiction 2020-2021
    try:
        model = SARIMAX(reconstructed, order=(1, 1, 0), trend='t')
        fit = model.fit(disp=False)
        forecast_obj = fit.get_forecast(steps=2)
        forecast_val = forecast_obj.predicted_mean.tolist()
        conf_int = forecast_obj.conf_int()
        conf_lower = conf_int.iloc[:, 0].tolist()
        conf_upper = conf_int.iloc[:, 1].tolist()
    except:
        # Fallback simple si SARIMA échoue
        forecast_val = [reconstructed[-1] * 1.02, reconstructed[-1] * 1.04]
        conf_lower = [v * 0.95 for v in forecast_val]
        conf_upper = [v * 1.05 for v in forecast_val]

    df_hist = pd.DataFrame({
        "Annee": years,
        "Sejours": reconstructed
    })

    df_forecast = pd.DataFrame({
        "Annee": [2020, 2021],
        "Sejours": forecast_val,
        "Conf_Lower": conf_lower,
        "Conf_Upper": conf_upper
    })

    return df_hist, df_forecast

@st.cache_data
def get_capacite_lits_reconstituee(df_local, df_nation):
    """Reconstitue la capacité en lits de 2012 à 2019 (comme dans le notebook)"""
    # Préparation des données nationales
    df_nat = df_nation.copy()
    df_nat.columns = ["Annee", "MCO_Complet", "MCO_Partiel", "Lits_Total", "Places_Partiel"]
    df_nat = df_nat.sort_values("Annee")
    df_nat["Var_Lits"] = df_nat["Lits_Total"].pct_change()

    # Années de reconstitution
    years = list(range(2012, 2020))

    # Valeurs réelles connues
    known_values = {row['Annee']: row['Lits_Total'] for _, row in df_local.iterrows() if pd.notna(row.get('Lits_Total'))}

    # Reconstitution
    reconstructed = []
    current_val = known_values.get(2012, 200)  # Valeur par défaut si 2012 manquant

    for y in years:
        if y in known_values:
            current_val = known_values[y]
        else:
            var = df_nat[df_nat['Annee'] == y]['Var_Lits'].values
            rate = var[0] if len(var) > 0 and not np.isnan(var[0]) else 0.0
            current_val = current_val * (1 + rate)
        reconstructed.append(current_val)

    df_recon = pd.DataFrame({
        "Annee": years,
        "Lits_Estimes": reconstructed
    })

    return df_recon, known_values

# Charger les données
df_local, df_nation, df_urgences, df_covid, df_passages_hiver, df_vagues, df_vagues_all, df_covid_impact = load_data()
df_capacite = get_capacite_reconstituee(df_nation)
df_lits_recon, lits_reels = get_capacite_lits_reconstituee(df_local, df_nation)
df_sejours_hist, df_sejours_forecast = get_sejours_reconstitues(df_local, df_nation)

# Mapping mois texte -> numéro
mois_map = {
    "Janvier": 1, "Février": 2, "Mars": 3, "Avril": 4,
    "Mai": 5, "Juin": 6, "Juillet": 7, "Août": 8,
    "Septembre": 9, "Octobre": 10, "Novembre": 11, "Décembre": 12
}
mois_num = mois_map[mois]

# =============================================================================
# KPIs - ÉTAT NORMAL
# =============================================================================
if st.session_state.mode == "normal":

    # Récupérer les données pour l'année sélectionnée
    cap_annee = df_capacite[df_capacite['Annee'] == annee]

    if len(cap_annee) > 0:
        lits_complet = int(cap_annee['Lits_Complet'].values[0])
        places_partiel = int(cap_annee['Places_Partiel'].values[0])
        mco_complet = int(cap_annee['MCO_Complet'].values[0])
        mco_partiel = int(cap_annee['MCO_Partiel'].values[0])
    else:
        lits_complet = 0
        places_partiel = 0
        mco_complet = 0
        mco_partiel = 0

    # Passages urgences filtrés par année et mois
    urgences_filtre = df_urgences[(df_urgences['Annee'] == annee) & (df_urgences['Mois'] == mois_num)]
    passages = int(urgences_filtre['Passages'].sum()) if len(urgences_filtre) > 0 else 0

    # Fonction formatage avec espaces
    def fmt(n):
        return f"{n:,}".replace(",", " ")

    # Affichage des 4 KPIs
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style='background-color: #e8f4fd; border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #3498db;'>
            <p style='color: #666; margin: 0; font-size: 14px;'>Lits Hospi. Complète</p>
            <p style='color: #222; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;'>{fmt(lits_complet)}</p>
            <p style='color: #888; margin: 0; font-size: 12px;'>{annee}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color: #e8f8f0; border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #27ae60;'>
            <p style='color: #666; margin: 0; font-size: 14px;'>Places Hospi. Partielle</p>
            <p style='color: #222; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;'>{fmt(places_partiel)}</p>
            <p style='color: #888; margin: 0; font-size: 12px;'>{annee}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background-color: #fef4e8; border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #f39c12;'>
            <p style='color: #666; margin: 0; font-size: 14px;'>Séjours MCO Complet</p>
            <p style='color: #222; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;'>{fmt(mco_complet)}</p>
            <p style='color: #888; margin: 0; font-size: 12px;'>{annee}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style='background-color: #f8e8f8; border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #9b59b6;'>
            <p style='color: #666; margin: 0; font-size: 14px;'>Séjours MCO Partiel</p>
            <p style='color: #222; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;'>{fmt(mco_partiel)}</p>
            <p style='color: #888; margin: 0; font-size: 12px;'>{annee}</p>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================================================
    # GRAPHIQUE - Passages aux urgences par mois (période hivernale)
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color: #222; font-weight: bold; font-size: 16px;'>Passages aux urgences par mois</p>", unsafe_allow_html=True)

    if df_passages_hiver is not None and annee <= 2019:
        # Filtrer par période selon l'année sélectionnée
        # Les périodes sont: juil. 2017 à fin juin 2018, juil. 2018 à fin juin 2019, etc.
        periode_map = {
            2017: "juil. 2017",
            2018: "juil. 2018",
            2019: "juil. 2019"
        }

        if annee in periode_map:
            periode_filtre = df_passages_hiver[df_passages_hiver['Periode'].str.contains(periode_map[annee], na=False)]

            if len(periode_filtre) > 0:
                # Ordre des mois
                mois_ordre = ['Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre',
                             'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin']

                # Nettoyer les noms de mois (enlever accents problématiques)
                periode_filtre = periode_filtre.copy()
                periode_filtre['Mois'] = periode_filtre['Mois'].str.replace('Ao�t', 'Août')
                periode_filtre['Mois'] = periode_filtre['Mois'].str.replace('D�cembre', 'Décembre')
                periode_filtre['Mois'] = periode_filtre['Mois'].str.replace('F�vrier', 'Février')

                fig = px.line(
                    periode_filtre,
                    x='Mois',
                    y='Passages',
                    markers=True,
                    title=f"Nombre de passages quotidiens moyens ({periode_map[annee]} - juin {annee+1})"
                )
                fig.update_layout(
                    xaxis_title="Mois",
                    yaxis_title="Nombre de passages",
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black', size=12),
                    title_font=dict(color='black', size=16),
                    xaxis=dict(
                        title_font=dict(color='black', size=14),
                        tickfont=dict(color='black', size=12),
                        gridcolor='#e0e0e0'
                    ),
                    yaxis=dict(
                        title_font=dict(color='black', size=14),
                        tickfont=dict(color='black', size=12),
                        gridcolor='#e0e0e0'
                    )
                )
                fig.update_traces(line_color='#3498db', marker_color='#3498db', line_width=2, marker_size=8)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Pas de données disponibles pour {annee}")
        else:
            st.info(f"Données disponibles uniquement pour 2017-2019")
    else:
        if annee > 2019:
            st.info("Graphique disponible uniquement pour les années 2017-2019 (pré-COVID)")
        else:
            st.warning("Données de passages non disponibles")

    # ==========================================================================
    # GRAPHIQUE - Capacité en Lits (2012-2019)
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color: #222; font-weight: bold; font-size: 16px;'>Reconstitution de la Capacité Structurelle (2012-2019)</p>", unsafe_allow_html=True)

    if df_lits_recon is not None and len(df_lits_recon) > 0:
        import plotly.graph_objects as go

        fig_lits = go.Figure()

        # Ligne trajectoire reconstituée (pointillés bleus)
        fig_lits.add_trace(go.Scatter(
            x=df_lits_recon["Annee"],
            y=df_lits_recon["Lits_Estimes"],
            mode='lines',
            name='Trajectoire Reconstituée',
            line=dict(color='#3498db', width=2, dash='dash')
        ))

        # Points réels (points rouges)
        annees_reelles = list(lits_reels.keys())
        valeurs_reelles = list(lits_reels.values())

        fig_lits.add_trace(go.Scatter(
            x=annees_reelles,
            y=valeurs_reelles,
            mode='markers',
            name='Points Réels (Audit)',
            marker=dict(color='#e74c3c', size=14, symbol='circle')
        ))

        fig_lits.update_layout(
            xaxis_title="Année",
            yaxis_title="Capacité en Lits",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='black')
            ),
            xaxis=dict(
                title_font=dict(color='black', size=14),
                tickfont=dict(color='black', size=12),
                gridcolor='#e0e0e0',
                dtick=1
            ),
            yaxis=dict(
                title_font=dict(color='black', size=14),
                tickfont=dict(color='black', size=12),
                gridcolor='#e0e0e0'
            )
        )

        st.plotly_chart(fig_lits, use_container_width=True)

        st.markdown("""
        <p style='color: #666; font-size: 13px; font-style: italic;'>
        Les points rouges représentent les données auditées réelles. La courbe en pointillés montre la tendance
        reconstituée à partir des variations nationales.
        </p>
        """, unsafe_allow_html=True)
    else:
        st.warning("Données de capacité non disponibles")

    # ==========================================================================
    # GRAPHIQUE - Projection Activité SARIMA (Business as Usual)
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color: #222; font-weight: bold; font-size: 16px;'>Projection de l'Activité 'Business as Usual' (SARIMA)</p>", unsafe_allow_html=True)

    if df_sejours_hist is not None and len(df_sejours_hist) > 0:
        import plotly.graph_objects as go

        fig_sarima = go.Figure()

        # Historique (ligne verte avec marqueurs)
        fig_sarima.add_trace(go.Scatter(
            x=df_sejours_hist["Annee"],
            y=df_sejours_hist["Sejours"],
            mode='lines+markers',
            name='Historique Activité',
            line=dict(color='#27ae60', width=2),
            marker=dict(color='#27ae60', size=8)
        ))

        # Prévision (ligne bleue pointillée avec marqueurs losange)
        fig_sarima.add_trace(go.Scatter(
            x=df_sejours_forecast["Annee"],
            y=df_sejours_forecast["Sejours"],
            mode='lines+markers',
            name='Prévision SARIMA (Baseline)',
            line=dict(color='#3498db', width=2, dash='dash'),
            marker=dict(color='#3498db', size=10, symbol='diamond')
        ))

        # Intervalle de confiance (zone bleue transparente)
        fig_sarima.add_trace(go.Scatter(
            x=list(df_sejours_forecast["Annee"]) + list(df_sejours_forecast["Annee"])[::-1],
            y=list(df_sejours_forecast["Conf_Upper"]) + list(df_sejours_forecast["Conf_Lower"])[::-1],
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalle de Confiance 95%',
            showlegend=True
        ))

        fig_sarima.update_layout(
            xaxis_title="Année",
            yaxis_title="Nombre de Séjours annuels",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black', size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='black')
            ),
            xaxis=dict(
                title_font=dict(color='black', size=14),
                tickfont=dict(color='black', size=12),
                gridcolor='#e0e0e0',
                dtick=1
            ),
            yaxis=dict(
                title_font=dict(color='black', size=14),
                tickfont=dict(color='black', size=12),
                gridcolor='#e0e0e0'
            )
        )

        st.plotly_chart(fig_sarima, use_container_width=True)

        st.markdown("""
        <p style='color: #666; font-size: 13px; font-style: italic;'>
        Le modèle SARIMA projette l'activité prévue pour 2020-2021 si le COVID n'avait pas existé.
        La zone bleue pâle représente l'intervalle de confiance à 95%.
        </p>
        """, unsafe_allow_html=True)
    else:
        st.warning("Données de séjours non disponibles")

# =============================================================================
# KPIs - ÉTAT D'URGENCE (COVID)
# =============================================================================
elif st.session_state.mode == "urgence":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Mapping vague sélectionnée -> clé dans les données
    vague_key_map = {
        "Vague 1 : mars 2020 - juillet 2020": "vague 1",
        "Vague 2 : juillet 2020 - janvier 2021": "vague 2",
        "Vague 3 : janvier 2021 - juillet 2021": "vague 3",
        "Vague 4 : juillet 2021 - septembre 2021": "vague 4"
    }
    vague_dates_map = {
        "vague 1": ("2020-03-02", "2020-07-06"),
        "vague 2": ("2020-07-07", "2021-01-04"),
        "vague 3": ("2021-01-05", "2021-07-05"),
        "vague 4": ("2021-07-06", "2021-09-06")
    }
    vague_label_map = {
        "vague 1": "Vague 1",
        "vague 2": "Vague 2",
        "vague 3": "Vague 3",
        "vague 4": "Vague 4"
    }

    vague_key = vague_key_map.get(vague_select, "vague 1")
    date_debut, date_fin = vague_dates_map[vague_key]
    vague_label = vague_label_map[vague_key]

    # Noms des hôpitaux
    hopital_names = {'75': 'La Salpêtrière', '94': 'Charles Foix'}

    def fmt(n):
        if isinstance(n, float):
            return f"{int(n):,}".replace(",", " ")
        return f"{n:,}".replace(",", " ")

    # --- KPIs COVID filtrés par période de la vague ---
    covid_vague = df_covid[(df_covid['Date'] >= date_debut) & (df_covid['Date'] <= date_fin)]

    if len(covid_vague) > 0:
        hospitalisations = int(covid_vague['Hospitalisations'].sum())
        soins_critiques = int(covid_vague['SoinsCritiques'].sum())
        deces = int(covid_vague['Deces'].sum())
        ratio_rea = round((soins_critiques / hospitalisations * 100), 1) if hospitalisations > 0 else 0
    else:
        hospitalisations = 0
        soins_critiques = 0
        deces = 0
        ratio_rea = 0

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style='background-color: #fdecea; border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #e74c3c;'>
            <p style='color: #666; margin: 0; font-size: 14px;'>Hospitalisations</p>
            <p style='color: #222; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;'>{fmt(hospitalisations)}</p>
            <p style='color: #888; margin: 0; font-size: 12px;'>{vague_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background-color: #fef4e8; border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #f39c12;'>
            <p style='color: #666; margin: 0; font-size: 14px;'>Soins Critiques</p>
            <p style='color: #222; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;'>{fmt(soins_critiques)}</p>
            <p style='color: #888; margin: 0; font-size: 12px;'>réanimation</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background-color: #f5f5f5; border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #555;'>
            <p style='color: #666; margin: 0; font-size: 14px;'>Décès</p>
            <p style='color: #222; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;'>{fmt(deces)}</p>
            <p style='color: #888; margin: 0; font-size: 12px;'>{vague_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style='background-color: #e8f4fd; border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #3498db;'>
            <p style='color: #666; margin: 0; font-size: 14px;'>Ratio Réanimation</p>
            <p style='color: #222; margin: 5px 0 0 0; font-size: 28px; font-weight: bold;'>{ratio_rea}%</p>
            <p style='color: #888; margin: 0; font-size: 12px;'>des hospitalisés</p>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================================================
    # GRAPHIQUE - Évolution COVID sur la vague sélectionnée
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #222; font-weight: bold; font-size: 16px;'>Évolution hebdomadaire COVID-19 - {vague_label}</p>", unsafe_allow_html=True)

    if len(covid_vague) > 0:
        fig_covid = go.Figure()
        fig_covid.add_trace(go.Scatter(x=covid_vague['Date'], y=covid_vague['Hospitalisations'],
            mode='lines+markers', name='Hospitalisations', line=dict(color='#e74c3c', width=2), marker=dict(size=6)))
        fig_covid.add_trace(go.Scatter(x=covid_vague['Date'], y=covid_vague['SoinsCritiques'],
            mode='lines+markers', name='Soins Critiques', line=dict(color='#f39c12', width=2), marker=dict(size=6)))
        fig_covid.add_trace(go.Scatter(x=covid_vague['Date'], y=covid_vague['Deces'],
            mode='lines+markers', name='Décès', line=dict(color='#555', width=2), marker=dict(size=6)))
        fig_covid.update_layout(xaxis_title="Date", yaxis_title="Nombre de cas", height=400,
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black', size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='black')),
            xaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=12), gridcolor='#e0e0e0'),
            yaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=12), gridcolor='#e0e0e0'))
        st.plotly_chart(fig_covid, use_container_width=True)
    else:
        st.info(f"Pas de données COVID disponibles pour cette période")

    # ==========================================================================
    # GRAPHIQUE - Simulation : Impact du choc COVID sur la saturation
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #222; font-weight: bold; font-size: 16px;'>Simulation : Impact du choc sur la saturation - {vague_label}</p>", unsafe_allow_html=True)

    if df_vagues is not None and len(df_vagues) > 0:
        # Variation moyenne de la vague sélectionnée (sur les 2 hôpitaux)
        vague_impact = df_vagues[df_vagues['Periode'] == vague_key]
        choc_vague = vague_impact['Variation'].mean()

        # Baseline : activité prévue 2020 (SARIMA)
        baseline_2020 = df_sejours_forecast['Sejours'].iloc[0]

        # Capacité structurelle projetée 2020 (dernière estimation * léger déclin)
        capa_lits_2020 = df_lits_recon['Lits_Estimes'].iloc[-1] * 0.99
        capacite_sejours_2020 = capa_lits_2020 * 60  # facteur rotation approximatif

        # Demande en situation de crise (la baisse des passages traduit un report/saturation)
        # On montre : demande réelle attendue vs ce que le système pouvait absorber
        demande_crise = baseline_2020 * (1 + abs(choc_vague / 100))

        fig_satu = go.Figure()

        # Barres : Capacité structurelle, Activité normale, Demande en crise
        categories = ['Capacité structurelle', 'Activité normale\n(prévision SARIMA)', f'Demande en crise\n({vague_label})']
        valeurs = [capacite_sejours_2020, baseline_2020, demande_crise]
        couleurs = ['#27ae60', '#3498db', '#e74c3c']

        fig_satu.add_trace(go.Bar(
            x=categories, y=valeurs,
            marker_color=couleurs,
            text=[fmt(int(v)) for v in valeurs],
            textposition='outside',
            textfont=dict(color='black', size=13)
        ))

        # Ligne horizontale pour la capacité max
        fig_satu.add_hline(
            y=capacite_sejours_2020,
            line_dash="dash", line_color="#27ae60", line_width=2,
            annotation_text="Seuil de capacité",
            annotation_position="top right",
            annotation_font=dict(color='#27ae60', size=12)
        )

        fig_satu.update_layout(
            yaxis_title="Volume de séjours (annualisé)", height=450,
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black', size=12),
            showlegend=False,
            xaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=12)),
            yaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=12), gridcolor='#e0e0e0'))
        st.plotly_chart(fig_satu, use_container_width=True)

        # Carte résumé saturation
        surplus = demande_crise - capacite_sejours_2020
        taux_saturation = round(demande_crise / capacite_sejours_2020 * 100, 1)
        color_satu = '#e74c3c' if taux_saturation > 110 else '#f39c12' if taux_saturation > 100 else '#27ae60'

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""
            <div style='background-color: #e8f8f0; border-radius: 10px; padding: 15px; text-align: center; border-left: 4px solid #27ae60;'>
                <p style='color: #666; margin: 0; font-size: 13px;'>Capacité structurelle</p>
                <p style='color: #222; margin: 5px 0 0 0; font-size: 22px; font-weight: bold;'>{fmt(int(capacite_sejours_2020))}</p>
                <p style='color: #888; margin: 0; font-size: 11px;'>séjours / an</p>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""
            <div style='background-color: #fdecea; border-radius: 10px; padding: 15px; text-align: center; border-left: 4px solid #e74c3c;'>
                <p style='color: #666; margin: 0; font-size: 13px;'>Demande en crise</p>
                <p style='color: #222; margin: 5px 0 0 0; font-size: 22px; font-weight: bold;'>{fmt(int(demande_crise))}</p>
                <p style='color: #888; margin: 0; font-size: 11px;'>choc {vague_label} ({choc_vague:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"""
            <div style='background-color: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; border-left: 4px solid {color_satu};'>
                <p style='color: #666; margin: 0; font-size: 13px;'>Taux de saturation</p>
                <p style='color: {color_satu}; margin: 5px 0 0 0; font-size: 22px; font-weight: bold;'>{taux_saturation}%</p>
                <p style='color: #888; margin: 0; font-size: 11px;'>surplus : {fmt(int(surplus))} séjours</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <p style='color: #666; font-size: 13px; font-style: italic; margin-top: 10px;'>
        La simulation confronte la capacité structurelle (basée sur les lits reconstitués) avec la demande
        estimée lors de la {vague_label} (impact moyen de {choc_vague:.1f}% sur les passages aux urgences).
        </p>
        """, unsafe_allow_html=True)

# =============================================================================
# RECOMMANDATION
# =============================================================================
elif st.session_state.mode == "recommandation":
    import plotly.graph_objects as go

    def fmt(n):
        if isinstance(n, float):
            return f"{int(n):,}".replace(",", " ")
        return f"{n:,}".replace(",", " ")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #222; font-weight: bold; font-size: 18px;'>
    Analyse Stratégique & Recommandations de Gestion de Crise
    </p>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # 1. Simulation 2020 : Effet Ciseaux (Capacité vs Demande)
    # ==========================================================================
    st.markdown("<p style='color: #222; font-weight: bold; font-size: 16px;'>1. Simulation 2020 : L'Effet Ciseaux</p>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #555; font-size: 14px;'>
    Confrontation entre la capacité d'accueil (en baisse structurelle) et la demande en situation de crise (en pic).
    </p>
    """, unsafe_allow_html=True)

    # Calcul du choc de demande
    if df_vagues is not None and len(df_vagues) > 0:
        choc_v1 = df_vagues[df_vagues['Periode'] == 'vague 1']['Variation'].mean()
    else:
        choc_v1 = -31.0

    impact_factor = 1 + abs(choc_v1 / 100)
    baseline_2020 = df_sejours_forecast['Sejours'].iloc[0]
    demande_crise_2020 = baseline_2020 * impact_factor

    # Capacité projetée 2020
    capa_lits_2020 = df_lits_recon['Lits_Estimes'].iloc[-1] * 0.99
    capacite_sejours_2020 = capa_lits_2020 * 60  # Facteur rotation approximatif

    fig_ciseaux = go.Figure()
    scenarios = ['Normal (Baseline)', 'Crise (Vague 1)']
    demandes = [baseline_2020, demande_crise_2020]
    capacites = [capacite_sejours_2020, capacite_sejours_2020]

    fig_ciseaux.add_trace(go.Bar(
        x=scenarios, y=demandes,
        name='Demande de Soins',
        marker_color=['#95a5a6', '#e74c3c'],
        text=[fmt(int(v)) for v in demandes],
        textposition='outside',
        textfont=dict(color='black')
    ))
    fig_ciseaux.add_trace(go.Bar(
        x=scenarios, y=capacites,
        name="Capacité d'Accueil Théorique",
        marker_color='#2ecc71',
        text=[fmt(int(v)) for v in capacites],
        textposition='outside',
        textfont=dict(color='black')
    ))
    fig_ciseaux.update_layout(
        barmode='group', yaxis_title="Volume de Séjours", height=450,
        plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black', size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='black')),
        xaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=13)),
        yaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=12), gridcolor='#e0e0e0'))
    st.plotly_chart(fig_ciseaux, use_container_width=True)

    surplus = demande_crise_2020 - baseline_2020
    st.markdown(f"""
    <div style='background-color: #fdecea; border-radius: 10px; padding: 15px; border-left: 4px solid #e74c3c;'>
        <p style='color: #222; margin: 0; font-size: 14px;'>
        <b>Surplus de demande estimé :</b> +{fmt(int(surplus))} séjours sur l'année 2020 (impact vague 1 : {choc_v1:.1f}%)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # 2. Comparaison des 4 vagues
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color: #222; font-weight: bold; font-size: 16px;'>2. Comparaison de l'impact des 4 vagues</p>", unsafe_allow_html=True)

    if df_vagues is not None and len(df_vagues) > 0:
        vagues_agg = df_vagues.groupby('Periode').agg(
            Passages=('Passages', 'sum'),
            Passages_Ref=('Passages_Ref', 'sum'),
            Variation=('Variation', 'mean')
        ).reset_index().sort_values('Periode')

        fig_comp = go.Figure()
        colors_vagues = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']

        fig_comp.add_trace(go.Bar(
            x=vagues_agg['Periode'], y=abs(vagues_agg['Variation']),
            marker_color=colors_vagues,
            text=[f"{abs(v):.1f}%" for v in vagues_agg['Variation']],
            textposition='outside',
            textfont=dict(color='black', size=13)
        ))
        fig_comp.update_layout(
            xaxis_title="Vague", yaxis_title="Baisse des passages (%)", height=400,
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black', size=12),
            showlegend=False,
            xaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=13)),
            yaxis=dict(title_font=dict(color='black', size=14), tickfont=dict(color='black', size=12), gridcolor='#e0e0e0'))
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("""
        <p style='color: #555; font-size: 13px; font-style: italic;'>
        La vague 1 a provoqué la plus forte baisse (-31%) due au confinement strict.
        La vague 4 montre une reprise quasi-normale (-5%) grâce à la vaccination et l'adaptation du système.
        </p>
        """, unsafe_allow_html=True)

    # ==========================================================================
    # 3. Recommandations stratégiques
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color: #222; font-weight: bold; font-size: 16px;'>3. Recommandations stratégiques</p>", unsafe_allow_html=True)

    recos = [
        {
            "titre": "Plan Blanc & Capacité d'Urgence",
            "icon_color": "#e74c3c",
            "desc": "Prévoir l'ouverture de lits d'urgence supplémentaires dès le déclenchement du seuil d'alerte. "
                    "La simulation montre un surplus de demande significatif lors de la vague 1 nécessitant une réponse immédiate."
        },
        {
            "titre": "Déprogrammation Ciblée",
            "icon_color": "#f39c12",
            "desc": "Mettre en place un protocole de déprogrammation des activités chirurgicales non urgentes "
                    "pour libérer des lits MCO. Prioriser selon la gravité et l'urgence des interventions."
        },
        {
            "titre": "Gestion RH & Renforts",
            "icon_color": "#3498db",
            "desc": "Anticiper le recours aux heures supplémentaires et au personnel intérimaire. "
                    "La vague 1 a montré une pression maximale sur les équipes nécessitant des renforts rapides."
        },
        {
            "titre": "Adaptation Progressive",
            "icon_color": "#27ae60",
            "desc": "L'analyse des 4 vagues montre une adaptation du système : de -31% (vague 1) à -5% (vague 4). "
                    "Capitaliser sur les protocoles mis en place pour améliorer la résilience à chaque nouvelle crise."
        }
    ]

    for reco in recos:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; border-left: 4px solid {reco["icon_color"]};'>
            <p style='color: #222; margin: 0 0 8px 0; font-size: 15px; font-weight: bold;'>{reco["titre"]}</p>
            <p style='color: #555; margin: 0; font-size: 14px;'>{reco["desc"]}</p>
        </div>
        """, unsafe_allow_html=True)

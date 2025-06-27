import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
import json
import os

# --------- UTILS ---------
@st.cache_resource
def load_model():
    try:
        with open('models/best_regression_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler_X'], data['scaler_y'], data.get('metrics', None)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None, None, None, None

def load_lottieurl(url: str, local_file: str = None):
    if local_file and os.path.exists(local_file):
        try:
            with open(local_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Erreur lors du chargement du fichier local {local_file} : {str(e)}")
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"Échec du chargement de l'animation depuis l'URL : {url}. Code de statut : {r.status_code}")
    except Exception as e:
        st.warning(f"Erreur lors du chargement de l'animation depuis l'URL : {str(e)}")
    return None

# --------- ANIMATIONS ---------
credit_animation = load_lottieurl(
    "https://lottie.host/8f7a5303-d6fc-4f53-a251-83e668af0dde/K2g0gSOKlQ.json",
    "animations/credit_animation.json"
)
loading_animation = load_lottieurl(
    "https://lottie.host/7b6e1d43-4b53-4124-90e0-2a0f10d4eaff/6q3nNaM8zT.json",
    "animations/loading_animation.json"
)
about_animation = load_lottieurl(
    "https://lottie.host/4e8f2a3b-0e32-4a98-9e1c-6e94e5d39e2a/6d5w1O8pY8.json",
    "animations/about_animation.json"
)

# --------- CONFIGURATION DE LA PAGE ---------
st.set_page_config(
    page_title="Prédicteur de Dépenses par Carte de Crédit",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------- CSS ---------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@500;600;700&display=swap');
    :root {
        --primary: #2563eb;
        --primary-dark: #1e3a8a;
        --accent: #ca8a04;
        --secondary: #6d28d9;
        --bg-light: #f8fafc;
        --bg-dark: #1f2937;
        --text-primary: #1f2937;
        --text-secondary: #4b5563;
    }
    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }
    .main {
        background: linear-gradient(135deg, var(--bg-light) 0%, #e2e8f0 100%);
        padding: 2rem;
        min-height: 100vh;
        transition: background 0.3s ease;
    }
    .dark-mode .main {
        background: linear-gradient(135deg, var(--bg-dark) 0%, #374151 100%);
    }
    .dark-mode .stMarkdown, .dark-mode .section-title-visual {
        color: #e5e7eb;
    }
    .dark-mode .section-card, .dark-mode .visual-card {
        background: rgba(31, 41, 55, 0.9);
        border: 1px solid rgba(107, 114, 128, 0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        position: relative;
        overflow: hidden;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
        background: linear-gradient(90deg, var(--primary-dark) 0%, var(--primary) 100%);
    }
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.4s ease, height 0.4s ease;
    }
    .stButton>button:hover::after {
        width: 200px;
        height: 200px;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(209, 213, 219, 0.5);
        padding: 2rem;
        border-radius: 0 16px 16px 0;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: auto;
        transition: background 0.3s ease;
    }
    .dark-mode .sidebar .sidebar-content {
        background: rgba(31, 41, 55, 0.95);
        border-right: 1px solid rgba(107, 114, 128, 0.5);
    }
    .stSelectbox, .stNumberInput {
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.9);
        padding: 0.5rem;
        backdrop-filter: blur(5px);
        position: relative;
    }
    .dark-mode .stSelectbox, .dark-mode .stNumberInput {
        background: rgba(55, 65, 81, 0.9);
    }
    .stSelectbox > div > div, .stNumberInput > div > div {
        border: 2px solid rgba(209, 213, 219, 0.5);
        border-radius: 10px;
        background: transparent;
        transition: border-color 0.3s ease;
    }
    .dark-mode .stSelectbox > div > div, .dark-mode .stNumberInput > div > div {
        border: 2px solid rgba(107, 114, 128, 0.5);
    }
    .stSelectbox > div > div:hover, .stNumberInput > div > div:hover {
        border-color: var(--primary);
    }
    .stProgress > div > div {
        background: var(--accent);
        border-radius: 10px;
    }
    .stAlert {
        border-radius: 10px;
        background: rgba(220, 252, 231, 0.9);
        color: #15803d;
        backdrop-filter: blur(5px);
        padding: 1rem;
    }
    .dark-mode .stAlert {
        background: rgba(34, 197, 94, 0.2);
        color: #34d399;
    }
    .section-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(209, 213, 219, 0.3);
    }
    .section-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    .section-title {
        font-family: 'Poppins', sans-serif;
        color: var(--primary-dark);
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        animation: zoomIn 0.5s ease-out;
    }
    @keyframes zoomIn {
        0% { transform: scale(0.95); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    .badge {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.3rem;
        display: inline-block;
        animation: pulse 2s infinite;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .about-avatar {
        border-radius: 50%;
        border: 4px solid var(--accent);
        box-shadow: 0 4px 16px rgba(202, 138, 4, 0.3);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    .about-avatar:hover {
        transform: scale(1.05);
    }
    .about-contact-btn {
        background: linear-gradient(90deg, var(--secondary) 0%, #4c1d95 100%);
        color: white;
        border-radius: 20px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(109, 40, 217, 0.3);
    }
    .about-contact-btn:hover {
        background: linear-gradient(90deg, #4c1d95 0%, var(--secondary) 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(109, 40, 217, 0.4);
    }
    .card-fade {
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.8s ease-out forwards;
    }
    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .section-sep {
        border: none;
        border-top: 1px solid rgba(209, 213, 219, 0.5);
        margin: 2rem 0;
        width: 100%;
    }
    .dark-mode .section-sep {
        border-top: 1px solid rgba(107, 114, 128, 0.5);
    }
    .section-title-visual {
        font-family: 'Poppins', sans-serif;
        font-size: 1.75rem;
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .visual-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(209, 213, 219, 0.3);
        animation: fadeInUp 0.8s ease-out;
    }
    .visual-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    .metric-card {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.02);
    }
    .metric-card h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.75rem;
        margin-bottom: 1rem;
    }
    .metric-list {
        list-style: none;
        padding: 0;
        font-size: 1rem;
    }
    .metric-list li {
        margin: 0.75rem 0;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
    }
    .metric-list li:last-child {
        border-bottom: none;
    }
    .header-container {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.3);
        position: relative;
        overflow: hidden;
        animation: zoomIn 0.5s ease-out;
    }
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1), transparent);
        z-index: 0;
    }
    .header-container > * {
        position: relative;
        z-index: 1;
    }
    .footer {
        text-align: center;
        color: var(--text-secondary);
        padding: 2rem;
        font-size: 0.9rem;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        margin-top: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    .dark-mode .footer {
        background: rgba(31, 41, 55, 0.9);
        color: #9ca3af;
    }
    .toggle-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 1rem 0;
    }
    .toggle-label {
        font-family: 'Poppins', sans-serif;
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-right: 0.5rem;
    }
    .dark-mode .toggle-label {
        color: #9ca3af;
    }
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        .section-title {
            font-size: 1.5rem;
        }
        .stButton>button {
            padding: 10px 20px;
            font-size: 0.9rem;
        }
        .sidebar .sidebar-content {
            padding: 1rem;
            border-radius: 0;
        }
        .header-container {
            padding: 2rem 1rem;
        }
        .section-card, .visual-card, .metric-card {
            padding: 1.5rem;
        }
    }
    </style>
    <script>
        function toggleDarkMode() {
            document.body.classList.toggle('dark-mode');
        }
    </script>
""", unsafe_allow_html=True)

# --------- BARRE LATÉRALE ---------
with st.sidebar:
    st.markdown("<div class='section-card' style='text-align:center; padding: 1rem;'>", unsafe_allow_html=True)
    if credit_animation:
        st_lottie(credit_animation, height=120, key="sidebar_animation")
    else:
        st.markdown("<p style='text-align:center; color:#4b5563; font-size:0.9rem;'>Animation non disponible</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title="Navigation",
        options=["Accueil", "Prédiction", "Analyse", "À Propos"],
        icons=['house-fill', 'credit-card-2-front-fill', 'graph-up', 'info-circle-fill'],
        menu_icon="menu-button-wide-fill",
        default_index=0,
        styles={
            "container": {"padding": "0.5rem", "background": "transparent"},
            "icon": {"color": "#2563eb", "font-size": "20px"},
            "nav-link": {
                "font-family": "'Poppins', sans-serif",
                "font-size": "16px",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "12px",
                "--hover-color": "rgba(37, 99, 235, 0.1)",
                "color": "#1f2937",
                "border-radius": "8px",
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #2563eb 0%, #1e3a8a 100%)",
                "color": "white",
                "border-radius": "8px",
                "font-weight": "500",
            },
        }
    )
    st.markdown("<div class='toggle-container'>", unsafe_allow_html=True)
    st.markdown("<span class='toggle-label'>Mode Sombre</span>", unsafe_allow_html=True)
    st.markdown("""
        <input type="checkbox" onchange="toggleDarkMode()" style="cursor:pointer;">
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------- ACCUEIL ---------
if selected == "Accueil":
    st.markdown("""
        <div class='header-container card-fade'>
            <h1 class='section-title' style='color:white; font-size:3rem;'>💸 Prédicteur de Dépenses</h1>
            <p style='font-size:1.2rem; margin-bottom:1rem; color:#f8fafc;'>Anticipez et optimisez les dépenses annuelles par carte de crédit avec l'intelligence artificielle.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1], gap="large")
    with col1:
        st.markdown("""
        <div class='section-card card-fade'>
            <h3 class='section-title'>🎯 Notre Mission</h3>
            <p style='color:#4b5563; font-size:1rem;'>Offrir aux professionnels de la finance une solution IA intuitive pour prévoir les dépenses annuelles des clients en fonction de leurs profils financiers et personnels.</p>
        </div>
        <div class='section-card card-fade'>
            <h3 class='section-title'>🔬 Technologies Utilisées</h3>
            <div style='display:flex; flex-wrap:wrap; gap:0.5rem;'>
                <span class='badge'>Random Forest</span>
                <span class='badge'>XGBoost</span>
                <span class='badge'>SVR</span>
                <span class='badge'>GridSearchCV</span>
                <span class='badge'>Scikit-learn</span>
                <span class='badge'>Streamlit</span>
                <span class='badge'>Plotly</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if credit_animation:
            st_lottie(credit_animation, height=220, key="main_animation")
        else:
            st.markdown("<p style='text-align:center; color:#4b5563; font-size:0.9rem;'>Animation non disponible</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card card-fade'>
            <h3>📈 Impact</h3>
            <p style='font-size:1.2rem; margin:0; font-family:"Poppins", sans-serif;'>Plus de <b>2 000</b> clients analysés</p>
        </div>
        """, unsafe_allow_html=True)

# --------- PRÉDICTION ---------
elif selected == "Prédiction":
    st.warning("Note : Le modèle a été entraîné avec une version antérieure de scikit-learn et XGBoost. Les résultats peuvent varier. Pour des prédictions optimales, re-sauvegardez le modèle avec les versions actuelles.")
    st.markdown("""
        <div class='header-container card-fade'>
            <h1 class='section-title' style='color:white;'>🔮 Prévoir les Dépenses</h1>
            <p style='font-size:1.2rem; color:#f8fafc;'>Saisissez les données du client pour estimer ses dépenses annuelles par carte de crédit.</p>
        </div>
    """, unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown("<div class='section-card card-fade'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("<h4 class='section-title'>Informations Personnelles</h4>", unsafe_allow_html=True)
            age = st.number_input("Âge", min_value=18, max_value=100, value=35, step=1, help="Âge du client")
            owner = st.selectbox("Propriétaire d'une maison", ["Non", "Oui"], help="Statut de propriété")
            selfemp = st.selectbox("Travailleur indépendant", ["Non", "Oui"], help="Statut d'emploi")
            dependents = st.number_input("Personnes à charge", min_value=0, max_value=10, value=0, step=1, help="Nombre de personnes à charge")
        with col2:
            st.markdown("<h4 class='section-title'>Informations Financières</h4>", unsafe_allow_html=True)
            income = st.number_input("Revenu annuel ($)", min_value=0, max_value=500000, value=50000, step=1000, help="Revenu annuel en dollars")
            share = st.slider("Part du revenu sur la carte (%)", min_value=0, max_value=100, value=10, help="Pourcentage du revenu dépensé via carte")
            reports = st.number_input("Rapports de crédit", min_value=0, max_value=20, value=2, step=1, help="Nombre de rapports de crédit")
            months = st.number_input("Ancienneté du compte (mois)", min_value=0, max_value=240, value=12, step=1, help="Durée du compte en mois")
            majorcards = st.number_input("Cartes principales", min_value=0, max_value=5, value=1, step=1, help="Nombre de cartes principales")
            active = st.number_input("Comptes actifs", min_value=0, max_value=10, value=2, step=1, help="Nombre de comptes actifs")
        st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
        real_expenditure = st.number_input("Dépense réelle (optionnel)", min_value=0, max_value=100000, value=0, step=100, help="Dépense réelle pour comparaison")
        submit_button = st.form_submit_button("💡 Prédire", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submit_button:
        with st.spinner("Prédiction en cours..."):
            if loading_animation:
                st_lottie(loading_animation, height=100, key="loading")
            else:
                st.markdown("<p style='text-align:center; color:#4b5563; font-size:0.9rem;'>Chargement...</p>", unsafe_allow_html=True)
            model, scaler_X, scaler_y, metrics = load_model()
            if model is None:
                st.error("Impossible de charger le modèle. Veuillez vérifier le fichier best_regression_model.pkl.")
            else:
                input_df = pd.DataFrame({
                    'income': [income],
                    'share': [share],
                    'age': [age],
                    'owner_No': [1 if owner == "Non" else 0],
                    'owner_Yes': [1 if owner == "Oui" else 0],
                    'selfemp_No': [1 if selfemp == "Non" else 0],
                    'selfemp_Yes': [1 if selfemp == "Oui" else 0],
                    'reports': [reports],
                    'dependents': [dependents],
                    'months': [months],
                    'majorcards': [majorcards],
                    'active': [active]
                })
                X_cols = scaler_X.feature_names_in_
                input_df = input_df.reindex(columns=X_cols, fill_value=0)
                X_scaled = scaler_X.transform(input_df)
                y_pred_scaled = model.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
                st.markdown("<div class='section-card card-fade'>", unsafe_allow_html=True)
                st.markdown("<h2 class='section-title' style='text-align:center;'>Résultat de la Prédiction</h2>", unsafe_allow_html=True)
                st.metric("Dépense Prédite ($)", f"{y_pred:,.2f}", delta_color="normal")
                if real_expenditure > 0:
                    st.metric("Dépense Réelle ($)", f"{real_expenditure:,.2f}", delta=f"{y_pred-real_expenditure:,.2f}")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=["Prédite", "Réelle"],
                        y=[y_pred, real_expenditure],
                        marker_color=["#2563eb", "#6d28d9"],
                        text=[f"{y_pred:,.2f}", f"{real_expenditure:,.2f}"],
                        textposition="auto"
                    ))
                    fig.update_layout(
                        title="Prédiction vs Réalité",
                        yaxis_title="Dépense ($)",
                        template="plotly_white",
                        height=350,
                        margin=dict(l=20, r=20, t=50, b=20),
                        font=dict(family="Inter, sans-serif", color="#1f2937"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

# --------- ANALYSE ---------
elif selected == "Analyse":
    st.warning("Note : Le modèle a été entraîné avec une version antérieure de scikit-learn et XGBoost. Les résultats peuvent varier. Pour des prédictions optimales, re-sauvegardez le modèle avec les versions actuelles.")
    st.markdown("""
        <div class='header-container card-fade'>
            <h1 class='section-title' style='color:white;'>📊 Tableau de Bord Analytique</h1>
            <p style='font-size:1.2rem; color:#f8fafc;'>Explorez les performances du modèle et découvrez les tendances clés des données.</p>
        </div>
    """, unsafe_allow_html=True)

    df = pd.read_csv('AER_credit_card_data.csv')
    X = df.drop(['expenditure', 'card'], axis=1)
    y = df['expenditure']
    X = pd.get_dummies(X, columns=['owner', 'selfemp'])
    model, scaler_X, scaler_y, metrics = load_model()
    if model is None:
        st.error("Impossible de charger le modèle. Veuillez vérifier le fichier best_regression_model.pkl.")
    else:
        X = X.reindex(columns=scaler_X.feature_names_in_, fill_value=0)
        X_scaled = scaler_X.transform(X)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.markdown("<div class='metric-card card-fade'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-family:Poppins, sans-serif;'>✨ Performance du Modèle</h3>", unsafe_allow_html=True)
        if metrics:
            st.markdown(f"""
                <ul class="metric-list">
                    <li><b>RMSE (test)</b>: {metrics.get('rmse', 'N/A'):.2f}</li>
                    <li><b>MAE (test)</b>: {metrics.get('mae', 'N/A'):.2f}</li>
                    <li><b>R² (test)</b>: {metrics.get('r2', 'N/A'):.3f}</li>
                    {f"<li><b>Score CV</b>: {metrics['cv_score']:.3f}</li>" if 'cv_score' in metrics else ""}
                </ul>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <p style='color:#f8fafc; font-size:0.9rem;'>(Métriques calculées sur l'ensemble des données)</p>
                <ul class="metric-list">
                    <li><b>RMSE</b>: {rmse:.2f}</li>
                    <li><b>MAE</b>: {mae:.2f}</li>
                    <li><b>R²</b>: {r2:.3f}</li>
                </ul>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr class='section-sep'/><div class='section-title-visual'>Analyse Visuelle</div>", unsafe_allow_html=True)

        # Nuage de Points
        st.markdown("""
            <div class="visual-card card-fade">
                <h4 class='section-title'>1. Prédictions vs Valeurs Réelles</h4>
                <p style='color:#4b5563; font-size:0.9rem;'>Chaque point représente un client. La proximité avec la diagonale indique une meilleure précision.</p>
        """, unsafe_allow_html=True)
        fig1 = px.scatter(
            x=y, y=y_pred,
            labels={'x': 'Valeur Réelle ($)', 'y': 'Valeur Prédite ($)'},
            color_discrete_sequence=["#2563eb"],
            template="plotly_white",
            opacity=0.7
        )
        fig1.add_shape(
            type="line",
            x0=y.min(), y0=y.min(),
            x1=y.max(), y1=y.max(),
            line=dict(color="#ca8a04", dash="dash", width=2)
        )
        fig1.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter, sans-serif", color="#1f2937"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Histogramme
        st.markdown("""
            <div class="visual-card card-fade">
                <h4 class='section-title'>2. Distribution des Dépenses</h4>
                <p style='color:#4b5563; font-size:0.9rem;'>Répartition des dépenses annuelles des clients.</p>
        """, unsafe_allow_html=True)
        fig2 = px.histogram(
            df, x="expenditure", nbins=40,
            color_discrete_sequence=["#6d28d9"],
            template="plotly_white"
        )
        fig2.update_layout(
            xaxis_title="Dépense Annuelle ($)",
            yaxis_title="Nombre de Clients",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter, sans-serif", color="#1f2937"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Importance des Variables
        st.markdown("""
            <div class="visual-card card-fade">
                <h4 class='section-title'>3. Importance des Variables</h4>
                <p style='color:#4b5563; font-size:0.9rem;'>Facteurs clés influençant les prédictions du modèle.</p>
        """, unsafe_allow_html=True)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            features = X.columns
            imp_df = pd.DataFrame({"Variable": features, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=True)
            fig3 = px.bar(
                imp_df,
                x="Importance", y="Variable",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#2563eb", "#6d28d9"],
                template="plotly_white",
                height=400
            )
            fig3.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(family="Inter, sans-serif", color="#1f2937"),
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("L'importance des variables n'est pas disponible pour ce modèle.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Boîte à Moustaches
        st.markdown("""
            <div class="visual-card card-fade">
                <h4 class='section-title'>4. Dépenses par Statut de Propriétaire</h4>
                <p style='color:#4b5563; font-size:0.9rem;'>Comparaison des dépenses selon le statut de propriété.</p>
        """, unsafe_allow_html=True)
        fig4 = px.box(
            df, x="owner", y="expenditure",
            color="owner",
            color_discrete_sequence=["#2563eb", "#6d28d9"],
            points="all",
            template="plotly_white",
            height=350
        )
        fig4.update_layout(
            xaxis_title="Statut de Propriétaire",
            yaxis_title="Dépense Annuelle ($)",
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter, sans-serif", color="#1f2937"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Matrice de Corrélation
        st.markdown("""
            <div class="visual-card card-fade">
                <h4 class='section-title'>5. Matrice de Corrélation</h4>
                <p style='color:#4b5563; font-size:0.9rem;'>Relations linéaires entre les variables du jeu de données.</p>
        """, unsafe_allow_html=True)
        corr = df.select_dtypes(include=[np.number]).corr()
        fig5 = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=["#f8fafc", "#6d28d9"],
            aspect="auto",
            template="plotly_white",
            height=450
        )
        fig5.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter, sans-serif", color="#1f2937"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --------- À PROPOS ---------
elif selected == "À Propos":
    st.markdown("""
        <div class='header-container card-fade'>
            <h1 class='section-title' style='color:white;'>À Propos</h1>
            <p style='font-size:1.2rem; color:#f8fafc;'>Découvrez le créateur et l'histoire derrière ce projet.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        if about_animation:
            st_lottie(about_animation, height=200, key="about_animation")
        else:
            st.markdown("<p style='text-align:center; color:#4b5563; font-size:0.9rem;'>Animation non disponible</p>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='text-align:center;'>
                <img src="https://avatars.githubusercontent.com/u/TheBeyonder237" class="about-avatar" style="width:160px; height:160px;" alt="Ngoue David">
                <p style='color:#4b5563; font-size:0.9rem; margin-top:0.5rem;'>Ngoue David</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align:center; margin-top:1.5rem;'>
                <button class='about-contact-btn' onclick="window.open('mailto:ngouedavidrogeryannick@gmail.com')">📧 Email</button>
                <button class='about-contact-btn' onclick="window.open('https://github.com/TheBeyonder237')">🌐 GitHub</button>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='section-card card-fade'>
                <h2 class='section-title'>Qui suis-je ?</h2>
                <p style='color:#4b5563; font-size:1rem;'>Passionné par l'IA et les données, je suis étudiant en Master IA et Big Data, développant des solutions innovantes pour la finance et la santé.</p>
                <h3 class='section-title'>Compétences</h3>
                <div style='display:flex; flex-wrap:wrap; gap:0.5rem;'>
                    <span class='badge'>Python</span>
                    <span class='badge'>Machine Learning</span>
                    <span class='badge'>Deep Learning</span>
                    <span class='badge'>NLP</span>
                    <span class='badge'>Data Science</span>
                    <span class='badge'>Cloud Computing</span>
                    <span class='badge'>Streamlit</span>
                    <span class='badge'>Scikit-learn</span>
                    <span class='badge'>XGBoost</span>
                    <span class='badge'>Pandas</span>
                    <span class='badge'>Plotly</span>
                    <span class='badge'>SQL</span>
                </div>
                <h3 class='section-title' style='margin-top:1.5rem;'>Projets Récents</h3>
                <ul style='font-size:0.95rem; color:#4b5563;'>
                    <li><b>💸 Prédicteur de Dépenses par Carte de Crédit</b> : Application IA pour anticiper les dépenses.</li>
                    <li><b>🫀 HeartGuard AI</b> : Prédiction des risques cardiaques via l'IA.</li>
                    <li><b>🔊 Multi-IA</b> : Plateforme intégrant texte, voix et traduction.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class='footer'>
            Développé avec ❤️ par Ngoue David
        </div>
    """, unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
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
    with open('models/best_regression_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler_X'], data['scaler_y'], data.get('metrics', None)

def load_lottieurl(url: str, local_file: str = None):
    # Try loading local file first
    if local_file and os.path.exists(local_file):
        try:
            with open(local_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Erreur lors du chargement du fichier local {local_file} : {str(e)}")
    
    # Fallback to URL if local file fails or doesn't exist
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
# Prioritize local files, with new URLs as fallback
credit_animation = load_lottieurl(
    "https://lottie.host/8f7a5303-d6fc-4f53-a251-83e668af0dde/K2g0gSOKlQ.json",  # Finance-related animation
    "animations/credit_animation.json"
)
loading_animation = load_lottieurl(
    "https://lottie.host/7b6e1d43-4b53-4124-90e0-2a0f10d4eaff/6q3nNaM8zT.json",  # Loading animation
    "animations/loading_animation.json"
)
about_animation = load_lottieurl(
    "https://lottie.host/4e8f2a3b-0e32-4a98-9e1c-6e94e5d39e2a/6d5w1O8pY8.json",  # Profile-related animation
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        padding: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #14b8a6 0%, #0d9488 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        border: none;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(20, 184, 166, 0.3);
        background: linear-gradient(90deg, #0d9488 0%, #14b8a6 100%);
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
    }
    .stSelectbox, .stNumberInput {
        border-radius: 8px;
        background: #f9fafb;
        padding: 0.5rem;
    }
    .stSelectbox > div > div, .stNumberInput > div > div {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background: white;
    }
    .stProgress > div > div {
        background: #14b8a6;
    }
    .stMarkdown {
        color: #1f2937;
    }
    .stAlert {
        border-radius: 8px;
        background: #f0fdf4;
        color: #166534;
    }
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .section-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        transform: translateY(-4px);
    }
    .section-title {
        color: #0d9488;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge {
        background: linear-gradient(90deg, #14b8a6 0%, #0d9488 100%);
        color: white;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.2rem;
        display: inline-block;
    }
    .about-avatar {
        border-radius: 50%;
        border: 3px solid #14b8a6;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.2);
        margin-bottom: 1rem;
    }
    .about-contact-btn {
        background: linear-gradient(90deg, #14b8a6 0%, #0d9488 100%);
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 500;
        margin: 0.3rem;
        transition: all 0.3s ease;
    }
    .about-contact-btn:hover {
        background: linear-gradient(90deg, #0d9488 0%, #14b8a6 100%);
        transform: translateY(-2px);
    }
    .card-fade {
        opacity: 0;
        transform: translateY(20px);
        animation: fadeInUp 0.6s ease-out forwards;
    }
    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .section-sep {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
        width: 100%;
    }
    .section-title-visual {
        font-size: 1.6rem;
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .visual-card {
        background: #f9fafb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .visual-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    .metric-card {
        background: linear-gradient(90deg, #14b8a6 0%, #0d9488 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        font-size: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-list {
        list-style: none;
        padding: 0;
        font-size: 1rem;
    }
    .metric-list li {
        margin: 0.5rem 0;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-list li:last-child {
        border-bottom: none;
    }
    .header-container {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    @media (max-width: 768px) {
        .section-title {
            font-size: 1.5rem;
        }
        .stButton>button {
            padding: 10px 20px;
            font-size: 0.9rem;
        }
        .sidebar .sidebar-content {
            padding: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --------- BARRE LATÉRALE ---------
with st.sidebar:
    if credit_animation:
        st_lottie(credit_animation, height=100, key="sidebar_animation")
    else:
        st.markdown("<p style='text-align:center; color:#6b7280;'>Animation non disponible</p>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title="Menu",
        options=["Accueil", "Prédiction", "Analyse", "À Propos"],
        icons=['house', 'credit-card', 'bar-chart', 'info-circle'],
        menu_icon="menu-app",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background": "transparent"},
            "icon": {"color": "#14b8a6", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "10px",
                "--hover-color": "#e6f3f2",
                "color": "#1f2937",
            },
            "nav-link-selected": {
                "background": "#14b8a6",
                "color": "white",
                "border-radius": "8px",
            },
        }
    )

# --------- ACCUEIL ---------
if selected == "Accueil":
    st.markdown("""
        <div class='header-container card-fade'>
            <h1 class='section-title' style='color:white; font-size:2.5rem;'>💸 Prédicteur de Dépenses par Carte de Crédit</h1>
            <p style='font-size:1.2rem; margin-bottom:1rem;'>Utilisez l'IA pour prédire et optimiser les dépenses annuelles de vos clients par carte de crédit.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1], gap="medium")
    with col1:
        st.markdown("""
        <div class='section-card card-fade'>
            <h3 class='section-title'>🎯 Notre Mission</h3>
            <p>Fournir aux professionnels de la finance un outil intuitif basé sur l'IA pour prédire les dépenses annuelles des clients en fonction de leurs profils financiers et personnels.</p>
        </div>
        <div class='section-card card-fade'>
            <h3 class='section-title'>🔬 Technologies</h3>
            <span class='badge'>Random Forest</span>
            <span class='badge'>XGBoost</span>
            <span class='badge'>SVR</span>
            <span class='badge'>GridSearchCV</span>
            <span class='badge'>Scikit-learn</span>
            <span class='badge'>Streamlit</span>
            <span class='badge'>Plotly</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if credit_animation:
            st_lottie(credit_animation, height=200, key="main_animation")
        else:
            st.markdown("<p style='text-align:center; color:#6b7280;'>Animation non disponible</p>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card card-fade' style='margin-top:1rem;'>
            <h3>📈 Impact</h3>
            <p style='font-size:1.1rem; margin:0;'>Plus de <b>2 000</b> clients analysés</p>
        </div>
        """, unsafe_allow_html=True)

# --------- PRÉDICTION ---------
elif selected == "Prédiction":
    st.markdown("""
        <divsubj class='header-container card-fade'>
            <h1 class='section-title' style='color:white;'>🔮 Prédiction des Dépenses</h1>
            <p style='font-size:1.1rem;'>Entrez les informations du client pour estimer ses dépenses annuelles par carte de crédit.</p>
        </div>
    """, unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown("<div class='section-card card-fade'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("<h4 class='section-title'>Informations Personnelles</h4>", unsafe_allow_html=True)
            age = st.number_input("Âge", min_value=18, max_value=100, value=35, step=1)
            owner = st.selectbox("Propriétaire d'une maison", ["Non", "Oui"])
            selfemp = st.selectbox("Travailleur indépendant", ["Non", "Oui"])
            dependents = st.number_input("Personnes à charge", min_value=0, max_value=10, value=0, step=1)
        with col2:
            st.markdown("<h4 class='section-title'>Informations Financières</h4>", unsafe_allow_html=True)
            income = st.number_input("Revenu annuel ($)", min_value=0, max_value=500000, value=50000, step=1000)
            share = st.slider("Part du revenu sur la carte (%)", min_value=0, max_value=100, value=10)
            reports = st.number_input("Rapports de crédit", min_value=0, max_value=20, value=2, step=1)
            months = st.number_input("Ancienneté du compte (mois)", min_value=0, max_value=240, value=12, step=1)
            majorcards = st.number_input("Cartes principales", min_value=0, max_value=5, value=1, step=1)
            active = st.number_input("Comptes actifs", min_value=0, max_value=10, value=2, step=1)
        st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
        real_expenditure = st.number_input("Dépense réelle (optionnel)", min_value=0, max_value=100000, value=0, step=100)
        submit_button = st.form_submit_button("💡 Prédire", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submit_button:
        with st.spinner("Prédiction en cours..."):
            if loading_animation:
                st_lottie(loading_animation, height=80, key="loading")
            else:
                st.markdown("<p style='text-align:center; color:#6b7280;'>Chargement...</p>", unsafe_allow_html=True)
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
            model, scaler_X, scaler_y, metrics = load_model()
            X_cols = scaler_X.feature_names_in_
            input_df = input_df.reindex(columns=X_cols, fill_value=0)
            X_scaled = scaler_X.transform(input_df)
            y_pred_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
            st.markdown("<div class='section-card card-fade'>", unsafe_allow_html=True)
            st.markdown("<h2 class='section-title' style='text-align:center;'>Résultat de la Prédiction</h2>", unsafe_allow_html=True)
            st.metric("Dépense Prédite ($)", f"{y_pred:,.2f}")
            if real_expenditure > 0:
                st.metric("Dépense Réelle ($)", f"{real_expenditure:,.2f}", delta=f"{y_pred-real_expenditure:,.2f}")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Prédite", "Réelle"],
                    y=[y_pred, real_expenditure],
                    marker_color=["#14b8a6", "#0d9488"]
                ))
                fig.update_layout(
                    title="Prédiction vs Réalité",
                    yaxis_title="Dépense ($)",
                    template="plotly_white",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# --------- ANALYSE ---------
elif selected == "Analyse":
    st.markdown("""
        <div class='header-container card-fade'>
            <h1 class='section-title' style='color:white;'>📊 Tableau de Bord Analytique</h1>
            <p style='font-size:1.1rem;'>Explorez les performances du modèle et les tendances clés des données.</p>
        </div>
    """, unsafe_allow_html=True)

    df = pd.read_csv('AER_credit_card_data.csv')
    X = df.drop(['expenditure', 'card'], axis=1)
    y = df['expenditure']
    X = pd.get_dummies(X, columns=['owner', 'selfemp'])
    model, scaler_X, scaler_y, metrics = load_model()
    X = X.reindex(columns=scaler_X.feature_names_in_, fill_value=0)
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    if metrics:
        st.markdown(f"""
            <div class="metric-card card-fade">
                <h3>✨ Performance du Modèle</h3>
                <ul class="metric-list">
                    <li><b>RMSE (test)</b>: {metrics.get('rmse', 'N/A'):.2f}</li>
                    <li><b>MAE (test)</b>: {metrics.get('mae', 'N/A'):.2f}</li>
                    <li><b>R² (test)</b>: {metrics.get('r2', 'N/A'):.3f}</li>
                    {f"<li><b>Score CV</b>: {metrics['cv_score']:.3f}</li>" if 'cv_score' in metrics else ""}
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="metric-card card-fade">
                <h3>✨ Performance du Modèle</h3>
                <p>(Métriques calculées sur l'ensemble des données)</p>
                <ul class="metric-list">
                    <li><b>RMSE</b>: {rmse:.2f}</li>
                    <li><b>MAE</b>: {mae:.2f}</li>
                    <li><b>R²</b>: {r2:.3f}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-sep'/><div class='section-title-visual'>Analyse Visuelle</div>", unsafe_allow_html=True)

    # Nuage de Points
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 class='section-title'>1. Prédictions vs Valeurs Réelles</h4>
            <p style='color:#4b5563;'>Chaque point représente un client. Plus les points sont proches de la diagonale, plus la prédiction est précise.</p>
    """, unsafe_allow_html=True)
    fig1 = px.scatter(
        x=y, y=y_pred,
        labels={'x': 'Valeur Réelle ($)', 'y': 'Valeur Prédite ($)'},
        color_discrete_sequence=["#14b8a6"],
        template="plotly_white"
    )
    fig1.add_shape(
        type="line",
        x0=y.min(), y0=y.min(),
        x1=y.max(), y1=y.max(),
        line=dict(color="#f43f5e", dash="dash")
    )
    fig1.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Histogramme
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 class='section-title'>2. Distribution des Dépenses</h4>
            <p style='color:#4b5563;'>Répartition des dépenses annuelles des clients.</p>
    """, unsafe_allow_html=True)
    fig2 = px.histogram(
        df, x="expenditure", nbins=40,
        color_discrete_sequence=["#0d9488"],
        template="plotly_white"
    )
    fig2.update_layout(
        xaxis_title="Dépense Annuelle ($)",
        yaxis_title="Nombre de Clients",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Importance des Variables
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 class='section-title'>3. Importance des Variables</h4>
            <p style='color:#4b5563;'>Variables les plus influentes dans le modèle de prédiction.</p>
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
            color_continuous_scale=["#14b8a6", "#0d9488"],
            template="plotly_white",
            height=350
        )
        fig3.update_layout(margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("L'importance des variables n'est pas disponible pour ce modèle.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Boîte à Moustaches
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 class='section-title'>4. Dépenses par Statut de Propriétaire</h4>
            <p style='color:#4b5563;'>Comparaison des dépenses entre propriétaires et non-propriétaires.</p>
    """, unsafe_allow_html=True)
    fig4 = px.box(
        df, x="owner", y="expenditure", 
        color="owner", 
        color_discrete_sequence=["#14b8a6", "#0d9488"],
        points="all",
        template="plotly_white",
        height=320
    )
    fig4.update_layout(
        xaxis_title="Statut de Propriétaire",
        yaxis_title="Dépense Annuelle ($)",
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Matrice de Corrélation
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 class='section-title'>5. Matrice de Corrélation</h4>
            <p style='color:#4b5563;'>Relations linéaires entre les variables du jeu de données.</p>
    """, unsafe_allow_html=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig5 = px.imshow(
        corr, 
        text_auto=".2f", 
        color_continuous_scale=["#f0f4f8", "#14b8a6"],
        aspect="auto", 
        template="plotly_white",
        height=400
    )
    fig5.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------- À PROPOS ---------
elif selected == "À Propos":
    st.markdown("""
        <div class='header-container card-fade'>
            <h1 class='section-title' style='color:white;'>À Propos</h1>
            <p style='font-size:1.1rem;'>Découvrez le créateur et le projet.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2], gap="medium")
    with col1:
        if about_animation:
            st_lottie(about_animation, height=180, key="about_animation")
        else:
            st.markdown("<p style='text-align:center; color:#6b7280;'>Animation non disponible</p>", unsafe_allow_html=True)
        st.image(
            "https://avatars.githubusercontent.com/u/TheBeyonder237",
            width=150,
            caption="Ngoue David",
            output_format="auto",
            use_column_width=False
        )
        st.markdown("""
        <div style='text-align:center; margin-top:1rem;'>
            <button class='about-contact-btn' onclick="window.open('mailto:ngouedavidrogeryannick@gmail.com')">📧 Email</button>
            <button class='about-contact-btn' onclick="window.open('https://github.com/TheBeyonder237')">🌐 GitHub</button>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='section-card card-fade'>
            <h2 class='section-title'>Qui suis-je ?</h2>
            <p>Passionné par l'IA et les données, je poursuis un Master en IA et Big Data, travaillant sur des solutions innovantes en finance et santé.</p>
            <h3 class='section-title'>Compétences</h3>
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
            <h3 class='section-title' style='margin-top:1rem;'>Projets Récents</h3>
            <ul style='font-size:0.95rem;'>
                <li><b>💸 Prédicteur de Dépenses par Carte de Crédit</b> : Application de prédiction des dépenses par IA.</li>
                <li><b>🫀 HeartGuard AI</b> : Prédiction des risques cardiaques par IA.</li>
                <li><b>🔊 Multi-IA</b> : Plateforme multi-modèles pour génération de texte, synthèse vocale et traduction.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #6b7280; padding: 1.5rem;'>
            Développé avec ❤️ par Ngoue David
        </div>
    """, unsafe_allow_html=True)
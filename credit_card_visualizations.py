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

# --------- UTILS ---------
@st.cache_resource
def load_model():
    with open('models/best_regression_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler_X'], data['scaler_y'], data.get('metrics', None)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --------- ANIMATIONS ---------
credit_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")
loading_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_p8bfn5to.json")
about_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json")

# --------- PAGE CONFIG ---------
st.set_page_config(
    page_title="Credit Card Expenditure Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------- CSS ---------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%); }
    .stButton>button {
        background: linear-gradient(45deg, #4b79a1, #283e51);
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(75, 121, 161, 0.3);
        transition: all 0.3s ease;
        font-weight: 600;
        font-size: 1.1em;
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.04);
        box-shadow: 0 6px 20px rgba(75, 121, 161, 0.4);
        background: linear-gradient(45deg, #283e51, #4b79a1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    .stSelectbox, .stNumberInput { border-radius: 10px; }
    .stProgress > div > div { background-color: #4b79a1; }
    .stMarkdown { color: #2c3e50; }
    .stAlert { border-radius: 10px; }
    .section-card {
        background: white;
        padding: 2rem;
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(76, 110, 245, 0.10);
        margin-bottom: 2rem;
        transition: box-shadow 0.3s;
    }
    .section-card:hover {
        box-shadow: 0 16px 40px 0 rgba(76, 110, 245, 0.18);
    }
    .section-title {
        color: #4b79a1;
        font-size: 2.2em;
        font-weight: 700;
        margin-bottom: 0.5em;
        text-shadow: 1px 1px 2px #e4e8eb;
    }
    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #4b79a1 0%, #283e51 100%);
        color: white;
        border-radius: 12px;
        padding: 0.3em 1em;
        font-size: 1em;
        font-weight: 600;
        margin: 0.2em 0.3em;
        box-shadow: 0 2px 8px rgba(76, 110, 245, 0.10);
    }
    .about-avatar {
        border-radius: 50%;
        border: 4px solid #4b79a1;
        box-shadow: 0 4px 16px rgba(76, 110, 245, 0.15);
        margin-bottom: 1em;
    }
    .about-contact-btn {
        background: linear-gradient(90deg, #4b79a1 0%, #283e51 100%);
        color: white;
        border-radius: 20px;
        padding: 0.5em 1.5em;
        border: none;
        font-weight: 600;
        margin: 0.5em 0.5em 0.5em 0;
        font-size: 1.1em;
        box-shadow: 0 2px 8px rgba(76, 110, 245, 0.10);
        transition: background 0.2s;
    }
    .about-contact-btn:hover {
        background: linear-gradient(90deg, #283e51 0%, #4b79a1 100%);
        color: #fff;
    }
    .card-fade {
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.8s forwards;
        animation-delay: 0.2s;
    }
    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: none;
        }
    }
    .section-sep {
        border: none;
        border-top: 2px solid #e4e8eb;
        margin: 2.5em 0 2em 0;
        width: 80%;
    }
    .section-title-visual {
        font-size: 2em;
        color: #283e51;
        font-weight: 700;
        margin-bottom: 0.7em;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px #e4e8eb;
    }
    .visual-card {
        background: linear-gradient(120deg, #fafdff 0%, #f5f7fa 100%);
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(76, 110, 245, 0.10);
        padding: 1.5em 2em 1.5em 2em;
        margin-bottom: 2.5em;
        transition: box-shadow 0.3s, transform 0.3s;
    }
    .visual-card:hover {
        box-shadow: 0 12px 32px 0 rgba(76, 110, 245, 0.18);
        transform: translateY(-4px) scale(1.01);
    }
    .metric-card {
        background: linear-gradient(90deg, #4b79a1 0%, #283e51 100%);
        color: white;
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(76, 110, 245, 0.10);
        padding: 2em 2em 1.5em 2em;
        margin-bottom: 2.5em;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .metric-card h3 {
        font-size: 1.7em;
        margin-bottom: 0.7em;
        color: #fff;
        letter-spacing: 1px;
    }
    .metric-list {
        list-style: none;
        padding: 0;
        margin: 0 auto;
        font-size: 1.15em;
    }
    .metric-list li {
        margin: 0.7em 0;
        padding: 0.5em 0;
        border-bottom: 1px solid #ffffff22;
    }
    .metric-list li:last-child {
        border-bottom: none;
    }
    </style>
""", unsafe_allow_html=True)

# --------- SIDEBAR ---------
with st.sidebar:
    st_lottie(credit_animation, height=120, key="sidebar_animation")
    selected = option_menu(
        menu_title="Navigation",
        options=["Accueil", "Prédiction", "Analyse", "À propos"],
        icons=['house', 'credit-card', 'bar-chart', 'info-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#ffffff"},
            "icon": {"color": "#4b79a1", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "padding": "10px",
                "--hover-color": "#4b79a1",
            },
            "nav-link-selected": {"background-color": "#4b79a1"},
        }
    )

# --------- ACCUEIL ---------
if selected == "Accueil":
    st.markdown("""
        <div style='
            text-align: center; 
            padding: 2.5rem 1rem 2rem 1rem; 
            background: linear-gradient(135deg, #f8fafc 0%, #e4e8eb 100%);
            border-radius: 22px; 
            margin-bottom: 36px;
            box-shadow: 0 6px 32px 0 rgba(76, 110, 245, 0.08);
        '>
            <h1 class='section-title' style='font-size:2.8em; margin-bottom:0.2em;'>💳 Credit Card Expenditure Predictor</h1>
            <p style='color: #2c3e50; font-size: 1.35em; font-weight: 400; margin-bottom:0.8em;'>
                <i>Prédisez les dépenses annuelles de vos clients grâce à l'IA, pour une gestion financière plus intelligente et personnalisée.</i>
            </p>
            <hr style='border: none; border-top: 1.5px solid #e4e8eb; width: 60%; margin: 1.5em auto 1.5em auto;'/>
            <p style='color: #4b79a1; font-size: 1.1em; max-width: 700px; margin: auto;'>
                Cette application met la puissance du machine learning au service de la finance : 
                <b>analysez, prédisez et optimisez</b> les dépenses de carte de crédit de vos clients en quelques clics.<br>
                <span style='color:#283e51;'>Pensée pour les professionnels, accessible à tous.</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        <div class='section-card' style='margin-bottom:1.5em;'>
            <h3 style='color:#4b79a1; font-size:1.3em;'>🎯 Mission</h3>
            <p style='font-size:1.08em;'>
                Offrir un outil prédictif fiable et intuitif pour anticiper les dépenses annuelles des clients, 
                en s'appuyant sur leurs caractéristiques financières et personnelles.
            </p>
        </div>
        <div class='section-card'>
            <h3 style='color:#4b79a1; font-size:1.2em;'>🔬 Technologies</h3>
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
        st_lottie(credit_animation, height=220, key="main_animation")
        st.markdown("""
        <div style='margin-top: 18px; background: linear-gradient(135deg, #4b79a1 0%, #283e51 100%); color: white; padding: 18px; border-radius: 12px; text-align: center; box-shadow: 0 2px 10px rgba(76,110,245,0.10);'>
            <h2 style='margin:0;'>+2000</h2>
            <p style='margin:0;'>Clients analysés</p>
        </div>
        """, unsafe_allow_html=True)

# --------- PREDICTION ---------
elif selected == "Prédiction":
    st.markdown("""
        <div style='
            background: linear-gradient(120deg, #f8fafc 0%, #e4e8eb 100%);
            padding: 2.2rem 1rem 2rem 1rem; 
            border-radius: 22px; 
            color: #283e51; 
            margin-bottom: 2.2rem; 
            text-align: center; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.07);
        '>
            <h1 class='section-title' style='color:#4b79a1;'>🔮 Prédiction de Dépense</h1>
            <p style='font-size:1.15em;'>Remplissez le formulaire ci-dessous pour estimer la dépense annuelle d'un client</p>
        </div>
    """, unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown("<div class='section-card' style='background:#fafdff;'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='color:#4b79a1; margin-bottom:0.5em;'>Informations personnelles</h4>", unsafe_allow_html=True)
            age = st.number_input("Âge", min_value=18, max_value=100, value=35)
            owner = st.selectbox("Propriétaire d'une maison", ["Non", "Oui"])
            selfemp = st.selectbox("Travailleur indépendant", ["Non", "Oui"])
            dependents = st.number_input("Nombre de personnes à charge", min_value=0, max_value=10, value=0)
        with col2:
            st.markdown("<h4 style='color:#4b79a1; margin-bottom:0.5em;'>Informations financières</h4>", unsafe_allow_html=True)
            income = st.number_input("Revenu annuel ($)", min_value=0, max_value=500000, value=50000)
            share = st.slider("Part de revenu allouée à la carte (%)", min_value=0, max_value=100, value=10)
            reports = st.number_input("Nombre de rapports de crédit", min_value=0, max_value=20, value=2)
            months = st.number_input("Ancienneté (mois)", min_value=0, max_value=240, value=12)
            majorcards = st.number_input("Nombre de cartes principales", min_value=0, max_value=5, value=1)
            active = st.number_input("Nombre de comptes actifs", min_value=0, max_value=10, value=2)
        st.markdown("<hr style='margin:1.5em 0;'/>", unsafe_allow_html=True)
        real_expenditure = st.number_input("Valeur réelle de la dépense (optionnel)", min_value=0, max_value=100000, value=0)
        submit_button = st.form_submit_button("💡 Prédire la dépense", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submit_button:
        with st.spinner("Prédiction en cours..."):
            st_lottie(loading_animation, height=100, key="loading")
            # Préparation des données
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
            # Charger modèle et scalers
            model, scaler_X, scaler_y, metrics = load_model()
            # Adapter les colonnes à l'ordre attendu
            X_cols = scaler_X.feature_names_in_
            input_df = input_df.reindex(columns=X_cols, fill_value=0)
            # Normaliser
            X_scaled = scaler_X.transform(input_df)
            y_pred_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
            # Affichage
            st.markdown("""
                <div class='section-card' style='margin-top:2rem;'>
                    <h2 class='section-title' style='text-align:center;'>Résultat de la Prédiction</h2>
            """, unsafe_allow_html=True)
            st.metric("Dépense prédite ($)", f"{y_pred:,.2f}")
            if real_expenditure > 0:
                st.metric("Valeur réelle ($)", f"{real_expenditure:,.2f}", delta=f"{y_pred-real_expenditure:,.2f}")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Prédiction", "Réel"],
                    y=[y_pred, real_expenditure],
                    marker_color=["#4b79a1", "#283e51"]
                ))
                fig.update_layout(title="Comparaison Prédiction vs Réel", yaxis_title="Dépense ($)")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# --------- ANALYSE ---------
elif selected == "Analyse":
    st.markdown("""
        <div style='background: linear-gradient(120deg, #283e51 0%, #4b79a1 100%);
                    padding: 2.5rem 1rem 2rem 1rem;
                    border-radius: 22px;
                    color: white;
                    margin-bottom: 2.5rem;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.10);'>
            <h1 class='section-title'>📊 Tableau de Bord Analytique</h1>
            <p style='font-size:1.2em; color:#e4e8eb;'>Explorez la performance du modèle et les tendances clés du dataset.</p>
        </div>
    """, unsafe_allow_html=True)

    # Charger le dataset et le modèle
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

    # --- Bloc Performance du Modèle ---
    if metrics:
        # On affiche uniquement les vraies métriques du best model
        st.markdown(f"""
            <div class="metric-card card-fade">
                <h3>✨ Performance du Meilleur Modèle</h3>
                <ul class="metric-list">
                    <li><b>RMSE (test)</b> : {metrics.get('rmse', 'N/A'):.2f}</li>
                    <li><b>MAE (test)</b> : {metrics.get('mae', 'N/A'):.2f}</li>
                    <li><b>R² (test)</b> : {metrics.get('r2', 'N/A'):.3f}</li>
                    {f"<li><b>Score CV</b> : {metrics['cv_score']:.3f}</li>" if 'cv_score' in metrics else ""}
                </ul>
            </div>
            <hr class="section-sep"/>
            <div class="section-title-visual">Analyse Visuelle</div>
        """, unsafe_allow_html=True)
    else:
        # fallback si jamais metrics n'est pas dispo
        st.markdown("""
            <div class="metric-card card-fade">
                <h3>✨ Performance du Modèle</h3>
                <p style="color:#fff;">(Métriques calculées sur tout le dataset, à titre indicatif)</p>
                <ul class="metric-list">
                    <li><b>RMSE</b> : {:.2f}</li>
                    <li><b>MAE</b> : {:.2f}</li>
                    <li><b>R²</b> : {:.3f}</li>
                </ul>
            </div>
            <hr class="section-sep"/>
            <div class="section-title-visual">Analyse Visuelle</div>
        """.format(rmse, mae, r2), unsafe_allow_html=True)

    # --- 1. Scatter plot Prédiction vs Réel ---
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 style='color:#4b79a1; margin-bottom:0.3em;'>1. Prédictions vs Valeurs réelles</h4>
            <p style='color:#444; margin-bottom:1.2em;'>Chaque point représente un client. Plus les points sont proches de la diagonale, plus la prédiction est précise.</p>
    """, unsafe_allow_html=True)
    fig1 = px.scatter(
        x=y, y=y_pred,
        labels={'x': 'Valeur réelle', 'y': 'Prédiction'},
        color_discrete_sequence=["#4b79a1"],
        title=None
    )
    fig1.add_shape(
        type="line",
        x0=y.min(), y0=y.min(),
        x1=y.max(), y1=y.max(),
        line=dict(color="#e74c3c", dash="dash")
    )
    fig1.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 2. Distribution des Dépenses Réelles ---
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 style='color:#4b79a1; margin-bottom:0.3em;'>2. Distribution des dépenses réelles</h4>
            <p style='color:#444; margin-bottom:1.2em;'>Visualisation de la répartition des dépenses annuelles des clients.</p>
    """, unsafe_allow_html=True)
    fig2 = px.histogram(df, x="expenditure", nbins=40, color_discrete_sequence=["#283e51"])
    fig2.update_layout(
        xaxis_title="Dépense annuelle ($)",
        yaxis_title="Nombre de clients",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 3. Importance des variables ---
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 style='color:#4b79a1; margin-bottom:0.3em;'>3. Importance des variables</h4>
            <p style='color:#444; margin-bottom:1.2em;'>Les variables les plus influentes dans la prédiction selon le modèle.</p>
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
            color_continuous_scale="blues",
            height=350
        )
        fig3.update_layout(margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("L'importance des variables n'est pas disponible pour ce modèle.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 4. Dépense moyenne par statut de propriétaire ---
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 style='color:#4b79a1; margin-bottom:0.3em;'>4. Dépense moyenne selon le statut de propriétaire</h4>
            <p style='color:#444; margin-bottom:1.2em;'>Comparaison des dépenses annuelles entre propriétaires et non-propriétaires.</p>
    """, unsafe_allow_html=True)
    fig4 = px.box(
        df, x="owner", y="expenditure", 
        color="owner", 
        color_discrete_sequence=["#4b79a1", "#283e51"],
        points="all",
        height=320
    )
    fig4.update_layout(
        xaxis_title="Statut de propriétaire",
        yaxis_title="Dépense annuelle ($)",
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 5. Matrice de corrélation ---
    st.markdown("""
        <div class="visual-card card-fade">
            <h4 style='color:#4b79a1; margin-bottom:0.3em;'>5. Corrélation entre variables</h4>
            <p style='color:#444; margin-bottom:1.2em;'>Les relations linéaires entre les principales variables du dataset.</p>
    """, unsafe_allow_html=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig5 = px.imshow(
        corr, 
        text_auto=True, 
        color_continuous_scale="blues", 
        aspect="auto", 
        height=400
    )
    fig5.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------- A PROPOS ---------
elif selected == "À propos":
    st.markdown("""
        <div style='background: linear-gradient(120deg, #4b79a1 0%, #283e51 100%); padding: 2rem; border-radius: 18px; color: white; margin-bottom: 2rem; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h1 class='section-title' style='color:white;'>À propos</h1>
            <p>Découvrez le créateur, le projet et les technologies utilisées</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st_lottie(about_animation, height=220, key="about_animation")
        st.image(
            "https://avatars.githubusercontent.com/u/TheBeyonder237",
            width=180,
            caption="Ngoue David",
            output_format="auto",
            use_column_width=False,
            channels="RGB"
        )
        st.markdown("""
            <div style='text-align:center; margin-top:1em;'>
                <button class='about-contact-btn' onclick="window.open('mailto:ngouedavidrogeryannick@gmail.com')">📧 Email</button>
                <button class='about-contact-btn' onclick="window.open('https://github.com/TheBeyonder237')">🌐 GitHub</button>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='section-card'>
            <h2 class='section-title'>Qui suis-je ?</h2>
            <p>
                Je suis un passionné de l'intelligence artificielle et de la donnée.<br>
                Actuellement en Master 2 en IA et Big Data, je travaille sur des solutions innovantes dans le domaine de l'Intelligence Artificielle appliquée à la finance et à la santé.
            </p>
            <h3 style='color:#4b79a1;'>Compétences</h3>
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
            <h3 style='color:#4b79a1; margin-top:1.5em;'>Projets Récents</h3>
            <ul>
                <li><b>💳 Credit Card Expenditure Predictor</b> : Application de prédiction de dépenses de carte de crédit.</li>
                <li><b>🫀 HeartGuard AI</b> : Prédiction de risques cardiaques par IA.</li>
                <li><b>🔊 Multi-IA</b> : Plateforme multi-modèles pour la génération de texte, synthèse vocale et traduction.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
        Développé avec ❤️ par Ngoue David
        </div>
    """, unsafe_allow_html=True)
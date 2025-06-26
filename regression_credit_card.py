import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv('AER_credit_card_data.csv')

# Préparation des données
X = df.drop(['expenditure', 'card'], axis=1)  # Features
y = df['expenditure']  # Target variable

# Conversion des variables catégorielles
X = pd.get_dummies(X, columns=['owner', 'selfemp'])

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
# StandardScaler pour les features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# MinMaxScaler pour la target variable (expenditure)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# Définition des modèles et leurs paramètres
models = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    }
}

def evaluate_regression_model(y_true, y_pred, model_name):
    """Évaluation complète d'un modèle de régression"""
    # Métriques
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nRésultats pour {model_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def perform_grid_search():
    results = {}
    
    for name, model_info in models.items():
        print(f"\nEntraînement du modèle {name}...")
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Entraînement
        grid_search.fit(X_train_scaled, y_train_scaled)
        
        # Meilleurs paramètres
        print(f"\nMeilleurs paramètres pour {name}:")
        print(grid_search.best_params_)
        
        # Prédictions
        y_pred_scaled = grid_search.predict(X_test_scaled)
        
        # Conversion des prédictions à l'échelle originale
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Évaluation
        metrics = evaluate_regression_model(y_test, y_pred, name)
        
        # Stockage des résultats
        results[name] = {
            'best_params': grid_search.best_params_,
            'metrics': metrics,
            'best_model': grid_search.best_estimator_,
            'predictions': y_pred
        }
    
    return results

# Exécution de GridSearchCV
print("Début de l'entraînement des modèles...")
results = perform_grid_search()

# Visualisation des résultats
def plot_regression_results(results):
    # Préparation des données pour le graphique
    metrics = ['rmse', 'mae', 'r2']
    models = list(results.keys())
    
    # Création du graphique
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results[model]['metrics'][metric] for model in models]
        if metric == 'r2':  # R² doit être sur une échelle de 0 à 1
            axes[i].set_ylim(0, 1)
        sns.barplot(x=models, y=values, ax=axes[i])
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_xticklabels(models, rotation=45)
        
        # Ajout des valeurs sur les barres
        for j, v in enumerate(values):
            axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('regression_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Visualisation des prédictions vs valeurs réelles
def plot_predictions_vs_actual(results):
    plt.figure(figsize=(15, 5))
    
    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(1, 3, i)
        plt.scatter(y_test, result['predictions'], alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f'{name} - Prédictions vs Réelles')
        plt.xlabel('Valeurs réelles')
        plt.ylabel('Prédictions')
    
    plt.tight_layout()
    plt.savefig('regression_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Affichage des résultats
plot_regression_results(results)
plot_predictions_vs_actual(results)

# Sauvegarde du meilleur modèle
import pickle
best_model_name = max(results, key=lambda x: results[x]['metrics']['r2'])
best_model = results[best_model_name]['best_model']

# Création du dossier models s'il n'existe pas
import os
if not os.path.exists('models'):
    os.makedirs('models')

# Sauvegarde du modèle et des scalers
with open('models/best_regression_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, f)

print(f"\nMeilleur modèle: {best_model_name}")
print(f"R²: {results[best_model_name]['metrics']['r2']:.4f}")
print("Modèle et scalers sauvegardés avec succès!") 
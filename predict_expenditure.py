import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_model_and_scalers():
    """Charge le modèle et les scalers sauvegardés"""
    try:
        with open('models/best_regression_model.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        return saved_data['model'], saved_data['scaler_X'], saved_data['scaler_y']
    except FileNotFoundError:
        raise Exception("Le modèle n'a pas été trouvé. Veuillez d'abord exécuter regression_credit_card.py")

def prepare_input_data(data):
    """Prépare les données d'entrée pour la prédiction"""
    # Vérification des colonnes requises
    required_columns = ['reports', 'age', 'income', 'share', 'owner', 'selfemp', 
                       'dependents', 'months', 'majorcards', 'active']
    
    # Vérification des colonnes manquantes
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes: {missing_columns}")
    
    # Conversion des variables catégorielles
    data = pd.get_dummies(data, columns=['owner', 'selfemp'])
    
    # Vérification des valeurs manquantes
    if data.isnull().any().any():
        raise ValueError("Les données contiennent des valeurs manquantes")
    
    return data

def predict_expenditure(input_data):
    """
    Fait des prédictions sur de nouvelles données
    
    Parameters:
    -----------
    input_data : pandas.DataFrame
        DataFrame contenant les données d'entrée avec les colonnes requises
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant les données d'entrée et les prédictions
    """
    # Chargement du modèle et des scalers
    model, scaler_X, scaler_y = load_model_and_scalers()
    
    # Préparation des données
    prepared_data = prepare_input_data(input_data)
    
    # Normalisation des features
    X_scaled = scaler_X.transform(prepared_data)
    
    # Prédiction
    y_pred_scaled = model.predict(X_scaled)
    
    # Conversion des prédictions à l'échelle originale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Ajout des prédictions au DataFrame
    result_df = input_data.copy()
    result_df['predicted_expenditure'] = y_pred
    
    return result_df

# Exemple d'utilisation
if __name__ == "__main__":
    # Création d'un exemple de données
    example_data = pd.DataFrame({
        'reports': [0, 1, 2],
        'age': [35, 45, 25],
        'income': [4.5, 3.2, 2.8],
        'share': [0.05, 0.03, 0.04],
        'owner': ['yes', 'no', 'yes'],
        'selfemp': ['no', 'yes', 'no'],
        'dependents': [2, 1, 3],
        'months': [36, 24, 48],
        'majorcards': [1, 1, 0],
        'active': [12, 8, 15]
    })
    
    try:
        # Prédiction
        results = predict_expenditure(example_data)
        
        # Affichage des résultats
        print("\nRésultats des prédictions:")
        print("-------------------------")
        for idx, row in results.iterrows():
            print(f"\nExemple {idx + 1}:")
            print(f"Âge: {row['age']}")
            print(f"Revenu: {row['income']}")
            print(f"Propriétaire: {row['owner']}")
            print(f"Prédiction de dépenses: {row['predicted_expenditure']:.2f}")
        
        # Sauvegarde des résultats
        results.to_csv('predictions_results.csv', index=False)
        print("\nLes résultats ont été sauvegardés dans 'predictions_results.csv'")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")

def predict_single_example(age, income, owner='yes', selfemp='no', reports=0, 
                         share=0.05, dependents=0, months=36, majorcards=1, active=12):
    """
    Fonction utilitaire pour prédire les dépenses pour un seul exemple
    
    Parameters:
    -----------
    age : float
        Âge de la personne
    income : float
        Revenu
    owner : str
        Statut de propriétaire ('yes' ou 'no')
    selfemp : str
        Statut d'emploi indépendant ('yes' ou 'no')
    reports : int
        Nombre de rapports
    share : float
        Part de revenu
    dependents : int
        Nombre de personnes à charge
    months : int
        Nombre de mois
    majorcards : int
        Nombre de cartes majeures
    active : int
        Nombre de cartes actives
    
    Returns:
    --------
    float
        Prédiction des dépenses
    """
    # Création du DataFrame pour un seul exemple
    data = pd.DataFrame({
        'reports': [reports],
        'age': [age],
        'income': [income],
        'share': [share],
        'owner': [owner],
        'selfemp': [selfemp],
        'dependents': [dependents],
        'months': [months],
        'majorcards': [majorcards],
        'active': [active]
    })
    
    # Prédiction
    results = predict_expenditure(data)
    return results['predicted_expenditure'].iloc[0]

# Exemple d'utilisation de predict_single_example
if __name__ == "__main__":
    print("\nExemple de prédiction pour un seul cas:")
    print("--------------------------------------")
    prediction = predict_single_example(
        age=35,
        income=4.5,
        owner='yes',
        selfemp='no',
        dependents=2
    )
    print(f"Prédiction de dépenses: {prediction:.2f}") 
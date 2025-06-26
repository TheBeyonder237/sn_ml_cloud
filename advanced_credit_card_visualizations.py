import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Lire les données
df = pd.read_csv('AER_credit_card_data.csv')

# Configuration du style
plt.style.use('seaborn')
sns.set_palette("husl")

# 1. Analyse de la distribution des dépenses avec KDE et rug plot
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='expenditure', hue='card', common_norm=False)
sns.rugplot(data=df, x='expenditure', hue='card', alpha=0.3)
plt.title('Distribution des dépenses avec estimation de densité')
plt.xlabel('Dépenses')
plt.ylabel('Densité')
plt.savefig('expenditure_kde.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Analyse des quartiles et outliers des dépenses par âge
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x=pd.qcut(df['age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']), y='expenditure', hue='card')
plt.title('Distribution des dépenses par quartile d\'âge')
plt.xlabel('Quartile d\'âge')
plt.ylabel('Dépenses')
plt.savefig('expenditure_by_age_quartiles.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Analyse de la relation revenu-dépenses avec régression
plt.figure(figsize=(12, 6))
sns.regplot(data=df[df['card'] == 'yes'], x='income', y='expenditure', 
            scatter_kws={'alpha':0.3}, line_kws={'color': 'red'})
plt.title('Relation revenu-dépenses avec ligne de régression (détenteurs de cartes)')
plt.xlabel('Revenu')
plt.ylabel('Dépenses')
plt.savefig('income_expenditure_regression.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Analyse des composantes principales (PCA)
numeric_cols = ['age', 'income', 'share', 'expenditure', 'dependents', 'months', 'active']
X = df[numeric_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['card'].map({'yes': 1, 'no': 0}), alpha=0.6)
plt.title('Analyse en Composantes Principales (ACP)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.colorbar(label='Possession de carte')
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Heatmap des corrélations avec clustering hiérarchique
plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_cols].corr()
sns.clustermap(correlation_matrix, 
               cmap='coolwarm', 
               center=0,
               annot=True,
               fmt='.2f',
               figsize=(12, 8))
plt.title('Matrice de corrélation avec clustering hiérarchique')
plt.savefig('correlation_clustermap.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Analyse des dépenses par statut de propriétaire et nombre de dépendants
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='dependents', y='expenditure', hue='owner')
plt.title('Distribution des dépenses par nombre de dépendants et statut de propriétaire')
plt.xlabel('Nombre de dépendants')
plt.ylabel('Dépenses')
plt.savefig('expenditure_by_dependents_owner.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Analyse de la distribution des rapports de crédit avec violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='reports', y='expenditure', hue='card', split=True)
plt.title('Distribution des dépenses par nombre de rapports de crédit')
plt.xlabel('Nombre de rapports')
plt.ylabel('Dépenses')
plt.savefig('reports_violin.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Analyse des cartes actives par tranche d'âge
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                         labels=['18-25', '26-35', '36-45', '46-55', '55+'])
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='age_group', y='active', hue='card')
plt.title('Nombre de cartes actives par tranche d\'âge')
plt.xlabel('Tranche d\'âge')
plt.ylabel('Nombre de cartes actives')
plt.savefig('active_cards_by_age_group.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Analyse de la part de revenu vs dépenses avec hexbin
plt.figure(figsize=(12, 6))
plt.hexbin(df['share'], df['expenditure'], gridsize=30, cmap='YlOrRd')
plt.colorbar(label='Nombre d\'observations')
plt.title('Distribution de la part de revenu vs dépenses (Hexbin)')
plt.xlabel('Part de revenu')
plt.ylabel('Dépenses')
plt.savefig('share_expenditure_hexbin.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. Analyse des statistiques descriptives par groupe
stats_by_card = df.groupby('card')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
plt.figure(figsize=(15, 8))
sns.heatmap(stats_by_card, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Statistiques descriptives par statut de carte')
plt.savefig('descriptive_stats.png', dpi=300, bbox_inches='tight')
plt.close() 
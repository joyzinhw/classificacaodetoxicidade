import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Carrega dados
df = pd.read_csv('dataset.csv')

# Define X removendo colunas não numéricas e irrelevantes para o modelo
X = df.drop(columns=['name', 'CID', 'CAS', 'SMILES', 'source', 'toxicity_type', 'label', 'year'])

# Variável alvo
y = df['label']

# Treina modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Importância das features
importances = model.feature_importances_

feat_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Importância das features:")
print(feat_importance_df)

# Seleção das features importantes (threshold 0.01)
selector = SelectFromModel(model, threshold=0.01, prefit=True)

X_important = selector.transform(X)
important_feature_names = X.columns[selector.get_support()]

print(f"\nFeatures importantes selecionadas ({len(important_feature_names)}):")
print(important_feature_names.tolist())

# Seleciona as 3 features mais importantes
top3_features = feat_importance_df['feature'].head(2).tolist()

# Cria novo DataFrame com as 3 features + label
df_top3 = df[top3_features].copy()
df_top3['label'] = y.values

# Salva no CSV
df_top3.to_csv('importances.csv', index=False)

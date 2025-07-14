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

# Novo DataFrame só com features importantes + label
df_important = pd.DataFrame(X_important, columns=important_feature_names)
df_important['label'] = y.values

# Salva novo CSV
df_important.to_csv('importances.csv', index=False)

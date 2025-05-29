import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

df = pd.read_csv('dataset_final.csv')
X = df.drop(['name', 'CID', 'CAS', 'SMILES', 'source', 'toxicity_type', 'label'], axis=1)
y = df['label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

selector = SelectFromModel(model, threshold=0.01, prefit=True)
X_important = selector.transform(X)
important_features = X.columns[selector.get_support()]

pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_important)

df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
df_pca['label'] = y.values
df_pca.to_csv('dataset_pca_3_components.csv', index=False)

print(important_features.tolist())
print("Dataset PCA criado e salvo.")

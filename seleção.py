import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv('dataset_final.csv')

X = df.drop(['name', 'CID', 'CAS', 'SMILES', 'source', 'toxicity_type', 'label'], axis=1)
y = df['label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importances = model.feature_importances_

feat_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Importância das features:")
print(feat_importance_df)

selector = SelectFromModel(model, threshold=0.01, prefit=True)

X_important = selector.transform(X)
important_feature_names = X.columns[selector.get_support()]

print(f"\nFeatures importantes selecionadas ({len(important_feature_names)}):")
print(important_feature_names.tolist())

df_important = pd.DataFrame(X_important, columns=important_feature_names)
df_important['label'] = y.values

df_important.to_csv('dataset_features_importantes.csv', index=False)


# ===========================
# Imports
# ===========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ===========================
# 1. Carregar Dataset
# ===========================
df = pd.read_csv('dataset_final.csv')

# Converter SMILES em fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X = np.array([smiles_to_fingerprint(sm) for sm in df['SMILES']])
y = df['label'].values

# Remover entradas inválidas
valid_indices = [i for i, x in enumerate(X) if x is not None]
X = np.array([X[i] for i in valid_indices])
y = y[valid_indices]

# ===========================
# 2. Split e Escalonamento
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Distribuição de classes no treino:", np.bincount(y_train))

# ===========================
# 3. Modelos + Otimização
# ===========================
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(max_iter=500, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

param_grid = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'poly']
    },
    'MLP': {
        'hidden_layer_sizes': [(100,), (100,50)],
        'alpha': [0.0001, 0.001],
        'activation': ['relu', 'tanh']
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'max_depth': [-1, 10],
        'learning_rate': [0.01, 0.1]
    },
    'CatBoost': {
        'iterations': [100, 200],
        'depth': [4, 6],
        'learning_rate': [0.01, 0.1]
    }
}

# ===========================
# 4. Função de Avaliação (retorna métricas)
# ===========================
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Não Tóxico', 'Tóxico']))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Tóxico', 'Tóxico'],
                yticklabels=['Não Tóxico', 'Tóxico'])
    plt.title(f'Matriz de Confusão - {name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.show()

    return {
        'Modelo': name,
        'Acurácia': accuracy,
        'Precisão': precision,
        'Recall': recall,
        'F1-score': f1,
        'AUC-ROC': auc,
        'MCC': mcc
    }

# ===========================
# 5. Treinar, Avaliar e armazenar resultados
# ===========================
results = []

for name, model in models.items():
    print(f'\n🔍 Otimizando {name}...')
    search = RandomizedSearchCV(model, param_distributions=param_grid[name],
                                n_iter=5, cv=3, scoring='f1', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, name)
    results.append(metrics)

# ===========================
# 6. Comparação Final entre Modelos
# ===========================
results_df = pd.DataFrame(results)
print("\n=== Comparação entre modelos ===")
print(results_df.sort_values(by='F1-score', ascending=False))

# Gráfico comparativo das métricas F1, AUC-ROC e MCC
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df.melt(id_vars='Modelo', value_vars=['F1-score', 'AUC-ROC', 'MCC']),
            x='Modelo', y='value', hue='variable')
plt.title('Comparação de Métricas entre Modelos')
plt.ylabel('Valor da Métrica')
plt.xticks(rotation=45)
plt.legend(title='Métrica')
plt.show()

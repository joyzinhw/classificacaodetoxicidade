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
from imblearn.over_sampling import SMOTE
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


df = pd.read_csv('dataset.csv')

# Converter SMILES em fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X = np.array([smiles_to_fingerprint(sm) for sm in df['SMILES']])
y = df['label'].values


valid_indices = [i for i, x in enumerate(X) if x is not None]
X = np.array([X[i] for i in valid_indices])
y = y[valid_indices]


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


def get_metrics(y_test, y_pred, y_proba):
    return {
        'Acurácia': accuracy_score(y_test, y_pred),
        'Precisão': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }


all_results = []

for name, model in models.items():
    print(f'\n🔍 Otimizando {name}...')
    
    metrics_runs = {'Acurácia': [], 'Precisão': [], 'Recall': [], 
                    'F1-Score': [], 'AUC-ROC': [], 'MCC': []}

    for run in range(5):
        print(f'🏃‍♂️ Rodada {run+1}/5 para {name}')
        
        # Split e Escalonamento
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + run, stratify=y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Balanceamento com SMOTE
        sm = SMOTE(random_state=42 + run)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

        # Otimização
        search = RandomizedSearchCV(model, param_distributions=param_grid[name],
                                    n_iter=5, cv=3, scoring='f1', random_state=42 + run, n_jobs=-1)
        search.fit(X_train_bal, y_train_bal)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = get_metrics(y_test, y_pred, y_proba)

        for metric_name, metric_value in metrics.items():
            metrics_runs[metric_name].append(metric_value)

    # calcula média e desvio padrão
    result_summary = {'Modelo': name}
    for metric_name, values in metrics_runs.items():
        result_summary[f'{metric_name} Média'] = np.mean(values)
        result_summary[f'{metric_name} Desvio'] = np.std(values)
    
    all_results.append(result_summary)


df_results = pd.DataFrame(all_results)
print("\n=== Comparação resumida dos modelos ===")
print(df_results)


df_plot = df_results.set_index('Modelo')[[col for col in df_results.columns if 'Média' in col]]
df_plot.plot(kind='bar', figsize=(12,6), yerr=df_results[[col for col in df_results.columns if 'Desvio' in col]].values.T, capsize=4)

plt.title('Comparação de métricas dos modelos (médias e desvios)')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Criação do DataFrame com os resultados
df_results = pd.DataFrame(all_results)

# Salvar em arquivo CSV
df_results.to_csv('resultados_modelos.csv', index=False)

# Exibir resultados no terminal
print("\n=== Comparação resumida dos modelos ===")
print(df_results)

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import (accuracy_score, precision_score, recall_score,
#                              f1_score, roc_auc_score, matthews_corrcoef)
# from imblearn.over_sampling import SMOTE
# from rdkit import Chem
# from rdkit.Chem import AllChem, DataStructs
# import matplotlib.pyplot as plt
# import seaborn as sns

# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# # ===========================
# # 1. Carregar dataset
# # ===========================
# df = pd.read_csv('dataset_final.csv')

# # ===========================
# # 2. Função para converter SMILES em fingerprint
# # ===========================
# def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
#     arr = np.zeros((n_bits,), dtype=int)
#     DataStructs.ConvertToNumpyArray(fp, arr)
#     return arr

# # ===========================
# # 3. Gerar X e y
# # ===========================
# X = np.array([smiles_to_fingerprint(sm) for sm in df['SMILES']])
# y = df['label'].values

# # Remover entradas inválidas
# valid_indices = [i for i, x in enumerate(X) if x is not None]
# X = np.array([X[i] for i in valid_indices])
# y = y[valid_indices]

# # ===========================
# # 4. Modelos
# # ===========================
# models = {
#     'Random Forest': RandomForestClassifier(random_state=42),
#     'SVM': SVC(probability=True, random_state=42),
#     'MLP': MLPClassifier(max_iter=500, random_state=42),
#     'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
#     'LightGBM': LGBMClassifier(random_state=42),
#     'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
# }

# # ===========================
# # 5. Função de métricas
# # ===========================
# def get_metrics(y_test, y_pred, y_proba):
#     return {
#         'Acurácia': accuracy_score(y_test, y_pred),
#         'Precisão': precision_score(y_test, y_pred),
#         'Recall': recall_score(y_test, y_pred),
#         'F1-Score': f1_score(y_test, y_pred),
#         'AUC-ROC': roc_auc_score(y_test, y_proba),
#         'MCC': matthews_corrcoef(y_test, y_pred)
#     }

# # ===========================
# # 6. Avaliação dos modelos
# # ===========================
# all_results = []

# for name, model in models.items():
#     print(f'\n🔍 Avaliando {name}...')
    
#     metrics_runs = {'Acurácia': [], 'Precisão': [], 'Recall': [], 
#                     'F1-Score': [], 'AUC-ROC': [], 'MCC': []}

#     for run in range(5):
#         print(f'🏃‍♂️ Rodada {run+1}/5 para {name}')
        
#         # Split e Escalonamento
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42 + run, stratify=y)

#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

#         # Balanceamento com SMOTE
#         sm = SMOTE(random_state=42 + run)
#         X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

#         # Treinamento direto, sem otimização
#         model.fit(X_train_bal, y_train_bal)

#         y_pred = model.predict(X_test)
#         y_proba = model.predict_proba(X_test)[:, 1]

#         metrics = get_metrics(y_test, y_pred, y_proba)

#         for metric_name, metric_value in metrics.items():
#             metrics_runs[metric_name].append(metric_value)

#     # Após 5 execuções, calcula média e desvio padrão
#     result_summary = {'Modelo': name}
#     for metric_name, values in metrics_runs.items():
#         result_summary[f'{metric_name} Média'] = np.mean(values)
#         result_summary[f'{metric_name} Desvio'] = np.std(values)
    
#     all_results.append(result_summary)

# # ===========================
# # 7. Comparação Resumida
# # ===========================
# df_results = pd.DataFrame(all_results)
# print("\n=== Comparação resumida dos modelos ===")
# print(df_results)

# # ===========================
# # 8. Gráfico comparativo
# # ===========================
# df_plot = df_results.set_index('Modelo')[[col for col in df_results.columns if 'Média' in col]]
# df_plot.plot(kind='bar', figsize=(12,6), 
#              yerr=df_results[[col for col in df_results.columns if 'Desvio' in col]].values.T, 
#              capsize=4)

# plt.title('Comparação de métricas dos modelos (médias e desvios)')
# plt.ylabel('Score')
# plt.ylim(0, 1)
# plt.xticks(rotation=45)
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.show()

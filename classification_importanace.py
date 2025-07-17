import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('importances.csv')
features = df.drop(columns=['label']).columns.tolist()
y = df['label'].values
results = []

for feature in features:
    print(f"\n🔍 Avaliando feature: {feature}")
    
    X = df[[feature]].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    print("Antes do SMOTE:", np.bincount(y_train))
    print("Depois do SMOTE:", np.bincount(y_train_bal))

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n=== {feature} ===")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Não Tóxico', 'Tóxico']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Tóxico', 'Tóxico'],
                yticklabels=['Não Tóxico', 'Tóxico'])
    plt.title(f'Matriz de Confusão - {feature}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.savefig(f'matriz_confusao_{feature}.png')
    plt.close()

    results.append({
        'Feature': feature,
        'Acurácia': acc,
        'Precisão': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'MCC': mcc
    })

df_results = pd.DataFrame(results)
print("\n=== Comparação entre features (usando Random Forest) ===")
print(df_results)

metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC', 'MCC']
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    sns.barplot(
        x=df_results['Feature'],
        y=df_results[metric],
        ax=axes[i],
        palette='viridis'
    )
    axes[i].set_title(metric)
    axes[i].set_ylim(0, 1)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Score')
    for p in axes[i].patches:
        height = p.get_height()
        axes[i].annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2, height),
                         ha='center', va='bottom', fontsize=10)

plt.suptitle('Comparação detalhada de métricas por Feature (Random Forest)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('grafico_comparacao_features.png')
plt.close()

# informações do dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset_final.csv')

print("Informações do dataset:\n")
print(df.info(), "\n")

print("Informações das colunas do dataset:\n")
print(df.head(8))


print("\nEstatísticas descritivas:\n")
print(df.describe(), "\n")


# verificar se precisa limpar os dados
print("Verfica se existe valores nulos", "\n")    
print(df.isnull(), "\n")

print("Verfica se existe duplicatas", "\n")
print(df.duplicated(), "\n")

print("Proporção classes binárias:", df['label'].value_counts())


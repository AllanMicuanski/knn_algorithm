"""
===============================================================================
                    ALGORITMO K-NEAREST NEIGHBORS (KNN)
                    CLASSIFICAÇÃO DO DATASET IRIS
===============================================================================

DESCRIÇÃO DA TAREFA:
-------------------
Este script implementa o algoritmo KNN para classificar flores do dataset Iris.
O dataset contém 150 instâncias de 3 tipos de flores (50 de cada tipo):
- Iris-setosa
- Iris-versicolor 
- Iris-virginica

Cada flor é descrita por 4 atributos (em centímetros):
- sepal_length (comprimento da sépala)
- sepal_width (largura da sépala)
- petal_length (comprimento da pétala)
- petal_width (largura da pétala)

OBJETIVO:
---------
Treinar um modelo KNN que consiga prever o tipo de uma flor baseado
nas medidas de suas sépalas e pétalas.

MÉTRICA DE AVALIAÇÃO:
--------------------
Acurácia: porcentagem de previsões corretas em relação ao total de previsões.

"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# ETAPA 1: CARREGAR E PREPARAR OS DADOS
# ============================================================================

# 1️⃣ Carregar o dataset do arquivo
# O arquivo contém 150 linhas, cada uma com 4 medidas + 1 classe
print("📂 Carregando dataset...")
df = pd.read_csv("dataset-iris.txt", header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
print(f"   ✅ {len(df)} instâncias carregadas")

# 2️⃣ Separar atributos (X) e classes (y)
# X = matriz com as 4 medidas de cada flor (features)
# y = vetor com o tipo de cada flor (target/classes)
print("\n🔢 Separando atributos e classes...")
X = df.iloc[:, :-1].values  # Primeiras 4 colunas (atributos)
y = df.iloc[:, -1].values   # Última coluna (classes)
print(f"   ✅ X: {X.shape} (instâncias x atributos)")
print(f"   ✅ y: {len(y)} classes")

# 3️⃣ Dividir em conjuntos de treino (70%) e teste (30%)
# Treino: dados que o algoritmo usa para "aprender"
# Teste: dados que usamos para avaliar se o algoritmo aprendeu bem
print("\n✂️  Dividindo dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"   ✅ Treino: {len(X_train)} instâncias (70%)")
print(f"   ✅ Teste: {len(X_test)} instâncias (30%)")

# 4️⃣ Normalizar os dados (colocar todas as medidas na mesma escala)
# Importante porque as medidas têm escalas diferentes (ex: 1.0 vs 5.0)
# Sem normalização, medidas maiores dominam o cálculo da distância
print("\n📏 Normalizando os dados...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Aprende a escala com dados de treino
X_test = scaler.transform(X_test)        # Aplica a mesma escala nos dados de teste
print("   ✅ Dados normalizados (média=0, desvio=1)")

# ============================================================================
# ETAPA 2: IMPLEMENTAR O ALGORITMO KNN
# ============================================================================

def euclidean_distance(a, b):
    """
    Calcula a distância euclidiana entre dois pontos.
    
    É como medir a distância "em linha reta" entre duas flores
    baseado em suas características (sépalas e pétalas).
    
    Quanto MENOR a distância, mais PARECIDAS são as flores!
    """
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test_instance, k):
    """
    Classifica UMA flor usando o algoritmo KNN.
    
    PASSO A PASSO:
    1. Calcula distância da flor para TODAS as flores de treino
    2. Encontra as K flores mais parecidas (menores distâncias)  
    3. Vê qual tipo de flor é mais comum entre essas K vizinhas
    4. Essa é a previsão! (votação majoritária)
    """
    # Calcula distância para todas as flores de treino
    distances = [euclidean_distance(X_test_instance, x_train) for x_train in X_train]
    
    # Encontra os índices das K menores distâncias
    k_indices = np.argsort(distances)[:k]
    
    # Pega as classes das K flores mais próximas  
    k_nearest_labels = [y_train[i] for i in k_indices]
    
    # Votação: qual classe aparece mais vezes?
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

def knn(X_train, y_train, X_test, k):
    """
    Classifica MÚLTIPLAS flores de uma vez.
    
    Aplica o KNN para cada flor do conjunto de teste.
    """
    return [knn_predict(X_train, y_train, x_test, k) for x_test in X_test]

# ============================================================================
# ETAPA 3: TREINAR E AVALIAR O MODELO
# ============================================================================

# 6️⃣ Configurar e executar o KNN
# k=5 significa que vamos olhar os 5 vizinhos mais próximos para decidir
print("\n🤖 Executando o algoritmo KNN...")
k = 5
print(f"   🔢 Usando k = {k} vizinhos")

# Fazer previsões para todas as flores do conjunto de teste
y_pred = knn(X_train, y_train, X_test, k)
print(f"   ✅ {len(y_pred)} previsões realizadas")

# 7️⃣ Calcular a acurácia (métrica de avaliação)
def accuracy(y_true, y_pred):
    """
    Calcula a acurácia: quantas previsões estavam corretas?
    
    Fórmula: acertos / total_de_previsões
    Resultado: valor entre 0.0 (0%) e 1.0 (100%)
    """
    return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)

print("\n📊 Avaliando o desempenho do modelo...")
acc = accuracy(y_test, y_pred)

# ============================================================================
# ETAPA 4: EXIBIR OS RESULTADOS
# ============================================================================

print("\n" + "="*50)
print("🎯 RESULTADOS DA CLASSIFICAÇÃO KNN")
print("="*50)
print(f"Número de vizinhos (k): {k}")
print(f"Acurácia do modelo: {acc:.2f} ({acc*100:.2f}%)")
print("-"*50)

print(f"\n📝 Exemplos de previsões:")
print(f"{'Real':<15} | {'Previsto':<15} | Status")
print("-"*45)
for real, pred in zip(y_test[:10], y_pred[:10]):
    status = "✅ Correto" if real == pred else "❌ Erro" 
    print(f"{real:<15} | {pred:<15} | {status}")

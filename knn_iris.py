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
# DEFINIÇÃO DAS FUNÇÕES DO ALGORITMO KNN
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

def accuracy(y_true, y_pred):
    """
    Calcula a acurácia: quantas previsões estavam corretas?
    
    Fórmula: acertos / total_de_previsões
    Resultado: valor entre 0.0 (0%) e 1.0 (100%)
    """
    return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)

# ============================================================================
# FUNÇÃO PRINCIPAL - EXECUÇÃO DO PROGRAMA
# ============================================================================

def main():
    """
    Função principal que executa todo o pipeline do KNN:
    1. Carrega e prepara os dados
    2. Treina o modelo KNN  
    3. Faz previsões e avalia o desempenho
    4. Exibe resultados detalhados
    """
    
    # ========================================================================
    # ETAPA 1: CARREGAR E PREPARAR OS DADOS
    # ========================================================================
    
    # 1️⃣ Carregar o dataset do arquivo
    print("📂 Carregando dataset...")
    df = pd.read_csv("dataset-iris.txt", header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    print(f"   ✅ {len(df)} instâncias carregadas")

    # 2️⃣ Separar atributos (X) e classes (y)
    print("\n🔢 Separando atributos e classes...")
    X = df.iloc[:, :-1].values  # Primeiras 4 colunas (atributos)
    y = df.iloc[:, -1].values   # Última coluna (classes)
    print(f"   ✅ X: {X.shape} (instâncias x atributos)")
    print(f"   ✅ y: {len(y)} classes")

    # 3️⃣ Dividir em conjuntos de treino (70%) e teste (30%)
    print("\n✂️  Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   ✅ Treino: {len(X_train)} instâncias (70%)")
    print(f"   ✅ Teste: {len(X_test)} instâncias (30%)")

    # 4️⃣ Normalizar os dados
    print("\n📏 Normalizando os dados...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("   ✅ Dados normalizados (média=0, desvio=1)")
    
    # ========================================================================
    # ETAPA 2: TREINAR E AVALIAR O MODELO
    # ========================================================================
    
    # 5️⃣ Configurar e executar o KNN
    print("\n🤖 Executando o algoritmo KNN...")
    k = 5
    print(f"   🔢 Usando k = {k} vizinhos")

    # Fazer previsões para todas as flores do conjunto de teste
    y_pred = knn(X_train, y_train, X_test, k)
    print(f"   ✅ {len(y_pred)} previsões realizadas")

    # 6️⃣ Calcular a acurácia
    print("\n📊 Avaliando o desempenho do modelo...")
    acc = accuracy(y_test, y_pred)
    
    # ========================================================================
    # ETAPA 3: EXIBIR OS RESULTADOS COMPLETOS
    # ========================================================================
    
    # Calcular estatísticas detalhadas
    total_previsoes = len(y_test)
    total_acertos = int(acc * total_previsoes)
    total_erros = total_previsoes - total_acertos

    print("\n" + "="*60)
    print("🎯 RESULTADOS FINAIS DA CLASSIFICAÇÃO KNN")
    print("="*60)

    # Informações do modelo
    print(f"\n🔧 CONFIGURAÇÃO DO MODELO:")
    print(f"   Algoritmo: K-Nearest Neighbors (KNN)")
    print(f"   Número de vizinhos (k): {k}")
    print(f"   Total de instâncias de teste: {total_previsoes}")

    # Métricas de desempenho
    print(f"\n📊 DESEMPENHO DO MODELO:")
    print(f"   ✅ Acertos: {total_acertos}")
    print(f"   ❌ Erros: {total_erros}")
    print(f"   🎯 Acurácia: {acc:.4f} ({acc*100:.2f}%)")

    # Interpretação da acurácia
    if acc >= 0.95:
        interpretacao = "EXCELENTE! 🌟🌟🌟"
    elif acc >= 0.90:
        interpretacao = "MUITO BOM! ✅✅"
    elif acc >= 0.80:
        interpretacao = "BOM! 👍"
    else:
        interpretacao = "PRECISA MELHORAR ⚠️"

    print(f"   📈 Avaliação: {interpretacao}")

    print("-"*60)

    # Exemplos de previsões (melhorados)
    print(f"\n📝 EXEMPLOS DE PREVISÕES (primeiros 15 casos):")
    print(f"{'#':<3} | {'Real':<15} | {'Previsto':<15} | {'Status':<10}")
    print("-"*55)

    for i, (real, pred) in enumerate(zip(y_test[:15], y_pred[:15]), 1):
        status = "✅ Acerto" if real == pred else "❌ Erro" 
        print(f"{i:2d}  | {real:<15} | {pred:<15} | {status}")

    # Resumo por classe (análise detalhada)
    print(f"\n📊 ANÁLISE POR CLASSE:")
    print(f"{'Classe':<15} | {'Total':<6} | {'Acertos':<8} | {'Acurácia':<10}")
    print("-"*50)

    classes_unicas = np.unique(y_test)
    for classe in classes_unicas:
        # Máscara para filtrar apenas instâncias desta classe
        mask = y_test == classe
        y_real_classe = y_test[mask]
        y_pred_classe = np.array(y_pred)[mask]
        
        total_classe = len(y_real_classe)
        acertos_classe = np.sum(y_real_classe == y_pred_classe)
        acuracia_classe = acertos_classe / total_classe
        
        print(f"{classe:<15} | {total_classe:>5} | {acertos_classe:>7} | {acuracia_classe*100:>7.2f}%")

    print("\n" + "="*60)
    print("✅ ANÁLISE COMPLETA! O modelo KNN foi avaliado com sucesso.")
    print("="*60)

# ============================================================================
# EXECUÇÃO DO PROGRAMA
# ============================================================================

if __name__ == "__main__":
    """
    Ponto de entrada do programa.
    
    Executa a função principal quando o script é rodado diretamente.
    Isso permite que o código seja importado como módulo sem executar automaticamente.
    """
    main()

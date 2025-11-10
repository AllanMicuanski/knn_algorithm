# 🌸 Classificação Iris - KNN e SVM com Validação Cruzada

> Projeto acadêmico de Machine Learning implementando algoritmos KNN e SVM para classificação do dataset Iris usando validação cruzada.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Resultados](#-resultados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Usar](#-como-usar)
- [Algoritmos Implementados](#-algoritmos-implementados)
- [Conceitos Importantes](#-conceitos-importantes)
- [Dataset](#-dataset)
- [Autor](#-autor)

## 🎯 Sobre o Projeto

Este projeto implementa e compara dois algoritmos clássicos de Machine Learning para classificação de flores Iris:

- **KNN (K-Nearest Neighbors)** - Implementação customizada
- **SVM (Support Vector Machine)** - Usando scikit-learn

### Características Principais

✅ Implementação modular e profissional  
✅ Validação cruzada estratificada (5-fold)  
✅ Análise completa com matriz de confusão  
✅ Métricas detalhadas: Acurácia, Precisão, Revocação, F1-Score  
✅ Comparação entre modelos  
✅ Código bem documentado e testado

## 🏆 Resultados

### Desempenho dos Modelos

| Modelo  | Acurácia           | Precisão   | Revocação  | F1-Score   |
| ------- | ------------------ | ---------- | ---------- | ---------- |
| **KNN** | 94.00% ± 6.46%     | 94.29%     | 94.00%     | 94.01%     |
| **SVM** | **95.33% ± 4.52%** | **95.49%** | **95.33%** | **95.32%** |

### Matriz de Confusão - SVM (Melhor Modelo)

```
                    Predito
Verdadeiro   Setosa  Versicolor  Virginica
Setosa         50        0          0
Versicolor      0       48          2
Virginica       0        2         48
```

**🥇 Vencedor:** SVM com 95.33% de acurácia

## 📁 Estrutura do Projeto

```
KNN/
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências do projeto
├── main_np2.py                  # Script principal (NP2)
├── knn_iris.py                  # Implementação original simplificada
├── data/
│   └── dataset-iris.txt         # Dataset Iris
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── data_loader.py       # Carregamento e preprocessamento
    ├── models/
    │   ├── __init__.py
    │   ├── knn.py               # Algoritmo KNN customizado
    │   └── svm.py               # Wrapper SVM
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py           # Métricas e validação cruzada
    │   └── confusion_matrix.py  # Análise de matriz de confusão
    └── visualization/
        ├── __init__.py
        └── plots.py             # Gráficos (opcional)
```

## 🚀 Como Usar

### 1. Pré-requisitos

- Python 3.13+ (recomendado)
- pip (gerenciador de pacotes Python)

### 2. Instalação

```bash
# Clone o repositório
git clone https://github.com/AllanMicuanski/knn_algorithm.git
cd knn_algorithm

# Crie um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

### 3. Executar o Projeto

#### Opção 1: Atividade NP2 Completa (Recomendado)

```bash
python main_np2.py
```

**Saída esperada:**

- ✅ Carregamento e preprocessamento dos dados
- 🤖 Configuração dos modelos KNN e SVM
- 🔬 Validação cruzada (5-fold) para ambos
- 📊 Matrizes de confusão detalhadas
- ⚖️ Comparação entre modelos
- � Análise final com insights

#### Opção 2: Teste Rápido

```bash
python main_np2.py --test
```

Executa apenas um teste básico para verificar se tudo está funcionando.

#### Opção 3: Implementação Original Simples

```bash
python knn_iris.py
```

Versão simplificada com apenas KNN.

### 4. Usando o Ambiente Virtual

**Por que usar ambiente virtual?**

O ambiente virtual (`.venv`) é como um "apartamento separado" para as bibliotecas do projeto:

- ✅ Não interfere com outras instalações Python
- ✅ Mantém versões específicas das bibliotecas
- ✅ Facilita compartilhamento do projeto

**Comandos importantes:**

```bash
# Ativar ambiente virtual
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Desativar quando terminar
deactivate

# Usar Python do ambiente virtual
.venv/bin/python main_np2.py     # Linux/Mac
.venv\Scripts\python main_np2.py # Windows
```

## 🤖 Algoritmos Implementados

### KNN (K-Nearest Neighbors)

**Como funciona:**

1. Calcula a distância euclidiana entre a nova amostra e todas as amostras conhecidas
2. Seleciona os K vizinhos mais próximos (K=3 neste projeto)
3. Classifica baseado na votação majoritária dos vizinhos

**Características:**

- ✅ Implementação customizada em Python puro
- ✅ Usa distância euclidiana
- ✅ Interface compatível com scikit-learn
- ✅ K=3 (testado empiricamente)

**Código simplificado:**

```python
def _euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def predict(X):
    distances = [_euclidean_distance(x, x_train) for x_train in X_train]
    k_nearest = np.argsort(distances)[:k]
    k_labels = y_train[k_nearest]
    return most_common_label(k_labels)
```

### SVM (Support Vector Machine)

**Como funciona:**

1. Encontra o hiperplano que melhor separa as classes
2. Maximiza a margem entre as classes
3. Usa kernel RBF para problemas não-lineares

**Características:**

- ✅ Usa scikit-learn (`SVC`)
- ✅ Kernel RBF (Radial Basis Function)
- ✅ C=1.0 (parâmetro de regularização)
- ✅ Excelente para problemas não-lineares

## 💡 Conceitos Importantes

### Validação Cruzada (Cross-Validation)

**O que é?**

Técnica para avaliar modelos de forma mais robusta, dividindo os dados em K partes (folds):

```
Fold 1: [TESTE] [TREINO] [TREINO] [TREINO] [TREINO]
Fold 2: [TREINO] [TESTE] [TREINO] [TREINO] [TREINO]
Fold 3: [TREINO] [TREINO] [TESTE] [TREINO] [TREINO]
Fold 4: [TREINO] [TREINO] [TREINO] [TESTE] [TREINO]
Fold 5: [TREINO] [TREINO] [TREINO] [TREINO] [TESTE]
```

**Por que usar?**

- ✅ Usa todos os dados para treino e teste
- ✅ Reduz viés da divisão aleatória
- ✅ Fornece média e desvio padrão da performance
- ✅ Mais confiável que train/test simples

### Matriz de Confusão

Mostra onde o modelo acerta e erra:

```
                Predito
Verdadeiro   A    B    C
    A       50    0    0  ← Todos os A foram classificados corretamente
    B        0   48    2  ← 2 B foram classificados como C
    C        0    2   48  ← 2 C foram classificados como B
```

### Métricas de Avaliação

| Métrica       | Fórmula                 | O que mede                                                  |
| ------------- | ----------------------- | ----------------------------------------------------------- |
| **Acurácia**  | `Acertos / Total`       | Porcentagem geral de acertos                                |
| **Precisão**  | `VP / (VP + FP)`        | De todos que classifiquei como X, quantos eram realmente X? |
| **Revocação** | `VP / (VP + FN)`        | De todos os X verdadeiros, quantos eu encontrei?            |
| **F1-Score**  | `2 × (P × R) / (P + R)` | Média harmônica de Precisão e Revocação                     |

**Legenda:** VP = Verdadeiros Positivos, FP = Falsos Positivos, FN = Falsos Negativos

### Normalização de Dados

**Por que normalizar?**

As features têm escalas diferentes:

- Comprimento da sépala: 4.3 - 7.9 cm
- Largura da pétala: 0.1 - 2.5 cm

Sem normalização, features com valores maiores "dominam" o cálculo de distância.

**StandardScaler:**

```python
X_normalizado = (X - média) / desvio_padrão
```

Resultado: Todas as features com média=0 e desvio=1

## 📊 Dataset

### Dataset Iris

Criado por Ronald Fisher em 1936, é um dos datasets mais famosos em Machine Learning.

**Características:**

- 📦 **150 instâncias** (50 de cada classe)
- 🏷️ **3 classes:** Iris-setosa, Iris-versicolor, Iris-virginica
- 📏 **4 features numéricas:**
  - Comprimento da sépala (cm)
  - Largura da sépala (cm)
  - Comprimento da pétala (cm)
  - Largura da pétala (cm)

**Distribuição equilibrada:**

- Iris-setosa: 50 amostras (33.33%)
- Iris-versicolor: 50 amostras (33.33%)
- Iris-virginica: 50 amostras (33.33%)

**Dificuldade:** Moderada

- Iris-setosa é linearmente separável
- Iris-versicolor e Iris-virginica têm alguma sobreposição

## � Referências

- [Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [K-Nearest Neighbors Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)

## 👤 Autor

**Allan Micuanski**

- GitHub: [@AllanMicuanski](https://github.com/AllanMicuanski)
- Projeto: Atividade NP2 - Inteligência Artificial

## 📝 Licença

Este projeto é de código aberto e está disponível para fins educacionais.

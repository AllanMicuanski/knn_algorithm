# 🌸 Classificação Iris - KNN vs SVM

Implementação e comparação de algoritmos KNN e SVM para classificação do dataset Iris com validação cruzada.

## 🏆 Resultados

| Modelo  | Acurácia           | Precisão   | Revocação  | F1-Score   |
| ------- | ------------------ | ---------- | ---------- | ---------- |
| KNN     | 94.00% ± 6.46%     | 94.29%     | 94.00%     | 94.01%     |
| **SVM** | **95.33% ± 4.52%** | **95.49%** | **95.33%** | **95.32%** |

## 🚀 Uso Rápido

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar atividade completa
python main_np2.py ||
.venv/bin/python main_np2.py

# Teste rápido
python main_np2.py --test ||
.venv/bin/python main_np2.py --test

```

## 📁 Estrutura

```
├── main_np2.py              # Script principal
├── src/
│   ├── data/
│   │   └── data_loader.py   # Carregamento e preprocessamento
│   ├── models/
│   │   ├── knn.py           # KNN customizado
│   │   └── svm.py           # SVM wrapper
│   └── evaluation/
│       ├── metrics.py       # Métricas e validação cruzada
│       └── confusion_matrix.py
└── data/
    └── dataset-iris.txt     # Dataset
```

## 🤖 Algoritmos

### KNN (K=3)

- Implementação customizada
- Distância euclidiana
- Votação majoritária

### SVM (kernel RBF)

- Scikit-learn
- C=1.0
- Kernel RBF

## 📊 Features do Projeto

✅ Validação cruzada 5-fold  
✅ Matriz de confusão  
✅ Métricas completas (Acurácia, Precisão, Revocação, F1)  
✅ Normalização com StandardScaler  
✅ Código modular

## 📖 Conceitos

**Validação Cruzada:** Divide dados em 5 partes, treina e testa em combinações diferentes para resultados mais confiáveis.

**Métricas:**

- **Acurácia:** % de acertos
- **Precisão:** De todos classificados como X, quantos eram X?
- **Revocação:** De todos os X, quantos foram encontrados?
- **F1-Score:** Média harmônica de Precisão e Revocação

**Normalização:** Coloca todas as features na mesma escala (média=0, desvio=1).

## 🎓 Dataset Iris

- 150 instâncias (50 de cada classe)
- 3 classes: Setosa, Versicolor, Virginica
- 4 features: comprimento/largura da sépala e pétala

---

**Autor:** Allan Micuanski | **Projeto:** NP2 - Inteligência Artificial

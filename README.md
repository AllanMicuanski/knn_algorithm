# KNN - Classificação de Flores Iris

## O que faz

Algoritmo KNN que classifica flores Iris em 3 tipos baseado nas medidas das sépalas e pétalas.

## Como funciona

1. Mede a distância entre uma nova flor e todas as flores conhecidas
2. Encontra os 5 vizinhos mais próximos
3. A nova flor é classificada como o tipo mais comum entre os vizinhos

## Como executar

```bash
# Instalar dependências (se necessário)
pip install pandas numpy scikit-learn

# Executar
python knn_iris.py
```

## Resultado esperado

```
🎯 RESULTADOS FINAIS DA CLASSIFICAÇÃO KNN
============================================================
📊 DESEMPENHO DO MODELO:
   ✅ Acertos: 44
   ❌ Erros: 1
   🎯 Acurácia: 0.9778 (97.78%)
   📈 Avaliação: EXCELENTE! 🌟🌟🌟
```

## 🎯 Métricas de Avaliação

### O que é Acurácia?

A **acurácia** mede quantas previsões o algoritmo acertou:

- **Fórmula:** `Acertos ÷ Total de previsões`
- **Resultado:** Valor entre 0% (péssimo) e 100% (perfeito)

### Como interpretar os resultados?

- 🌟 **90-100%:** Excelente! O modelo está funcionando muito bem
- ✅ **80-90%:** Muito bom! Resultado satisfatório
- 👍 **70-80%:** Bom, mas pode melhorar
- ⚠️ **<70%:** Precisa de ajustes no modelo

## 💡 Conceitos importantes

### Por que dividir em Treino e Teste?

- **Treino (70%):** Dados que o algoritmo usa para "aprender"
- **Teste (30%):** Dados novos para avaliar se realmente aprendeu

É como estudar com um livro (treino) e depois fazer uma prova com questões novas (teste)!

### Por que normalizar os dados?

As medidas têm escalas diferentes:

- Comprimento: pode ser 4.0 a 7.0 cm
- Largura: pode ser 0.1 a 2.5 cm

Sem normalização, medidas maiores "dominam" o cálculo. A normalização coloca tudo na mesma escala (média=0, desvio=1).

### O que é o parâmetro K?

- **K=1:** Muito sensível a ruído (pode errar fácil)
- **K=3 a 7:** Bom equilíbrio (recomendado)
- **K muito alto:** Pode ignorar padrões importantes

## 📄 Dataset

O dataset Iris é um clássico em Machine Learning, criado por Ronald Fisher em 1936. Contém:

- **150 instâncias** de flores
- **3 classes** (50 de cada tipo)
- **4 atributos** por flor
- **Taxa de acurácia típica:** 95-98%

---

**Divirta-se explorando o mundo do Machine Learning!** 🤖

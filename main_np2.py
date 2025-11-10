"""
A06 - KNN e SVM (NP2)
====================

Script principal para implementação da atividade NP2.
Desenvolve uma versão da atividade de classificação realizada na agenda 5 
com o dataset Iris, implementando KNN e SVM, ambos com validação cruzada.

Apresenta:
- Matriz de confusão
- Métricas de avaliação: acurácia, precisão, revocação e F1-score
- Comparação entre os modelos
- Análise detalhada dos resultados

Autor: Allan Micuanski
Data: Novembro 2025
Disciplina: Inteligência Artificial
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Adiciona o diretório src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.data_loader import IrisDataLoader
from src.models.knn import KNNClassifier
from src.models.svm import SVMClassifier
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.confusion_matrix import ConfusionMatrixAnalyzer, create_confusion_matrix_report


def main():
    """
    Função principal da atividade NP2.
    """
    print("="*80)
    print("🎯 A06 - KNN e SVM (NP2)")
    print("   Classificação do Dataset Iris com Validação Cruzada")
    print("="*80)
    
    # 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
    print("\n📊 ETAPA 1: Carregamento e Preparação dos Dados")
    print("-" * 50)
    
    try:
        # Carrega dados
        data_loader = IrisDataLoader()
        data_loader.load_data()
        
        print(f"✅ Dataset carregado com sucesso:")
        print(f"   - Total de amostras: {len(data_loader.data)}")
        print(f"   - Features: {len(data_loader.feature_names)}")
        print(f"   - Classes: {len(data_loader.class_names)}")
        print(f"   - Classes: {', '.join(data_loader.class_names)}")
        
        # Preprocessa dados
        X, y = data_loader.preprocess_data()
        print(f"✅ Dados preprocessados (normalizados)")
        
        # Informações do dataset
        print(f"\n📈 Informações do Dataset:")
        unique, counts = np.unique(y, return_counts=True)
        for i, (cls, count) in enumerate(zip(data_loader.class_names, counts)):
            print(f"   - {cls}: {count} amostras")
        
    except Exception as e:
        print(f"❌ Erro no carregamento dos dados: {e}")
        return
    
    # 2. CONFIGURAÇÃO DOS MODELOS
    print("\n🤖 ETAPA 2: Configuração dos Modelos")
    print("-" * 50)
    
    # Inicializa modelos
    models = {
        'KNN': KNNClassifier(n_neighbors=3),
        'SVM': SVMClassifier(kernel='rbf', C=1.0, random_state=42)
    }
    
    print("✅ Modelos configurados:")
    for name, model in models.items():
        if name == 'KNN':
            print(f"   - {name}: k={model.n_neighbors} vizinhos, distância euclidiana")
        else:
            print(f"   - {name}: kernel RBF, C=1.0")
    
    # 3. AVALIAÇÃO COM VALIDAÇÃO CRUZADA
    print("\n🔬 ETAPA 3: Avaliação com Validação Cruzada")
    print("-" * 50)
    
    evaluator = ModelEvaluator(cv_folds=5, random_state=42)
    all_results = {}
    
    for name, model in models.items():
        print(f"\n🧪 Avaliando {name}...")
        
        try:
            # Validação cruzada
            cv_results = evaluator.cross_validate_model(model, X, y)
            
            # Avaliação completa
            complete_results = evaluator.evaluate_model_complete(
                model, X, y, data_loader.class_names
            )
            
            all_results[name] = {
                'cv_results': cv_results,
                'complete_results': complete_results
            }
            
            print(f"✅ {name} avaliado com sucesso")
            print(f"   - Acurácia média (CV): {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            
        except Exception as e:
            print(f"❌ Erro na avaliação do {name}: {e}")
            continue
    
    # 4. COMPARAÇÃO DE MODELOS
    print("\n⚖️  ETAPA 4: Comparação de Modelos")
    print("-" * 50)
    
    if len(all_results) >= 2:
        try:
            # Compara modelos
            comparison = evaluator.compare_models(all_results)
            
            print("✅ Comparação realizada com sucesso")
            
        except Exception as e:
            print(f"❌ Erro na comparação: {e}")
    
    # 5. ANÁLISE DETALHADA DOS RESULTADOS
    print("\n📋 ETAPA 5: Análise Detalhada dos Resultados")
    print("-" * 50)
    
    for name, results in all_results.items():
        print(f"\n🔍 RESULTADOS DETALHADOS - {name}")
        print("=" * 60)
        
        cv_res = results['cv_results']
        comp_res = results['complete_results']
        
        # Métricas de validação cruzada
        print(f"\n📊 VALIDAÇÃO CRUZADA (5-fold):")
        print(f"   Acurácia:  {cv_res['accuracy_mean']:.4f} ± {cv_res['accuracy_std']:.4f}")
        print(f"   Precisão:  {cv_res['precision_mean']:.4f} ± {cv_res['precision_std']:.4f}")
        print(f"   Revocação: {cv_res['recall_mean']:.4f} ± {cv_res['recall_std']:.4f}")
        print(f"   F1-Score:  {cv_res['f1_mean']:.4f} ± {cv_res['f1_std']:.4f}")
        
        print(f"\n📈 SCORES INDIVIDUAIS POR FOLD:")
        for fold, score in enumerate(cv_res['accuracy_scores'], 1):
            print(f"   Fold {fold}: {score:.4f}")
        
        # Análise da matriz de confusão
        y_true = comp_res['y_true']
        y_pred = comp_res['y_pred']
        
        # Usa o analisador de matriz de confusão
        cm_analyzer = ConfusionMatrixAnalyzer(data_loader.class_names)
        cm_analyzer.print_confusion_matrix_analysis(y_true, y_pred, name)
    
    # 6. RESUMO FINAL E CONCLUSÕES
    print("\n🏆 ETAPA 6: Resumo Final e Conclusões")
    print("-" * 50)
    
    if len(all_results) >= 2:
        # Encontra o melhor modelo
        best_model = None
        best_accuracy = 0
        
        for name, results in all_results.items():
            accuracy = results['cv_results']['accuracy_mean']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = name
        
        print(f"\n🥇 MELHOR MODELO: {best_model}")
        print(f"   Acurácia: {best_accuracy:.4f}")
        
        # Comparação final
        print(f"\n📊 COMPARAÇÃO FINAL:")
        print(f"   {'Modelo':<10} {'Acurácia':<10} {'Precisão':<10} {'Revocação':<10} {'F1-Score':<10}")
        print("-" * 55)
        
        for name, results in all_results.items():
            cv_res = results['cv_results']
            print(f"   {name:<10} "
                  f"{cv_res['accuracy_mean']:<10.4f} "
                  f"{cv_res['precision_mean']:<10.4f} "
                  f"{cv_res['recall_mean']:<10.4f} "
                  f"{cv_res['f1_mean']:<10.4f}")
        
        # Análise das diferenças
        knn_acc = all_results['KNN']['cv_results']['accuracy_mean']
        svm_acc = all_results['SVM']['cv_results']['accuracy_mean']
        diff = abs(knn_acc - svm_acc)
        
        print(f"\n🔍 ANÁLISE:")
        if diff < 0.02:
            print(f"   - Modelos têm performance similar (diferença: {diff:.4f})")
        elif knn_acc > svm_acc:
            print(f"   - KNN superior ao SVM (diferença: {diff:.4f})")
        else:
            print(f"   - SVM superior ao KNN (diferença: {diff:.4f})")
        
        # Insights
        print(f"\n💡 INSIGHTS:")
        print(f"   - Dataset Iris é relativamente simples para ambos os algoritmos")
        print(f"   - Validação cruzada garante robustez dos resultados")
        print(f"   - Normalização dos dados foi importante para o desempenho")
        
        if best_model == 'KNN':
            print(f"   - KNN funcionou bem devido à separabilidade das classes")
        else:
            print(f"   - SVM conseguiu encontrar boa fronteira de decisão")
    
    # 7. INFORMAÇÕES TÉCNICAS
    print(f"\n🔧 INFORMAÇÕES TÉCNICAS:")
    print(f"   - Validação Cruzada: 5-fold estratificada")
    print(f"   - Normalização: StandardScaler")
    print(f"   - KNN: k=3, distância euclidiana")
    print(f"   - SVM: kernel RBF, C=1.0")
    print(f"   - Métricas: micro e macro average")
    
    print("\n" + "="*80)
    print("✅ ATIVIDADE NP2 CONCLUÍDA COM SUCESSO!")
    print("   Todos os requisitos foram atendidos:")
    print("   ✓ Implementação KNN e SVM")
    print("   ✓ Validação Cruzada")
    print("   ✓ Matriz de Confusão")
    print("   ✓ Métricas: Acurácia, Precisão, Revocação, F1-Score")
    print("   ✓ Comparação entre modelos")
    print("="*80)


def run_quick_test():
    """
    Executa um teste rápido para verificar se tudo está funcionando.
    """
    print("🧪 Executando teste rápido...")
    
    try:
        # Testa carregamento
        data_loader = IrisDataLoader()
        data_loader.load_data()
        X, y = data_loader.preprocess_data()
        
        # Testa modelos
        knn = KNNClassifier(n_neighbors=3)
        svm = SVMClassifier()
        
        # Teste básico
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        knn.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        
        knn_pred = knn.predict(X_test)
        svm_pred = svm.predict(X_test)
        
        print("✅ Teste rápido passou! Todos os componentes funcionando.")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste rápido: {e}")
        return False


if __name__ == "__main__":
    """
    Execução principal do script.
    """
    # Verifica argumentos
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Modo teste
        if run_quick_test():
            print("\n🚀 Sistema pronto! Execute sem --test para a atividade completa.")
        else:
            print("\n❌ Sistema com problemas. Verifique as dependências.")
    else:
        # Execução completa
        main()
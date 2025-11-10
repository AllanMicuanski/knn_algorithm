"""
Módulo de Matriz de Confusão
============================

Este módulo contém funções específicas para trabalhar com matrizes de confusão:
- Cálculo de matriz de confusão
- Visualização da matriz
- Interpretação dos resultados

Autor: Allan Micuanski
Data: Novembro 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrixAnalyzer:
    """
    Classe para análise de matriz de confusão.
    
    Fornece métodos para calcular, visualizar e interpretar
    matrizes de confusão de modelos de classificação.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Inicializa o analisador.
        
        Args:
            class_names: Nomes das classes
        """
        self.class_names = class_names
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcula a matriz de confusão.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            
        Returns:
            Matriz de confusão
        """
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Matriz de Confusão", 
                            figsize: tuple = (8, 6), 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota a matriz de confusão.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            title: Título do gráfico
            figsize: Tamanho da figura
            save_path: Caminho para salvar o gráfico
            
        Returns:
            Figura matplotlib
        """
        # Calcula matriz de confusão
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        
        # Cria a figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plota usando seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names or range(cm.shape[1]),
                   yticklabels=self.class_names or range(cm.shape[0]),
                   ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predito', fontsize=12)
        ax.set_ylabel('Verdadeiro', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Analisa a matriz de confusão e extrai insights.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            
        Returns:
            Dicionário com análise detalhada
        """
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        
        analysis = {
            'confusion_matrix': cm,
            'total_samples': cm.sum(),
            'correct_predictions': np.trace(cm),
            'accuracy': np.trace(cm) / cm.sum(),
            'per_class_analysis': {}
        }
        
        # Análise por classe
        for i in range(n_classes):
            class_name = self.class_names[i] if self.class_names else f"Classe_{i}"
            
            # True Positives, False Positives, False Negatives
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp  # Preditos como classe i, mas não são
            fn = cm[i, :].sum() - tp  # São classe i, mas preditos como outras
            tn = cm.sum() - tp - fp - fn  # Verdadeiros negativos
            
            # Métricas por classe
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            analysis['per_class_analysis'][class_name] = {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': cm[i, :].sum()  # Total de amostras da classe
            }
        
        return analysis
    
    def print_confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      model_name: str = "Modelo"):
        """
        Imprime análise detalhada da matriz de confusão.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            model_name: Nome do modelo
        """
        analysis = self.analyze_confusion_matrix(y_true, y_pred)
        cm = analysis['confusion_matrix']
        
        print(f"\n" + "="*70)
        print(f"🎯 ANÁLISE DA MATRIZ DE CONFUSÃO - {model_name}")
        print("="*70)
        
        # Matriz de confusão
        print(f"\n📊 MATRIZ DE CONFUSÃO:")
        if self.class_names:
            # Cabeçalho
            print(f"   {'Verdadeiro \\ Predito':<20}", end="")
            for name in self.class_names:
                print(f"{name:>15}", end="")
            print(f"{'Total':>10}")
            print("-" * (20 + 15 * len(self.class_names) + 10))
            
            # Linhas da matriz
            for i, name in enumerate(self.class_names):
                print(f"   {name:<20}", end="")
                for j in range(len(self.class_names)):
                    print(f"{cm[i,j]:>15}", end="")
                print(f"{cm[i,:].sum():>10}")
            
            # Total por coluna
            print("-" * (20 + 15 * len(self.class_names) + 10))
            print(f"   {'Total':<20}", end="")
            for j in range(len(self.class_names)):
                print(f"{cm[:,j].sum():>15}", end="")
            print(f"{cm.sum():>10}")
        else:
            print(cm)
        
        # Estatísticas gerais
        print(f"\n📈 ESTATÍSTICAS GERAIS:")
        print(f"   Total de amostras: {analysis['total_samples']}")
        print(f"   Predições corretas: {analysis['correct_predictions']}")
        print(f"   Acurácia geral: {analysis['accuracy']:.4f} ({analysis['accuracy']*100:.2f}%)")
        
        # Análise por classe
        print(f"\n🔍 ANÁLISE POR CLASSE:")
        print(f"   {'Classe':<15} {'Precisão':<10} {'Revocação':<10} {'F1-Score':<10} {'Suporte':<10}")
        print("-" * 60)
        
        for class_name, metrics in analysis['per_class_analysis'].items():
            print(f"   {class_name:<15} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} "
                  f"{metrics['support']:<10}")
        
        # Identificar problemas
        print(f"\n⚠️  POSSÍVEIS PROBLEMAS:")
        
        for class_name, metrics in analysis['per_class_analysis'].items():
            issues = []
            
            if metrics['precision'] < 0.8:
                issues.append(f"baixa precisão ({metrics['precision']:.3f})")
            if metrics['recall'] < 0.8:
                issues.append(f"baixa revocação ({metrics['recall']:.3f})")
            if metrics['false_positives'] > metrics['true_positives']:
                issues.append("muitos falsos positivos")
            if metrics['false_negatives'] > metrics['true_positives']:
                issues.append("muitos falsos negativos")
            
            if issues:
                print(f"   - {class_name}: {', '.join(issues)}")
        
        if not any(metrics['precision'] < 0.8 or metrics['recall'] < 0.8 
                  for metrics in analysis['per_class_analysis'].values()):
            print("   ✅ Nenhum problema significativo detectado!")
        
        print("="*70)
    
    def compare_confusion_matrices(self, models_results: dict, 
                                 title: str = "Comparação de Matrizes de Confusão"):
        """
        Compara matrizes de confusão de múltiplos modelos.
        
        Args:
            models_results: Dicionário {nome_modelo: (y_true, y_pred)}
            title: Título da comparação
        """
        n_models = len(models_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, (y_true, y_pred)) in enumerate(models_results.items()):
            cm = self.calculate_confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names or range(cm.shape[1]),
                       yticklabels=self.class_names or range(cm.shape[0]),
                       ax=axes[idx])
            
            axes[idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predito')
            axes[idx].set_ylabel('Verdadeiro')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig


def create_confusion_matrix_report(y_true: np.ndarray, y_pred: np.ndarray,
                                 class_names: List[str], model_name: str = "Modelo") -> dict:
    """
    Função utilitária para criar relatório completo da matriz de confusão.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        class_names: Nomes das classes
        model_name: Nome do modelo
        
    Returns:
        Dicionário com relatório completo
    """
    analyzer = ConfusionMatrixAnalyzer(class_names)
    analysis = analyzer.analyze_confusion_matrix(y_true, y_pred)
    analyzer.print_confusion_matrix_analysis(y_true, y_pred, model_name)
    
    return analysis


if __name__ == "__main__":
    """
    Teste do módulo de matriz de confusão.
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    print("🧪 Testando módulo de matriz de confusão...")
    
    # Cria dados sintéticos
    X, y = make_classification(n_samples=150, n_features=4, n_classes=3, 
                              n_redundant=0, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treina modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Testa analisador
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    analyzer = ConfusionMatrixAnalyzer(class_names)
    
    # Análise completa
    analysis = analyzer.analyze_confusion_matrix(y_test, y_pred)
    analyzer.print_confusion_matrix_analysis(y_test, y_pred, "Random Forest")
    
    # Testa função utilitária
    print("\n🔧 Testando função utilitária...")
    report = create_confusion_matrix_report(y_test, y_pred, class_names, "Teste")
    
    print("\n✅ Todos os testes da matriz de confusão passaram!")
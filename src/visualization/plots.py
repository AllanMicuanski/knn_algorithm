"""
Módulo de Visualização
======================

Este módulo contém funções para visualização de dados e resultados:
- Gráficos de distribuição dos dados
- Matriz de confusão
- Gráficos de comparação de modelos
- Análise de features

Autor: Allan Micuanski
Data: Novembro 2025
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class IrisVisualizer:
    """
    Classe para visualização específica do dataset Iris.
    """
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Inicializa o visualizador.
        
        Args:
            figsize: Tamanho padrão das figuras
        """
        self.figsize = figsize
        self._setup_style()
    
    def _setup_style(self):
        """Configura o estilo dos gráficos."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_data_distribution(self, X: np.ndarray, y: np.ndarray, 
                             feature_names: List[str],
                             class_names: List[str],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota a distribuição dos dados por classe.
        
        Args:
            X: Features
            y: Labels
            feature_names: Nomes das features
            class_names: Nomes das classes
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        n_features = len(feature_names)
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.ravel()
        
        for i in range(n_features):
            for class_idx, class_name in enumerate(class_names):
                mask = y == class_idx
                axes[i].hist(X[mask, i], alpha=0.7, label=class_name, bins=15)
            
            axes[i].set_title(f'Distribuição - {feature_names[i]}', fontweight='bold')
            axes[i].set_xlabel(feature_names[i])
            axes[i].set_ylabel('Frequência')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Distribuição das Features por Classe', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pairwise_features(self, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             class_names: List[str],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota scatter plots de pares de features.
        
        Args:
            X: Features
            y: Labels
            feature_names: Nomes das features
            class_names: Nomes das classes
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Pares de features mais importantes
        feature_pairs = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3)
        ]
        
        colors = ['red', 'green', 'blue']
        
        for idx, (i, j) in enumerate(feature_pairs):
            for class_idx, class_name in enumerate(class_names):
                mask = y == class_idx
                axes[idx].scatter(X[mask, i], X[mask, j], 
                                c=colors[class_idx], alpha=0.7, 
                                label=class_name, s=50)
            
            axes[idx].set_xlabel(feature_names[i])
            axes[idx].set_ylabel(feature_names[j])
            axes[idx].set_title(f'{feature_names[i]} vs {feature_names[j]}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Scatter Plots - Pares de Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str],
                            title: str = "Matriz de Confusão",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota matriz de confusão.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            class_names: Nomes das classes
            title: Título do gráfico
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plota matriz
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Configurações
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='Classe Verdadeira',
               xlabel='Classe Predita')
        
        # Rotaciona labels do eixo x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        
        # Adiciona valores nas células
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict], 
                            metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota comparação entre modelos.
        
        Args:
            results: Resultados dos modelos {nome: {metrica: valor}}
            metrics: Métricas para comparar
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(results.keys())
        
        for idx, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            
            bars = axes[idx].bar(model_names, values, alpha=0.7)
            axes[idx].set_title(f'{metric.capitalize()}', fontweight='bold')
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim(0, 1)
            axes[idx].grid(True, alpha=0.3)
            
            # Adiciona valores nas barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{value:.3f}', ha='center', va='bottom',
                             fontweight='bold')
        
        plt.suptitle('Comparação de Modelos', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cross_validation_scores(self, cv_results: Dict[str, List[float]],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota scores de validação cruzada.
        
        Args:
            cv_results: Resultados CV {modelo: [scores]}
            save_path: Caminho para salvar
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(cv_results.keys())
        scores = list(cv_results.values())
        
        # Box plot
        bp = ax.boxplot(scores, labels=models, patch_artist=True)
        
        # Colorir boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(models)]):
            patch.set_facecolor(color)
        
        ax.set_title('Distribuição dos Scores - Validação Cruzada', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy Score')
        ax.grid(True, alpha=0.3)
        
        # Adiciona médias
        for i, score_list in enumerate(scores):
            mean_score = np.mean(score_list)
            ax.text(i+1, mean_score, f'μ={mean_score:.3f}', 
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class MLVisualizationSuite:
    """
    Suite completa de visualizações para Machine Learning.
    """
    
    def __init__(self):
        """Inicializa a suite de visualização."""
        self.iris_viz = IrisVisualizer()
    
    def create_complete_report(self, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str], class_names: List[str],
                             model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                             cv_results: Optional[Dict[str, List[float]]] = None,
                             output_dir: str = "visualizations") -> Dict[str, plt.Figure]:
        """
        Cria relatório visual completo.
        
        Args:
            X: Features
            y: Labels
            feature_names: Nomes das features
            class_names: Nomes das classes
            model_results: Resultados dos modelos {nome: (y_true, y_pred)}
            cv_results: Resultados de validação cruzada
            output_dir: Diretório de saída
            
        Returns:
            Dicionário com todas as figuras
        """
        figures = {}
        
        # 1. Distribuição dos dados
        print("📊 Criando gráfico de distribuição dos dados...")
        fig_dist = self.iris_viz.plot_data_distribution(
            X, y, feature_names, class_names,
            save_path=f"{output_dir}/data_distribution.png"
        )
        figures['data_distribution'] = fig_dist
        
        # 2. Scatter plots
        print("🔍 Criando scatter plots...")
        fig_scatter = self.iris_viz.plot_pairwise_features(
            X, y, feature_names, class_names,
            save_path=f"{output_dir}/pairwise_features.png"
        )
        figures['pairwise_features'] = fig_scatter
        
        # 3. Matrizes de confusão
        print("🎯 Criando matrizes de confusão...")
        for model_name, (y_true, y_pred) in model_results.items():
            fig_cm = self.iris_viz.plot_confusion_matrix(
                y_true, y_pred, class_names,
                title=f"Matriz de Confusão - {model_name}",
                save_path=f"{output_dir}/confusion_matrix_{model_name.lower()}.png"
            )
            figures[f'confusion_matrix_{model_name}'] = fig_cm
        
        # 4. Comparação de modelos
        if len(model_results) > 1:
            print("⚖️  Criando comparação de modelos...")
            # Calcula métricas para comparação
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            comparison_results = {}
            for model_name, (y_true, y_pred) in model_results.items():
                comparison_results[model_name] = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted'),
                    'recall': recall_score(y_true, y_pred, average='weighted'),
                    'f1_score': f1_score(y_true, y_pred, average='weighted')
                }
            
            fig_comp = self.iris_viz.plot_model_comparison(
                comparison_results,
                save_path=f"{output_dir}/model_comparison.png"
            )
            figures['model_comparison'] = fig_comp
        
        # 5. Validação cruzada
        if cv_results:
            print("📈 Criando gráfico de validação cruzada...")
            fig_cv = self.iris_viz.plot_cross_validation_scores(
                cv_results,
                save_path=f"{output_dir}/cross_validation_scores.png"
            )
            figures['cross_validation'] = fig_cv
        
        print(f"✅ Relatório visual completo criado! Figuras salvas em '{output_dir}/'")
        return figures


def create_visualization_directory(base_path: str = ".") -> str:
    """
    Cria diretório para visualizações.
    
    Args:
        base_path: Caminho base
        
    Returns:
        Caminho do diretório criado
    """
    import os
    
    viz_dir = os.path.join(base_path, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir


if __name__ == "__main__":
    """
    Teste do módulo de visualização.
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    print("🧪 Testando módulo de visualização...")
    
    # Carrega dados Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treina modelos
    rf = RandomForestClassifier(random_state=42)
    svm = SVC(random_state=42)
    
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_pred_svm = svm.predict(X_test)
    
    # Testa visualizações
    viz_suite = MLVisualizationSuite()
    
    model_results = {
        'Random Forest': (y_test, y_pred_rf),
        'SVM': (y_test, y_pred_svm)
    }
    
    cv_results = {
        'Random Forest': [0.95, 0.97, 0.93, 0.96, 0.94],
        'SVM': [0.92, 0.94, 0.91, 0.93, 0.90]
    }
    
    # Cria diretório
    viz_dir = create_visualization_directory()
    
    # Cria relatório
    figures = viz_suite.create_complete_report(
        X, y, feature_names, class_names,
        model_results, cv_results, viz_dir
    )
    
    print(f"\n✅ Teste completo! {len(figures)} figuras criadas em '{viz_dir}/'")
    
    # Mostra figuras (comentado para não bloquear)
    # plt.show()
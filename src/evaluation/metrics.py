"""
Módulo de Avaliação de Modelos
==============================

Este módulo contém funções para avaliar modelos de machine learning:
- Métricas: Acurácia, Precisão, Revocação, F1-Score
- Matriz de Confusão
- Validação Cruzada
- Comparação de modelos

Autor: Allan Micuanski
Data: Novembro 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Classe para avaliar modelos de machine learning.
    
    Fornece métodos para calcular métricas, matriz de confusão,
    validação cruzada e comparação entre modelos.
    """
    
    def __init__(self, class_names: List[str] = None, cv_folds: int = 5, random_state: int = 42):
        """
        Inicializa o avaliador.
        
        Args:
            class_names: Nomes das classes para relatórios
            cv_folds: Número de folds para validação cruzada
            random_state: Seed para reprodutibilidade
        """
        self.class_names = class_names
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula todas as métricas de avaliação.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            
        Returns:
            Dicionário com todas as métricas
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics
    
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
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv: int = None, random_state: int = None) -> Dict[str, Any]:
        """
        Realiza validação cruzada do modelo.
        
        Args:
            model: Modelo a ser avaliado
            X: Features
            y: Labels  
            cv: Número de folds (usa self.cv_folds se None)
            random_state: Seed para reprodutibilidade (usa self.random_state se None)
            
        Returns:
            Dicionário com resultados da validação cruzada
        """
        # Usa parâmetros da classe se não fornecidos
        if cv is None:
            cv = self.cv_folds
        if random_state is None:
            random_state = self.random_state
            
        # Configurar validação cruzada estratificada
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Calcular métricas com validação cruzada
        cv_results = {}
        
        # Acurácia
        accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        cv_results['accuracy_scores'] = accuracy_scores
        cv_results['accuracy_mean'] = accuracy_scores.mean()
        cv_results['accuracy_std'] = accuracy_scores.std()
        
        # Precisão
        precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision_weighted')
        cv_results['precision_scores'] = precision_scores
        cv_results['precision_mean'] = precision_scores.mean()
        cv_results['precision_std'] = precision_scores.std()
        
        # Revocação
        recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall_weighted')
        cv_results['recall_scores'] = recall_scores
        cv_results['recall_mean'] = recall_scores.mean()
        cv_results['recall_std'] = recall_scores.std()
        
        # F1-Score
        f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
        cv_results['f1_scores'] = f1_scores
        cv_results['f1_mean'] = f1_scores.mean()
        cv_results['f1_std'] = f1_scores.std()
        
        return cv_results
    
    def evaluate_model_complete(self, model, X: np.ndarray, y: np.ndarray, 
                              model_name: str = "Model", cv: int = 5) -> Dict[str, Any]:
        """
        Avaliação completa de um modelo usando validação cruzada.
        
        Args:
            model: Modelo a ser avaliado
            X: Features
            y: Labels
            model_name: Nome do modelo para identificação
            cv: Número de folds para validação cruzada
            
        Returns:
            Dicionário com avaliação completa
        """
        # Validação cruzada
        cv_results = self.cross_validate_model(model, X, y, cv=cv)
        
        # Treina o modelo com todos os dados para matriz de confusão
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Matriz de confusão
        conf_matrix = self.calculate_confusion_matrix(y, y_pred)
        
        # Relatório de classificação
        if self.class_names:
            report = classification_report(y, y_pred, target_names=self.class_names, 
                                        output_dict=True)
        else:
            report = classification_report(y, y_pred, output_dict=True)
        
        # Compila resultados
        results = {
            'model_name': model_name,
            'cross_validation': cv_results,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'y_true': y,
            'y_pred': y_pred
        }
        
        # Armazena para comparação posterior
        self.results[model_name] = results
        
        return results
    
    def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> pd.DataFrame:
        """
        Compara múltiplos modelos usando validação cruzada.
        
        Args:
            models: Dicionário com modelos {nome: modelo}
            X: Features
            y: Labels
            cv: Número de folds
            
        Returns:
            DataFrame com comparação dos modelos
        """
        comparison_data = []
        
        for name, model in models.items():
            # Avalia o modelo
            results = self.evaluate_model_complete(model, X, y, name, cv)
            
            # Extrai métricas médias
            cv_results = results['cross_validation']
            row = {
                'Model': name,
                'Accuracy_Mean': cv_results['accuracy_mean'],
                'Accuracy_Std': cv_results['accuracy_std'],
                'Precision_Mean': cv_results['precision_mean'],
                'Precision_Std': cv_results['precision_std'],
                'Recall_Mean': cv_results['recall_mean'],
                'Recall_Std': cv_results['recall_std'],
                'F1_Score_Mean': cv_results['f1_mean'],
                'F1_Score_Std': cv_results['f1_std']
            }
            
            comparison_data.append(row)
        
        # Cria DataFrame para comparação
        comparison_df = pd.DataFrame(comparison_data)
        
        # Ordena por F1-Score (métrica mais balanceada)
        comparison_df = comparison_df.sort_values('F1_Score_Mean', ascending=False)
        
        return comparison_df
    
    def print_model_results(self, model_name: str):
        """
        Imprime resultados detalhados de um modelo.
        
        Args:
            model_name: Nome do modelo
        """
        if model_name not in self.results:
            print(f"❌ Resultados para {model_name} não encontrados!")
            return
        
        results = self.results[model_name]
        cv_results = results['cross_validation']
        
        print(f"\n" + "="*60)
        print(f"📊 RESULTADOS DETALHADOS - {model_name}")
        print("="*60)
        
        # Métricas de validação cruzada
        print(f"\n🔄 VALIDAÇÃO CRUZADA (5-fold):")
        print(f"   Acurácia:  {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        print(f"   Precisão:  {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}")
        print(f"   Revocação: {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}")
        print(f"   F1-Score:  {cv_results['f1_score']['mean']:.4f} ± {cv_results['f1_score']['std']:.4f}")
        
        # Matriz de confusão
        print(f"\n🎯 MATRIZ DE CONFUSÃO:")
        conf_matrix = results['confusion_matrix']
        
        if self.class_names:
            print(f"   {'':>12}", end="")
            for name in self.class_names:
                print(f"{name:>12}", end="")
            print()
            
            for i, name in enumerate(self.class_names):
                print(f"   {name:>12}", end="")
                for j in range(len(self.class_names)):
                    print(f"{conf_matrix[i,j]:>12}", end="")
                print()
        else:
            print(conf_matrix)
        
        print("="*60)
    
    def print_comparison_summary(self, comparison_df: pd.DataFrame):
        """
        Imprime resumo da comparação entre modelos.
        
        Args:
            comparison_df: DataFrame com comparação
        """
        print("\n" + "="*80)
        print("🏆 COMPARAÇÃO DE MODELOS - VALIDAÇÃO CRUZADA")
        print("="*80)
        
        print(f"\n{'Modelo':<15} {'Acurácia':<12} {'Precisão':<12} {'Revocação':<12} {'F1-Score':<12}")
        print("-"*75)
        
        for _, row in comparison_df.iterrows():
            print(f"{row['Model']:<15} "
                  f"{row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.3f}  "
                  f"{row['Precision_Mean']:.4f}±{row['Precision_Std']:.3f}  "
                  f"{row['Recall_Mean']:.4f}±{row['Recall_Std']:.3f}  "
                  f"{row['F1_Score_Mean']:.4f}±{row['F1_Score_Std']:.3f}")
        
        # Destaca o melhor modelo
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 MELHOR MODELO: {best_model['Model']}")
        print(f"   F1-Score: {best_model['F1_Score_Mean']:.4f} ± {best_model['F1_Score_Std']:.4f}")
        
        print("="*80)


class ConfusionMatrixAnalyzer:
    """Análise detalhada de matriz de confusão."""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula matriz de confusão."""
        return confusion_matrix(y_true, y_pred)
    
    def analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Analisa matriz de confusão e retorna insights."""
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        
        analysis = {
            'confusion_matrix': cm,
            'total_samples': cm.sum(),
            'correct_predictions': np.trace(cm),
            'accuracy': np.trace(cm) / cm.sum(),
            'per_class_analysis': {}
        }
        
        for i in range(n_classes):
            class_name = self.class_names[i] if self.class_names else f"Classe_{i}"
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
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
                'support': cm[i, :].sum()
            }
        
        return analysis
    
    def print_confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      model_name: str = "Modelo"):
        """Imprime análise detalhada da matriz de confusão."""
        analysis = self.analyze_confusion_matrix(y_true, y_pred)
        cm = analysis['confusion_matrix']
        
        print(f"\n" + "="*70)
        print(f"🎯 ANÁLISE DA MATRIZ DE CONFUSÃO - {model_name}")
        print("="*70)
        
        print(f"\n📊 MATRIZ DE CONFUSÃO:")
        if self.class_names:
            print(f"   {'Verdadeiro \\ Predito':<20}", end="")
            for name in self.class_names:
                print(f"{name:>15}", end="")
            print(f"{'Total':>10}")
            print("-" * (20 + 15 * len(self.class_names) + 10))
            
            for i, name in enumerate(self.class_names):
                print(f"   {name:<20}", end="")
                for j in range(len(self.class_names)):
                    print(f"{cm[i,j]:>15}", end="")
                print(f"{cm[i,:].sum():>10}")
            
            print("-" * (20 + 15 * len(self.class_names) + 10))
            print(f"   {'Total':<20}", end="")
            for j in range(len(self.class_names)):
                print(f"{cm[:,j].sum():>15}", end="")
            print(f"{cm.sum():>10}")
        else:
            print(cm)
        
        print(f"\n📈 ESTATÍSTICAS GERAIS:")
        print(f"   Total de amostras: {analysis['total_samples']}")
        print(f"   Predições corretas: {analysis['correct_predictions']}")
        print(f"   Acurácia geral: {analysis['accuracy']:.4f} ({analysis['accuracy']*100:.2f}%)")
        
        print(f"\n🔍 ANÁLISE POR CLASSE:")
        print(f"   {'Classe':<15} {'Precisão':<10} {'Revocação':<10} {'F1-Score':<10} {'Suporte':<10}")
        print("-" * 60)
        
        for class_name, metrics in analysis['per_class_analysis'].items():
            print(f"   {class_name:<15} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} "
                  f"{metrics['support']:<10}")
        
        print(f"\n⚠️  POSSÍVEIS PROBLEMAS:")
        has_issues = False
        for class_name, metrics in analysis['per_class_analysis'].items():
            issues = []
            if metrics['precision'] < 0.8:
                issues.append(f"baixa precisão ({metrics['precision']:.3f})")
            if metrics['recall'] < 0.8:
                issues.append(f"baixa revocação ({metrics['recall']:.3f})")
            
            if issues:
                print(f"   - {class_name}: {', '.join(issues)}")
                has_issues = True
        
        if not has_issues:
            print("   ✅ Nenhum problema significativo detectado!")
        
        print("="*70)


def quick_evaluate(model, X: np.ndarray, y: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Função utilitária para avaliação rápida de um modelo.
    
    Args:
        model: Modelo a ser avaliado
        X: Features
        y: Labels
        model_name: Nome do modelo
        
    Returns:
        Dicionário com métricas
    """
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model_complete(model, X, y, model_name)
    return results['cross_validation']
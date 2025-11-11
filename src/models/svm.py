"""
Módulo SVM (Support Vector Machine)
===================================

Este módulo contém a implementação do classificador SVM
usando scikit-learn com interface padronizada.

Autor: Allan Micuanski
Data: Novembro 2025
"""

import numpy as np
from sklearn.svm import SVC
from typing import Dict, Any


class SVMClassifier:
    """
    Wrapper para o SVM do scikit-learn com interface padronizada.
    
    Esta classe encapsula o SVC do scikit-learn para manter
    consistência com nossa implementação de KNN.
    
    Attributes:
        kernel: Tipo de kernel usado
        C: Parâmetro de regularização
        gamma: Coeficiente do kernel
        model: Instância do SVC
        is_fitted: Se o modelo foi treinado
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', 
                 random_state: int = 42):
        """
        Inicializa o classificador SVM.
        
        Args:
            kernel: Tipo de kernel ('linear', 'poly', 'rbf', 'sigmoid')
                   - 'linear': Kernel linear (para dados linearmente separáveis)
                   - 'rbf': Kernel gaussiano (padrão, funciona bem na maioria dos casos)
                   - 'poly': Kernel polinomial
                   - 'sigmoid': Kernel sigmoidal
            C: Parâmetro de regularização
               - Valores baixos: mais regularização (menos overfitting)
               - Valores altos: menos regularização (pode ter overfitting)
            gamma: Coeficiente do kernel RBF
                  - 'scale': 1 / (n_features * X.var()) - padrão
                  - 'auto': 1 / n_features
                  - float: valor específico
            random_state: Seed para reprodutibilidade
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        
        # Cria o modelo SVM do scikit-learn
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True  # Habilita predict_proba
        )
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o modelo SVM.
        
        O SVM encontra o hiperplano que melhor separa as classes,
        maximizando a margem entre elas.
        
        Args:
            X: Features de treinamento (shape: n_samples, n_features)
            y: Labels de treinamento (shape: n_samples,)
            
        Returns:
            self: Instância do classificador para method chaining
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz as classes usando SVM.
        
        Args:
            X: Instâncias a serem classificadas
            
        Returns:
            Array com as classes preditas
            
        Raises:
            ValueError: Se o modelo não foi treinado
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula probabilidades de cada classe.
        
        Note: Para SVM, as probabilidades são estimadas usando
        calibração de Platt, não são probabilidades "verdadeiras".
        
        Args:
            X: Instâncias para calcular probabilidades
            
        Returns:
            Array com probabilidades de cada classe (shape: n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.predict_proba(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula a distância das amostras ao hiperplano de separação.
        
        Args:
            X: Instâncias para calcular distâncias
            
        Returns:
            Array com distâncias ao hiperplano
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.decision_function(X)
    
    def get_support_vectors(self) -> np.ndarray:
        """
        Retorna os support vectors (pontos de apoio) do modelo.
        
        Returns:
            Array com os support vectors
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.support_vectors_
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Retorna parâmetros do modelo.
        
        Args:
            deep: Se deve retornar parâmetros aninhados
            
        Returns:
            Dicionário com parâmetros
        """
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'SVMClassifier':
        """
        Define parâmetros do modelo.
        
        Args:
            **params: Parâmetros a serem definidos
            
        Returns:
            self: Instância do classificador
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Recria o modelo com novos parâmetros
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=self.random_state,
            probability=True
        )
        
        self.is_fitted = False  # Precisa retreinar
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo treinado.
        
        Returns:
            Dicionário com informações do modelo
        """
        if not self.is_fitted:
            return {"status": "Modelo não treinado"}
        
        return {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "n_support_vectors": self.model.n_support_.tolist(),
            "total_support_vectors": sum(self.model.n_support_),
            "classes": self.model.classes_.tolist()
        }


def create_svm_variants() -> Dict[str, SVMClassifier]:
    """
    Cria diferentes variantes do SVM para comparação.
    
    Returns:
        Dicionário com diferentes configurações de SVM
    """
    variants = {
        'SVM_Linear': SVMClassifier(kernel='linear', C=1.0),
        'SVM_RBF': SVMClassifier(kernel='rbf', C=1.0),
        'SVM_Poly': SVMClassifier(kernel='poly', C=1.0, gamma='scale'),
        'SVM_RBF_Soft': SVMClassifier(kernel='rbf', C=0.1),  # Mais regularização
        'SVM_RBF_Hard': SVMClassifier(kernel='rbf', C=10.0),  # Menos regularização
    }
    
    return variants
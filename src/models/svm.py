"""
Módulo SVM (Support Vector Machine)
===================================
Wrapper para SVM do scikit-learn.

Autor: Allan Micuanski
Data: Novembro 2025
"""

import numpy as np
from sklearn.svm import SVC
from typing import Dict, Any


class SVMClassifier:
    """Wrapper para o SVM do scikit-learn com interface padronizada."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', 
                 random_state: int = 42):
        """
        Args:
            kernel: Tipo de kernel ('linear', 'poly', 'rbf', 'sigmoid')
            C: Parâmetro de regularização
            gamma: Coeficiente do kernel
            random_state: Seed para reprodutibilidade
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True
        )
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina o modelo SVM."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz as classes usando SVM."""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calcula probabilidades de cada classe."""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        return self.model.predict_proba(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calcula distância das amostras ao hiperplano."""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        return self.model.decision_function(X)
    
    def get_support_vectors(self) -> np.ndarray:
        """Retorna os support vectors do modelo."""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        return self.model.support_vectors_
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Retorna parâmetros do modelo."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'SVMClassifier':
        """Define parâmetros do modelo."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=self.random_state,
            probability=True
        )
        
        self.is_fitted = False
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo treinado."""
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
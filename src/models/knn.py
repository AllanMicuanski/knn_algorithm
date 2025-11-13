"""
Módulo KNN (K-Nearest Neighbors)
=================================
Implementação própria do algoritmo KNN.

Autor: Allan Micuanski
Data: Novembro 2025
"""

import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """Implementação própria do algoritmo K-Nearest Neighbors."""
    
    def __init__(self, n_neighbors: int = 5, k: int = None):
        """
        Args:
            n_neighbors: Número de vizinhos (padrão: 5)
            k: Alias para n_neighbors
        """
        self.k = k if k is not None else n_neighbors
        self.n_neighbors = self.k
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Treina o modelo (memoriza os dados)."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.is_fitted = True
        return self
    
    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calcula distância euclidiana entre dois pontos."""
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _predict_single(self, x: np.ndarray):
        """Prediz a classe de uma única instância."""
        distances = [(self._euclidean_distance(x, self.X_train[i]), self.y_train[i]) 
                     for i in range(len(self.X_train))]
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        k_labels = [label for _, label in k_nearest]
        return Counter(k_labels).most_common(1)[0][0]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz as classes de múltiplas instâncias."""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        
        X = np.array(X)
        if X.ndim == 1:
            return np.array([self._predict_single(X)])
        
        return np.array([self._predict_single(x) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calcula probabilidades de cada classe."""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        unique_classes = np.unique(self.y_train)
        probabilities = []
        
        for x in X:
            distances = [(self._euclidean_distance(x, self.X_train[i]), self.y_train[i]) 
                         for i in range(len(self.X_train))]
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            k_labels = [label for _, label in k_nearest]
            
            class_probs = [k_labels.count(cls) / self.k for cls in unique_classes]
            probabilities.append(class_probs)
        
        return np.array(probabilities)
    
    def get_params(self, deep=True):
        """Retorna parâmetros do modelo."""
        return {'k': self.k}
    
    def set_params(self, **params):
        """Define parâmetros do modelo."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

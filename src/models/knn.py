"""
Módulo de Modelos de Machine Learning
=====================================

Este módulo contém as implementações dos algoritmos:
- KNN (K-Nearest Neighbors) - implementação própria
- SVM (Support Vector Machine) - usando scikit-learn

Ambos seguem uma interface comum para facilitar comparações.

Autor: Allan Micuanski
Data: Novembro 2025
"""

import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, List
import math


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Implementação própria do algoritmo K-Nearest Neighbors.
    
    O KNN é um algoritmo de aprendizado supervisionado que classifica
    uma nova instância baseado nas classes dos K vizinhos mais próximos.
    
    Attributes:
        k: Número de vizinhos a considerar
        X_train: Dados de treinamento (features)
        y_train: Labels de treinamento
        is_fitted: Se o modelo foi treinado
    """
    
    def __init__(self, n_neighbors: int = 5, k: int = None):
        """
        Inicializa o classificador KNN.
        
        Args:
            n_neighbors: Número de vizinhos a considerar (padrão: 5)
            k: Alias para n_neighbors (mantido para compatibilidade)
        """
        # Compatibilidade com ambos os nomes de parâmetro
        if k is not None:
            self.k = k
        else:
            self.k = n_neighbors
        
        # Propriedade para compatibilidade com scikit-learn
        self.n_neighbors = self.k
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o modelo KNN (apenas memoriza os dados).
        
        KNN é um "lazy learner" - não há treinamento real,
        apenas armazena os dados para uso na predição.
        
        Args:
            X: Features de treinamento
            y: Labels de treinamento
            
        Returns:
            self: Instância do classificador
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.is_fitted = True
        
        return self
    
    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calcula a distância euclidiana entre dois pontos.
        
        Args:
            point1: Primeiro ponto
            point2: Segundo ponto
            
        Returns:
            Distância euclidiana entre os pontos
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _predict_single(self, x: np.ndarray) -> Union[int, float]:
        """
        Prediz a classe de uma única instância.
        
        Args:
            x: Instância a ser classificada
            
        Returns:
            Classe predita
        """
        # Calcula distâncias para todos os pontos de treinamento
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        
        # Ordena por distância e pega os K mais próximos
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # Votação majoritária
        k_labels = [label for _, label in k_nearest]
        prediction = Counter(k_labels).most_common(1)[0][0]
        
        return prediction
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz as classes de múltiplas instâncias.
        
        Args:
            X: Instâncias a serem classificadas
            
        Returns:
            Array com as classes preditas
            
        Raises:
            ValueError: Se o modelo não foi treinado
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        X = np.array(X)
        if X.ndim == 1:
            return np.array([self._predict_single(X)])
        
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula as probabilidades de cada classe (aproximação via votação).
        
        Args:
            X: Instâncias para calcular probabilidades
            
        Returns:
            Array com probabilidades de cada classe
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Identifica classes únicas
        unique_classes = np.unique(self.y_train)
        probabilities = []
        
        for x in X:
            # Calcula distâncias e pega K vizinhos
            distances = []
            for i in range(len(self.X_train)):
                dist = self._euclidean_distance(x, self.X_train[i])
                distances.append((dist, self.y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            k_labels = [label for _, label in k_nearest]
            
            # Calcula probabilidade como proporção de votos
            class_probs = []
            for cls in unique_classes:
                prob = k_labels.count(cls) / self.k
                class_probs.append(prob)
            
            probabilities.append(class_probs)
        
        return np.array(probabilities)
    
    def get_params(self, deep=True):
        """Retorna parâmetros do modelo (compatibilidade scikit-learn)."""
        return {'k': self.k}
    
    def set_params(self, **params):
        """Define parâmetros do modelo (compatibilidade scikit-learn)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SVMClassifier:
    """
    Wrapper para o SVM do scikit-learn com interface padronizada.
    
    Esta classe encapsula o SVC do scikit-learn para manter
    consistência com nossa implementação de KNN.
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', 
                 random_state: int = 42):
        """
        Inicializa o classificador SVM.
        
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
        
        Args:
            X: Features de treinamento
            y: Labels de treinamento
            
        Returns:
            self: Instância do classificador
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
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula probabilidades de cada classe.
        
        Args:
            X: Instâncias para calcular probabilidades
            
        Returns:
            Array com probabilidades de cada classe
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        return self.model.predict_proba(X)
    
    def get_params(self, deep=True):
        """Retorna parâmetros do modelo."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Define parâmetros do modelo."""
        for key, value in params.items():
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


def create_models() -> dict:
    """
    Cria instâncias dos modelos com configurações padrão.
    
    Returns:
        Dicionário com os modelos instanciados
    """
    models = {
        'KNN': KNNClassifier(k=5),
        'SVM': SVMClassifier(kernel='rbf', C=1.0)
    }
    
    return models


if __name__ == "__main__":
    """
    Teste dos modelos de ML.
    """
    from sklearn.datasets import make_classification
    
    print("🧪 Testando módulo de modelos...")
    
    # Cria dados sintéticos para teste
    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, 
                              n_redundant=0, random_state=42)
    
    # Divide em treino e teste
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Testa KNN
    print("\n🔍 Testando KNN...")
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_proba = knn.predict_proba(X_test)
    
    print(f"KNN - Predições: {knn_pred[:5]}")
    print(f"KNN - Probabilidades shape: {knn_proba.shape}")
    
    # Testa SVM
    print("\n🤖 Testando SVM...")
    svm = SVMClassifier()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_proba = svm.predict_proba(X_test)
    
    print(f"SVM - Predições: {svm_pred[:5]}")
    print(f"SVM - Probabilidades shape: {svm_proba.shape}")
    
    # Testa função de criação
    print("\n🏭 Testando criação de modelos...")
    models = create_models()
    print(f"Modelos criados: {list(models.keys())}")
    
    print("\n✅ Todos os testes passaram!")
"""
Módulo de Carregamento de Dados
================================

Carrega e preprocessa o dataset Iris.

Autor: Allan Micuanski
Data: Novembro 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
import os


class IrisDataLoader:
    """Carrega e preprocessa o dataset Iris."""
    
    def __init__(self, data_path: str = "data/dataset-iris.txt"):
        self.data_path = data_path
        self.df = None
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.class_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """Carrega o dataset do arquivo."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.data_path}")
        
        # Carrega CSV
        self.df = pd.read_csv(self.data_path, header=None)
        self.df.columns = self.feature_names + ['class']
        self.df = self.df.dropna()
        self.df['class'] = self.df['class'].str.strip()
        
        # Armazena info
        self.class_names = sorted(self.df['class'].unique())
        self.data = self.df
        
        print(f"✅ Dataset carregado: {len(self.df)} instâncias")
        print(f"✅ Classes encontradas: {self.class_names}")
        
        return self.df
    
    def preprocess_data(self, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocessa os dados para ML."""
        if self.df is None:
            self.load_data()
        
        # Separa features e labels
        X = self.df[self.feature_names].values
        y = self.df['class'].values
        
        # Normaliza
        if normalize:
            X = self.scaler.fit_transform(X)
            print("✅ Features normalizadas (StandardScaler)")
        
        # Codifica classes
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.X = X
        self.y = y_encoded
        
        print(f"✅ Dados preprocessados: X{X.shape}, y{y_encoded.shape}")
        
        return X, y_encoded


def load_iris_data(normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, list]:
    """Função utilitária para carregar rapidamente o dataset Iris."""
    loader = IrisDataLoader()
    X, y = loader.preprocess_data(normalize=normalize)
    class_names = loader.class_names
    
    return X, y, class_names

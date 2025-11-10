"""
Módulo de Carregamento e Preprocessamento de Dados
==================================================

Este módulo é responsável por:
- Carregar o dataset Iris
- Preprocessar os dados (normalização, codificação)
- Dividir dados para validação cruzada
- Preparar dados para os algoritmos de ML

Autor: Allan Micuanski
Data: Novembro 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import os


class IrisDataLoader:
    """
    Classe para carregar e preprocessar o dataset Iris.
    
    Esta classe encapsula todas as operações relacionadas aos dados:
    - Carregamento do arquivo
    - Preprocessamento (normalização, encoding)
    - Statistics do dataset
    - Preparação para ML
    """
    
    def __init__(self, data_path: str = "data/dataset-iris.txt"):
        """
        Inicializa o carregador de dados.
        
        Args:
            data_path: Caminho para o arquivo do dataset
        """
        self.data_path = data_path
        self.df = None
        self.X = None  # Features (atributos)
        self.y = None  # Labels (classes)
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.class_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """
        Carrega o dataset do arquivo.
        
        Returns:
            DataFrame com os dados carregados
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            ValueError: Se o arquivo estiver mal formatado
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.data_path}")
        
        try:
            # Carrega o arquivo CSV
            self.df = pd.read_csv(self.data_path, header=None)
            
            # Define nomes das colunas
            self.df.columns = self.feature_names + ['class']
            
            # Remove linhas vazias ou com valores NaN
            self.df = self.df.dropna()
            
            # Normaliza nomes das classes (remove espaços)
            self.df['class'] = self.df['class'].str.strip()
            
            # Armazena nomes únicos das classes
            self.class_names = sorted(self.df['class'].unique())
            
            print(f"✅ Dataset carregado: {len(self.df)} instâncias")
            print(f"✅ Classes encontradas: {self.class_names}")
            
            return self.df
            
        except Exception as e:
            raise ValueError(f"Erro ao carregar dataset: {e}")
    
    def preprocess_data(self, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocessa os dados para uso em algoritmos de ML.
        
        Args:
            normalize: Se deve normalizar as features
            
        Returns:
            Tuple contendo (X, y) preprocessados
        """
        if self.df is None:
            self.load_data()
        
        # Separa features e labels
        X = self.df[self.feature_names].values
        y = self.df['class'].values
        
        # Normaliza as features se solicitado
        if normalize:
            X = self.scaler.fit_transform(X)
            print("✅ Features normalizadas (StandardScaler)")
        
        # Codifica as classes para números (necessário para alguns algoritmos)
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.X = X
        self.y = y_encoded
        
        print(f"✅ Dados preprocessados: X{X.shape}, y{y_encoded.shape}")
        
        return X, y_encoded
    
    def get_train_test_split(self, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide os dados em treino e teste.
        
        Args:
            test_size: Proporção dos dados para teste
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple contendo (X_train, X_test, y_train, y_test)
        """
        if self.X is None or self.y is None:
            self.preprocess_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y  # Mantém proporção das classes
        )
        
        print(f"✅ Dados divididos - Treino: {len(X_train)}, Teste: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_class_names(self) -> list:
        """
        Retorna os nomes originais das classes.
        
        Returns:
            Lista com nomes das classes
        """
        if self.class_names is None:
            self.load_data()
        return self.class_names
    
    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Decodifica labels numéricos de volta para nomes das classes.
        
        Args:
            y_encoded: Labels codificados numericamente
            
        Returns:
            Labels com nomes originais das classes
        """
        return self.label_encoder.inverse_transform(y_encoded)
    
    def get_dataset_info(self) -> dict:
        """
        Retorna informações estatísticas do dataset.
        
        Returns:
            Dicionário com estatísticas do dataset
        """
        if self.df is None:
            self.load_data()
        
        info = {
            'total_instances': len(self.df),
            'total_features': len(self.feature_names),
            'classes': self.class_names,
            'class_distribution': self.df['class'].value_counts().to_dict(),
            'feature_stats': self.df[self.feature_names].describe().to_dict()
        }
        
        return info
    
    def print_dataset_summary(self):
        """
        Imprime um resumo do dataset.
        """
        info = self.get_dataset_info()
        
        print("\n" + "="*60)
        print("📊 RESUMO DO DATASET IRIS")
        print("="*60)
        print(f"Total de instâncias: {info['total_instances']}")
        print(f"Total de features: {info['total_features']}")
        print(f"Features: {', '.join(self.feature_names)}")
        
        print(f"\n🌸 Distribuição das classes:")
        for class_name, count in info['class_distribution'].items():
            percentage = (count / info['total_instances']) * 100
            print(f"  - {class_name}: {count} instâncias ({percentage:.1f}%)")
        
        print("\n📈 Estatísticas das features:")
        for feature in self.feature_names:
            stats = info['feature_stats'][feature]
            print(f"  - {feature}:")
            print(f"    Média: {stats['mean']:.2f}, Desvio: {stats['std']:.2f}")
            print(f"    Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
        
        print("="*60)


# Função utilitária para uso rápido
def load_iris_data(normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Função utilitária para carregar rapidamente o dataset Iris.
    
    Args:
        normalize: Se deve normalizar as features
        
    Returns:
        Tuple contendo (X, y, class_names)
    """
    loader = IrisDataLoader()
    X, y = loader.preprocess_data(normalize=normalize)
    class_names = loader.get_class_names()
    
    return X, y, class_names


if __name__ == "__main__":
    """
    Teste do módulo de carregamento de dados.
    """
    print("🧪 Testando módulo de carregamento de dados...")
    
    # Cria o carregador
    loader = IrisDataLoader()
    
    # Carrega e mostra informações
    loader.load_data()
    loader.print_dataset_summary()
    
    # Preprocessa os dados
    X, y = loader.preprocess_data()
    
    # Testa divisão treino/teste
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    
    # Testa decodificação
    print(f"\n🔤 Teste de decodificação:")
    print(f"Primeira classe codificada: {y[0]}")
    print(f"Primeira classe decodificada: {loader.decode_labels([y[0]])[0]}")
    
    print("\n✅ Todos os testes passaram!")
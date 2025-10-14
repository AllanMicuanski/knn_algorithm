import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1️⃣ Carregar o dataset
df = pd.read_csv("dataset-iris.txt", header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# 2️⃣ Separar atributos e classes
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 3️⃣ Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4️⃣ Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Funções do KNN
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test_instance, k):
    distances = [euclidean_distance(X_test_instance, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

def knn(X_train, y_train, X_test, k):
    return [knn_predict(X_train, y_train, x_test, k) for x_test in X_test]

# 6️⃣ Treinar e testar
k = 5
y_pred = knn(X_train, y_train, X_test, k)

# 7️⃣ Acurácia
def accuracy(y_true, y_pred):
    return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)

acc = accuracy(y_test, y_pred)

# 8️⃣ Resultados
print("=== Classificação com KNN ===")
print(f"Número de vizinhos (k): {k}")
print(f"Acurácia do modelo: {acc:.2f}")

print("\nExemplos de previsões:")
for real, pred in zip(y_test[:10], y_pred[:10]):
    print(f"Real: {real:15s} | Previsto: {pred}")

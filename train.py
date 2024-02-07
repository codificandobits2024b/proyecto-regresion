import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de entrada
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

y = np.dot(X, np.array([1, 2])) + 3

# Creación y entrenamiento del modelo
reg = LinearRegression().fit(X, y)

# Validación del modelo entrenado
reg.score(X, y)
reg.coef_
reg.intercept_
reg.predict(np.array([[3, 5]]))
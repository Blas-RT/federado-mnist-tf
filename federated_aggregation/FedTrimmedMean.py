
"""
FedTrimmedMean (Recorte de valores extremos)

Este método descarta un porcentaje superior e inferior de los valores de cada peso y después promedia 
los valores restantes. Reduce el efecto de valores extremos y nos da una mejor alternativa frente a datos anómalos.

Para este método es necesario definir un parámetro de recorte (por ejemplo, trim_ratio = 0.2).
"""

import os
import numpy as np
import tensorflow as tf
import sys
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.abspath('../models'))
from TheModel import build

# Cargar modelos locales entrenados
model_paths = [os.path.join("..", "local_training", f"client_model_{i}.keras") for i in range(5)]
local_models = [tf.keras.models.load_model(path) for path in model_paths]
local_weights = [model.get_weights() for model in local_models]

# Función para trimmed mean
def trimmed_mean(weights_list, trim_ratio=0.2):
    trimmed = []
    for i in range(len(weights_list[0])):
        stacked = np.stack([w[i] for w in weights_list], axis=0)
        lower = int(len(stacked) * trim_ratio)
        upper = len(stacked) - lower
        trimmed_weights = np.sort(stacked, axis=0)[lower:upper]
        trimmed_mean = np.mean(trimmed_weights, axis=0)
        trimmed.append(trimmed_mean)
    return trimmed

# Aplicar agregación
aggregated_weights = trimmed_mean(local_weights, trim_ratio=0.2)

# Construir modelo global y asignar pesos
global_model = build.build_it()
global_model.set_weights(aggregated_weights)

# Evaluar en test set
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test / 255.0, -1)

y_pred = global_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("== Resultados para FedTrimmedMean ==")
print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

# Guardar modelo global
global_model.save("global_model_fedtrimmedmean.keras")
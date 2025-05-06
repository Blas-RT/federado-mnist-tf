
"""
FedMedian (Federated Median)

Este método toma la mediana de cada peso en la misma posición entre todos los modelos locales. Es menos sensible 
a valores atípicos (outliers) en los pesos y puede mejorar la estabilidad cuando algunos clientes tienen datos 
ruidosos o distribuciones distintas.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import sys

sys.path.append(os.path.abspath('../models'))
from TheModel import build

# Cargar modelos locales
model_dir = "../local_training"
model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".keras")])
models = [load_model(os.path.join(model_dir, f)) for f in model_files]

# Revisat que sea arquitectura compatible
for i in range(len(models) - 1):
    assert models[i].to_json() == models[i + 1].to_json(), "Modelos con arquitectura diferente"

# Extraer pesos
local_weights = [model.get_weights() for model in models]

# Calcular mediana elemento a elemento
median_weights = [np.median(np.array(w), axis=0) for w in zip(*local_weights)]

# Construir modelo global y asignar pesos
global_model = build.build_it()
global_model.set_weights(median_weights)

# Evaluación
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test / 255.0, -1)
y_pred = global_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("== Classification Report (FedMedian) ==")
print(classification_report(y_test, y_pred_classes))

# Guardar modelo
global_model.save("global_model_fedmedian.keras")

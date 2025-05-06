
"""
FedAvg (Federated Averaging)

Este método realiza un promedio simple de los pesos de los modelos locales entrenados por cada cliente.
Asume que todos los clientes tienen datasets similares y contribuyen equitativamente al modelo global.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import sys

sys.path.append(os.path.abspath('../models'))
from TheModel import build

# Configura la ruta a modelos locales
model_dir = "../local_training"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
model_files.sort()  # Asegura orden consistente

# Carga los modelos
local_models = [load_model(os.path.join(model_dir, f)) for f in model_files]

# Asegúrate de que todos los modelos tengan la misma arquitectura
for i in range(len(local_models) - 1):
    assert local_models[i].to_json() == local_models[i + 1].to_json(), "Modelos no compatibles"

# Promedia los pesos (FedAvg)
local_weights = [model.get_weights() for model in local_models]
averaged_weights = [np.mean(w, axis=0) for w in zip(*local_weights)]

# Construye modelo global y asigna pesos promediados
global_model = build.build_it()
global_model.set_weights(averaged_weights)

# Carga el set de prueba
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test / 255.0, -1)

# Evalúa el modelo global
y_pred = global_model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_labels))

# Guarda el modelo global
global_model.save("global_model_fedavg.keras")
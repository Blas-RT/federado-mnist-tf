import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from models.TheModel import build

# Datos de tamaños por cliente (ajusta si cambias los splits)
num_examples = [12000, 12000, 12000, 12000, 12000]  # Total: 60000
total = sum(num_examples)
weights_ratio = [n / total for n in num_examples]

# Cargar modelos locales
model_dir = "../local_training"
model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".keras")])
models = [load_model(os.path.join(model_dir, f)) for f in model_files]

# Verificar arquitectura compatible
for i in range(len(models) - 1):
    assert models[i].to_json() == models[i + 1].to_json(), "Modelos con arquitectura diferente"

# Extraer pesos
local_weights = [model.get_weights() for model in models]

# Agregación ponderada por tamaño
weighted_weights = []
for weights in zip(*local_weights):
    weighted_layer = np.zeros_like(weights[0])
    for i in range(len(models)):
        weighted_layer += weights_ratio[i] * weights[i]
    weighted_weights.append(weighted_layer)

# Construir y evaluar
global_model = build.build_it()
global_model.set_weights(weighted_weights)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test / 255.0, -1)
y_pred = global_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("== Classification Report (FedWeighted) ==")
print(classification_report(y_test, y_pred_classes))

global_model.save("global_model_fedweighted.keras")
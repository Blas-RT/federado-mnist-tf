# Aprendizaje Federado con MNIST

Este repositorio contiene una implementación completa de un flujo de trabajo de **Aprendizaje Federado** utilizando la base de datos MNIST y TensorFlow. El objetivo es simular un entorno distribuido donde múltiples clientes entrenan localmente sus modelos y luego colaboran para construir un modelo global más robusto, sin compartir directamente sus datos.

---

## Estructura del Proyecto

```bash
FEDERADO-MNIST/
│
├── .venv/                            # Entorno virtual
│
├── datos_confidenciales/            # Subconjuntos locales de datos (fuera del repo público)
│
├── federated_aggregation/           # Estrategias de agregación federada
│   ├── FedAvg.py
│   ├── FedMedian.py
│   └── FedWeighted.py
│
├── local_training/                  # Entrenamiento local por cliente
│   ├── client_model_0.keras
│   └── LocalTrain.ipynb
│
├── models/                          # Definición del modelo base
│   └── TheModel.py
│
├── evaluate_models.ipynb           # Evaluación de modelos globales y locales
├── split_data.ipynb                # División del dataset en subconjuntos
├── README.md
├── .gitignore
├── pyproject.toml
├── uv.lock
└── .python-version
```

## Flujo de Trabajo

### División de datos
En `split_data.ipynb` se divide el dataset **MNIST** en `n` subconjuntos estadísticamente equivalentes (uno por integrante del equipo) y se guardan como archivos `.npz` fuera del repositorio público en la carpeta `datos_confidenciales/`.

### Entrenamiento Local
En `local_training/LocalTrain.ipynb`, cada cliente entrena su propio modelo usando **solo su partición de datos**. El modelo entrenado se guarda como un archivo `.keras` (ej. `client_model_0.keras`).

### Agregación Federada
En `federated_aggregation/` se encuentran tres métodos de agregación para combinar los modelos locales entrenados:

- `FedAvg.py`: Promedio simple de pesos (`FedAvg`).
- `FedMedian.py`: Mediana por cada peso (`FedMedian`).
- `FedWeighted.py`: Promedio ponderado según la cantidad de datos por cliente (`FedWeighted`).

### Evaluación Global y Local
En `evaluate_models.ipynb` se evalúan todos los modelos disponibles:

- Modelos locales entrenados por cada cliente.
- Modelos globales obtenidos mediante cada estrategia de agregación.

---

## Objetivo

Comparar el rendimiento de diferentes estrategias de agregación de modelos en un entorno de aprendizaje federado. Se analizan las métricas de **precisión (accuracy)**, las **curvas de aprendizaje**, y los **classification reports** para determinar qué método ofrece mejores resultados y por qué.

---

## Notas

- Los datos particionados (`.npz`) **no deben subirse al repositorio** por temas de privacidad simulada.

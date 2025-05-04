# Aprendizaje Federado con MNIST
Este repositorio contiene una implementación personalizada de un flujo de trabajo de **Aprendizaje Federado** utilizando la base de datos **MNIST** y **TensorFlow**. El objetivo es simular un entorno distribuido donde múltiples clientes entrenan localmente sus modelos y luego colaboran para construir un modelo global más robusto, sin compartir directamente sus datos.

## Estructura del proyecto

- `/models`: contiene la definición del modelo global utilizado.
- `/local_training`: contiene el código para entrenar localmente cada partición de datos (una por integrante del equipo).
- `/federated_aggregation`: contiene la lógica para agregar los modelos locales usando **FedAvg** y otros dos métodos alternativos.
- `data/`: fuera del repositorio, se almacena la base de datos MNIST dividida en *n* subconjuntos estadísticamente equivalentes (uno por integrante del equipo).

## Objetivo
Comparar el rendimiento de diferentes estrategias de agregación de modelos en un entorno de aprendizaje federado, y analizar su comportamiento frente a métricas de precisión, curvas de aprendizaje y reportes de clasificación.

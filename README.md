# neural-network-from-scratch
neural network repository

## 📌 Descripción del Proyecto

Este repositorio contiene la implementación modular de una **paquete/librería de Redes Neuronales Profundas (Perceptrón Multicapa - MLP)** desarrollada desde cero (*from scratch*) utilizando exclusivamente **NumPy**. 


El objetivo principal es demostrar la comprensión matemática y analítica detrás de los componentes internos del Aprendizaje Profundo: cálculo de pasadas hacia adelante (*forward pass*), derivación analítica mediante la regla de la cadena para la propagación hacia atrás (*backpropagation*), funciones de activación, pérdidas de entropía cruzada y optimizadores estocásticos con decaimiento de tasa de aprendizaje.

Este es el primero de un grupo de librerias con el objetivo final de crear una red neural profunda aplicable a multiples campos desde Large language processing hasta image processing y generative 

---

## 🛠️ Arquitectura e Componentes Implementados

El modelo está modularizado mediante programación orientada a objetos (POO) en las siguientes clases:

* **Capas Densas (`Layer_Dense`):**
  * Inicialización de pesos con distribución Gaussiana y sesgos (*biases*) en cero.
  * *Forward pass* vectorial: $Z = X (dot) W + b$
  * *Backpropagation* analítico para gradientes respecto a entradas, weigths y biases.
* **Funciones de Activación:**
  * **ReLU (`Activation_ReLU`):** Filtrado y derivada para propagación de gradientes.
  * **Softmax (`activation_softmax`):** Transformación a distribución de probabilidad multiclase con estabilización numérica ($x - \max(x)$).
* **Funciones de Pérdida:**
  * **Categorical Cross-Entropy (`cat_cross_entropy`):** Métrica de pérdida para clasificación multiclase con acotamiento numérico (*clipping*) para evitar logaritmos indefinidos ($\log(0)$).
  * **Combinación Integrada (`Activation_softmax_Loss_categoricalcrossentropy`):** Gradiente optimizado y simplificado para el bloque Softmax + Cross-Entropy Loss.
* **Optimizadores:**
  * **Descenso de Gradiente Estocástico (`optimizer_GD`):** Actualización de parámetros con decaimiento dinámico de la tasa de aprendizaje (*Learning Rate Decay*).
 
    ## se integrarán más optimizadores##

---

Markdown
# Deep Learning & Neural Networks From Scratch (NumPy)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Core%20Engine-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)


---

## 📂 Estructura del Repositorio

El proyecto está organizado de forma modular como un paquete de Python importable (`my_N_N_library`):

```text
.
├── my_N_N_library/
│   ├── __init__.py           # Convierte la carpeta en un paquete importable de Python
│   ├── Full_N_N_numpy.py     # Núcleo funcional: capas, activaciones, pérdidas y optimizador
│   ├── N_N_Forward_pass.py   # Script de prueba simplificado para forward pass
│   └── data.py               # Generación y manipulación de datasets de prueba (Spirals, etc.)
│
├── test_GD.py                # Script de prueba integrador (entrenamiento y ajuste a patrones)
└── README.md                 # Documentación del proyecto

---

## 🎯 Visión del Proyecto y Hoja de Ruta (Roadmap)

Este paquete constituye la **fase inicial y el motor de cálculo fundamental** de un ecosistema modular de aprendizaje profundo más amplio, diseñado para extenderse progresivamente desde los fundamentos matriciales en NumPy hacia arquitecturas avanzadas de la industria:

* [x] **Fase 1 (Actual): Core Engine & MLP:** Implementación desde cero de capas densas, backpropagation manual, activaciones, pérdidas y optimización por gradiente.
* [ ] **Fase 2 (Computer Vision):** Capas Convolucionales (`Conv2D`), Pooling (`MaxPool2D`) y aplanamiento (*Flatten*) para procesamiento de imágenes.
* [ ] **Fase 3 (Sequence & NLP):** Redes Recurrentes (`RNN`, `LSTM`) y mecanismos de Atención (*Self-Attention*) para procesamiento de lenguaje natural.
* [ ] **Fase 4 (Generative AI & Transformers):** Ensamblado de bloques Transformer y modelos generativos simplificados.

---

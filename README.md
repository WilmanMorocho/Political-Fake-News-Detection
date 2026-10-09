# Evaluación de Modelos para la Detección de Desinformación Política en Español 🕵️‍♂️📊

[![GitHub last commit](https://img.shields.io/github/last-commit/WilmanMorocho/Political-Fake-News-Detection)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositorio contiene el código fuente, notebooks y artefactos generados para el Trabajo de Integración Curricular (TIC) titulado:

> **"Evaluación de rendimiento de la arquitectura Transformer frente a clasificadores lineales en la detección de desinformación política en español"**

**Autor:** Wilman Alejandro Morocho Carrión  
**Universidad:** Universidad Nacional de Loja - Carrera de Computación  

---

## 📑 Resumen del Proyecto

El objetivo principal de esta investigación es evaluar el compromiso (trade-off) entre el **rendimiento predictivo** (F1-Score) y el **costo computacional** (tiempo de entrenamiento y consumo de memoria) de la arquitectura **BETO** (Spanish BERT) frente a clasificadores tradicionales como **Máquinas de Vectores de Soporte (SVM)** y **Regresión Logística**, en la tarea de clasificación binaria de noticias políticas falsas en español.

El proyecto sigue las primeras fases de la metodología **CRISP-ML(Q)**, abarcando desde la comprensión de los datos hasta la evaluación exhaustiva de los modelos sobre un conjunto de datos ciego (Hold-out).

## 🗂️ Conjunto de Datos

Se utilizó el corpus **Spanish Political Fake News**, que consta de **57,231 noticias**.  
El procesamiento de los datos se realizó mediante un **flujo bifurcado**:
- **Flujo Estricto (Modelos Lineales):** Eliminación de puntuación, paso a minúsculas y remoción de *stopwords*, seguido de vectorización TF-IDF.
- **Flujo Conservador (BETO):** Conservación de mayúsculas, signos de puntuación y estructura sintáctica para aprovechar la atención contextual del modelo Transformer.

## 🚀 Arquitectura y Modelos

1. **Modelos Lineales:**
   - **SVM (Support Vector Machine):** Optimizado mediante GridSearchCV con validación cruzada estratificada (K=5).
   - **Regresión Logística:** Optimizado de manera similar a SVM.
2. **Modelo Transformer:**
   - **BETO (Spanish RoBERTa):** Ajuste fino (*fine-tuning*) realizado en la nube (Google Colab con GPU T4) utilizando un esquema K-Fold estratificado.

## 📊 Resultados Destacados

Los modelos fueron evaluados sobre un conjunto de prueba independiente (20% del corpus original, libre de contaminación):

| Modelo | F1-Score | Tiempo de Inferencia | Consumo de Memoria |
| :--- | :---: | :---: | :---: |
| **SVM** | 0.9067 | 3.59 s | ~63 MB (RAM) |
| **Regresión Logística** | 0.9062 | 3.03 s | ~63 MB (RAM) |
| **BETO (Transformer)** | **0.9684** | 85.00 s | ~561 MB (VRAM) |

*Nota: Aunque BETO supera a los modelos lineales en capacidad predictiva, demanda un costo computacional significativamente mayor, evidenciando el trade-off analizado en la investigación.*

## 📂 Estructura del Repositorio

El repositorio está organizado en base a los cuadernos de Jupyter (*Notebooks*) secuenciales que documentan todo el pipeline:

```text
📁 Political-Fake-News-Detection
├── 01_Actividad_Exploracion_Entendimiento.ipynb     # Análisis Exploratorio de Datos (EDA)
├── 02_Actividad_Limpieza_Normalizacion.ipynb        # Preprocesamiento bifurcado
├── 03_Actividad_Separacion_Pipeline.ipynb           # Hold-out y pipeline base
├── 04_Actividad_Diseno_CV_Entrenamiento_Lineales.ipynb # Diseño de CV para lineales
├── 05_Actividad_Entrenamiento_BETO_Nube.ipynb       # Fine-tuning de BETO
├── 06_Actividad_Evaluacion_Lineales.ipynb           # Evaluación local de SVM y Reg. Log.
├── 07_Actividad_Evaluacion_BETO_Nube.ipynb          # Evaluación de BETO
├── ... (otros cuadernos de GridSearch y pruebas)
├── 11_Evaluacion_HoldOut_Final.py                   # Script de evaluación final combinada
├── best_*.joblib                                    # Pesos de los mejores modelos lineales
├── DocumentoCuartoAvance.pdf                        # Reporte técnico / Documento de avance
├── EvaluacionPoC.pdf                                # Evaluación de Prueba de Concepto
└── data/                                            # Carpeta de datos preprocesados e intermedios
```

## 🛠️ Requisitos e Instalación

Para reproducir los cuadernos localmente, se recomienda crear un entorno virtual con `Python 3.8+` y ejecutar:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install torch transformers
```

*Para el entrenamiento de BETO, se recomienda encarecidamente el uso de un entorno con soporte para aceleración por GPU (ej. Google Colab).*

## 📝 Cita y Uso

Este código fue desarrollado con fines de investigación académica. Si utilizas este repositorio como referencia, por favor cita o enlaza el repositorio original.

---
*Universidad Nacional de Loja - 2026*

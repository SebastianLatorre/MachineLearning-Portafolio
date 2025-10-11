# Portafolio de Proyectos de Machine Learning

Este repositorio reúne notebooks Jupyter usados como portafolio y material didáctico sobre técnicas y proyectos de Machine Learning. Los notebooks están organizados por tema para facilitar la navegación y la revisión de ejemplos prácticos, ejercicios y proyectos personales.

## Estructura y propósito

-   Carpeta principal de notebooks: `Notebooks/`
-   Objetivo: mostrar ejemplos prácticos, implementaciones y ejercicios de cursos y proyectos personales en clasificación, regresión, aprendizaje profundo, procesamiento de imágenes, sistemas de recomendación y análisis de datos.

Si encuentras notebooks con nombres genéricos (por ejemplo, `Untitled*.ipynb`) que deberían documentarse mejor, indica cuáles y los puedo renombrar o agrupar.

## Organización por temas

A continuación se listan los notebooks organizados por tema. Los nombres de los enlaces son representativos y las descripciones son breves (una línea) basadas en el nombre del archivo.

### Clasificación supervisada

-   [Árboles de decisión desde 0](Notebooks/Árboles_de_decisión_desde_0.ipynb) - Implementación y teoría de árboles de decisión desde cero.
-   [Árboles con scikit-learn](Notebooks/Árboles_con_sklearn.ipynb) - Ejemplos de uso de árboles de decisión con scikit-learn.
-   [Support Vector Machines (SVM)](Notebooks/Support_vector_machines.ipynb) - Introducción y ejemplos con SVM.
-   [KNN desde 0](Notebooks/Knn_desde_0.ipynb) - Implementación del algoritmo k-NN paso a paso.
-   [KNN con Bag-of-Words (texto)](Notebooks/kNN_BoW.ipynb) - Uso de KNN sobre representaciones BoW para clasificación de texto.
-   [Hyperbolic Multiclass Classification](Notebooks/Hyperbolic_multiclass_classification.ipynb) - Enfoque avanzado para clasificación multiclase en espacios hiperbólicos.

### Ensambles y Boosting

-   [Random Forests](Notebooks/Random_forests.ipynb) - Implementación y evaluación de bosques aleatorios.
-   [Boosting sobre datos reales](Notebooks/Boosting_datos_reales.ipynb) - Ejemplos prácticos de métodos de boosting.
-   [Práctico Ensembles](Notebooks/5_practico_ensembles_SL.ipynb) - Taller práctico sobre ensamblados de modelos.

### Redes neuronales y Deep Learning

-   [Modelos Keras (genérico)](Notebooks/ModeloKeras.ipynb) - Ejemplos de modelos en Keras para tareas supervisadas.
-   [Keras: Heart model](Notebooks/ModeloKeras%20Heart.ipynb) - Modelo en Keras aplicado a un dataset cardíaco.
-   [MLP (Perceptrón Multicapa)](Notebooks/MLP.ipynb) - Implementación y entrenamiento de una red MLP.
-   [CNN sobre CIFAR](Notebooks/01_CNN_CIFAR.ipynb) - Red convolucional aplicada al dataset CIFAR.
-   [Keras notebooks y experimentos](Notebooks/Keras.ipynb) - Notebooks con ejemplos y experimentos en Keras.

### Procesamiento de imágenes y detección de objetos

-   [YOLOv5 fine-tuning y detección](Notebooks/yolov5customobj.ipynb) - Ajuste fino de modelos YOLOv5 para detección personalizada.
-   [Yolov11 Fine tuning](Notebooks/Yolov11%20Fine%20tuning%20EPP.ipynb) - Experimentos de fine-tuning para detección de objetos.
-   [Práctico de recomendación de imágenes](Notebooks/6_practico_recomendacion_imagenes_SL.ipynb) - Técnicas para recomendación basada en imágenes.

### Procesamiento de texto (NLP)

-   [k-NN con BoW para texto](Notebooks/kNN_BoW.ipynb) - Clasificación de texto usando representaciones Bag-of-Words.
-   [Taller: Folium y Wordcloud](<Notebooks/Taller_2_Folium_y_Wordcloud_[Sebastian_Latorre]%20(1).ipynb>) - Visualización y nubes de palabras (ejercicios con texto).

### Sistemas de recomendación

-   [Feedback implícito (práctico)](Notebooks/3_practico_implicit_feedback_SL.ipynb) - Modelos para señales implícitas de recomendación.
-   [Content-based recommender (práctico)](Notebooks/4_practico_content_based_SL.ipynb) - Recomendador basado en contenido.
-   [Práctico de sistemas (varios)](Notebooks/1_practico_clase1_SLatorre_SistRec.ipynb) - Ejercicios y prácticas de sistemas de recomendación.

### Análisis exploratorio de datos (EDA) y Pandas

-   [Pandas: Análisis de datos (introducción)](Notebooks/2.1%20-%20Pandas%20para%20análisis%20de%20datos.ipynb) - Fundamentos de pandas para análisis de datos.
-   [Análisis de datos con Pandas](Notebooks/1_2_Análisis_de_datos_con_Pandas.ipynb) - Notebook de análisis exploratorio con pandas.
-   [Ejercicio práctico EOD 2012](Notebooks/2.3%20-%20Ejercicio%20práctico%20-%20EOD%202012.ipynb) - Ejercicio práctico de análisis de datos.

### Reducción de dimensionalidad y feature engineering

-   [Taller: Reducción de dimensionalidad](<Notebooks/Taller_3_Reducción_de_dimensionalidad_[_Sebastian_Latorre_]%20(1).ipynb>) - Técnicas de PCA y reducción de dimensiones.

### Series temporales y datos de sensores

-   [Chonos IFOP (serie temporal/procesamiento)](Notebooks/ChonosIFOP.ipynb) - Análisis/visualización de series temporales (nombre sugiere datos de tiempo).

### Proyectos, tareas y evaluaciones (curso)

-   [Tarea Grupal - Fundamentos ML](Notebooks/Tarea_Grupal_Fundamentos_ML.ipynb) - Proyecto grupal de la asignatura Fundamentos de ML.
-   [Evaluaciones prácticas y ejercicios](Notebooks/SebastianLD%20Evaluación%20Práctica%201.ipynb) - Colección de prácticas y evaluaciones.
-   [Tareas y prácticas varias](Notebooks/Tarea1PlataformasML.ipynb) - Notebooks con tareas de curso y plataformas.

### Utilidades, experimentos y notebooks no categorizados

-   [Notebooks sin título / experimentos](Notebooks/Untitled0.ipynb) - Notebook de experimentación (ver contenido para clasificar).
-   [Varios 'Evaluación Práctica' y copias](Notebooks/Evaluación%20Práctica%202%20[%20Sebastian%20Latorre%20].ipynb) - Copias y versiones de ejercicios prácticos.

---

## Cómo usar este repositorio

1. Clona el repositorio.
2. Abre el notebook deseado en JupyterLab / Jupyter Notebook / VS Code.
3. Revisa los requisitos en cada notebook (los kernels o librerías pueden variar según el ejercicio).

Si quieres que organice con más detalle (por ejemplo, separar notebooks por nivel: introductorio / intermedio / avanzado) o que renombre notebooks para mayor claridad, dímelo y lo actualizo.

## Estado y notas

-   Este README fue generado automáticamente con base en los nombres de archivo dentro de `Notebooks/`. Algunas descripciones son inferidas; revisa los notebooks si necesitas descripciones exactas.

---

Archivo generado: versión agrupada por temas (automática). Para ver descripciones completas extraídas de cada notebook, consulta `notebooks_mapping.json`.

Última actualización: 2025-10-11

## Índice agrupado (resumen)

Listado agrupado por temas para facilitar la navegación. Abre cualquier notebook desde `Notebooks/<nombre>.ipynb`.

Revisa también `notebooks_mapping.json` para las descripciones completas extraídas de la primera celda de cada notebook y `notebooks_renamed_log.json` para el registro del renombrado.

## Deep Learning & Keras

-   [Cnn Cifar](Notebooks/cnn-cifar.ipynb) — Import TensorFlow
-   [Copia De Deeplearning Tarea1 Sl](Notebooks/copia-de-deeplearning-tarea1-sl.ipynb) — Tarea 1 Deep Learning
-   [Deeplearning Tarea1 Sl](Notebooks/deeplearning-tarea1-sl.ipynb) — Tarea 1 Deep Learning
-   [Keras](Notebooks/keras.ipynb) — Keras
-   [Keras Sobre Datos Estructurados](Notebooks/1-keras-sobre-datos-estructurados.ipynb) — Clasificador simple sobre datos estructurados
-   [Mlp](Notebooks/mlp.ipynb) — Setup
-   [Modelokeras](Notebooks/modelokeras.ipynb) — Clasificador simple sobre datos estructurados
-   [Modelokeras Heart](Notebooks/modelokeras-heart.ipynb) — Clasificador simple sobre datos estructurados
-   [Practico Recomendacion Imagenes Sl](Notebooks/practico-recomendacion-imagenes-sl.ipynb) — Práctico Deep Learning para Recomendación

## Computer Vision & Object Detection

-   [Yolov11 Fine Tuning Epp](Notebooks/yolov11-fine-tuning-epp.ipynb) — https://www.youtube.com/watch?v=qu2qAXr4Des&t=429s
-   [Yolov5customobj](Notebooks/yolov5customobj.ipynb) — Yolov5customobj

## Natural Language & NLP

-   [Knn Bow](Notebooks/knn-bow.ipynb) — Búsqueda de palabras semanticamente similares y análisis de sentimiento
-   [Knn Bow](Notebooks/knn-bow-v2.ipynb) — Búsqueda de palabras semanticamente similares y análisis de sentimiento
-   [Practico Content Based Sl](Notebooks/practico-content-based-sl.ipynb) — Práctico 4 - Content-based (Texto)
-   [Taller 2 Folium Y Wordcloud](Notebooks/taller-2-folium-y-wordcloud.ipynb) — Folium y Wordcloud - Datos Espaciales y de texto

## Recommender Systems

-   [Practico Clase1 Slatorre Sistrec](Notebooks/practico-clase1-slatorre-sistrec.ipynb) — Práctico Clase 1
-   [Practico Implicit Feedback Sl](Notebooks/practico-implicit-feedback-sl.ipynb) — Práctico 3 - Recomendación basada en feedback implícito.
-   [Practico Mab Sl](Notebooks/practico-mab-sl.ipynb) — Práctico Multi-armed bandits para recomendación

## Exploratory Data Analysis & Pandas

-   [Analisis De Datos](Notebooks/2.2-analisis-de-datos.ipynb) — Análisis de datos en Python
-   [Analisis De Datos Antes De Streamlit](Notebooks/6.1-analisis-de-datos-antes-de-streamlit.ipynb) — Pequeño análisis de datos antes de Streamlit
-   [Analisis De Datos Con Pandas](Notebooks/2-analisis-de-datos-con-pandas.ipynb) — Análisis de datos con Pandas
-   [Copia De 6.1 Analisis De Datos Antes De Streamlit](Notebooks/copia-de-6.1-analisis-de-datos-antes-de-streamlit.ipynb) — Copia de análisis para Streamlit
-   [Ejercicio Practico Eod 2012](Notebooks/2.3-ejercicio-practico-eod-2012.ipynb) — Ejemplo EOD 2012
-   [Pandas Para Analisis De Datos](Notebooks/2.1-pandas-para-analisis-de-datos.ipynb) — Pandas
-   [Pandas Y Altair](Notebooks/pandas-y-altair.ipynb) — Taller evaluado Pandas + Altair

## Ensembles & Trees

-   [Arboles Con Sklearn](Notebooks/arboles-con-sklearn.ipynb) — Clasificación con árboles de decisión en scikit-learn
-   [Arboles Con Sklearn (v2)](Notebooks/arboles-con-sklearn-v2.ipynb) — Variante
-   [Arboles De Decision Desde 0](Notebooks/arboles-de-decision-desde-0.ipynb) — Arboles de decision desde 0
-   [Arboles De Decision Desde 0 (v2)](Notebooks/arboles-de-decision-desde-0-v2.ipynb) — Variante
-   [Boosting Datos Reales](Notebooks/boosting-datos-reales.ipynb) — Boosting aplicado a indicadores financieros
-   [Practico Ensembles Sl](Notebooks/practico-ensembles-sl.ipynb) — Práctico Ensembles
-   [Random Forests](Notebooks/random-forests.ipynb) — Random forests
-   [Random Forests (v2)](Notebooks/random-forests-v2.ipynb) — Variante
-   [Sobreajuste De Arboles](Notebooks/sobreajuste-de-arboles.ipynb) — Sobreajuste de arboles
-   [Sobreajuste De Arboles (v2)](Notebooks/sobreajuste-de-arboles-v2.ipynb) — Variante
-   [Tarea Copia Boosting](Notebooks/tarea-copia-boosting.ipynb) — Código de boosting (copia)

## Classical ML (SVM / KNN / Support)

-   [Extension De Svm](Notebooks/extension-de-svm.ipynb) — Ejercicio SVM
-   [Extension De Svm (v2)](Notebooks/extension-de-svm-v2.ipynb) — Variante
-   [Knn Desde 0](Notebooks/knn-desde-0.ipynb) — Knn desde 0
-   [Knn Desde 0 (v2)](Notebooks/knn-desde-0-v2.ipynb) — Variante
-   [Ktp](Notebooks/ktp.ipynb) — Comparación / tests
-   [Support Vector Machines](Notebooks/support-vector-machines.ipynb) — SVM básico

## Graph / GNN / Manifolds

-   [Hyperbolic Multiclass Classification](Notebooks/hyperbolic-multiclass-classification.ipynb) — Notas sobre manifolds / geoopt
-   [Untitled3](Notebooks/untitled3.ipynb) — Referencia GCN

## Workshops & Tutorials

-   [Copia De 2 Practico Clase2](Notebooks/copia-de-2-practico-clase2.ipynb) — Práctico Clase 2
-   [Copia De Apollo 11](Notebooks/copia-de-apollo-11.ipynb) — Contenido de ejemplo
-   [Copia De Evaluacion Practica 33](Notebooks/copia-de-evaluacion-practica-33.ipynb) — Entregas / plantilla
-   [Copia De Tarea Grupal Fundamentos Ml](Notebooks/copia-de-tarea-grupal-fundamentos-ml.ipynb) — Copia tarea grupal
-   [Nvidialab1](Notebooks/nvidialab1.ipynb) — Enlace a recurso
-   [Parte1 Ep Sebastianl](Notebooks/parte1-ep-sebastianl.ipynb) — Evaluación práctica I - NetworkX
-   [Taller 3 Reduccion De Dimensionalidad](Notebooks/taller-3-reduccion-de-dimensionalidad.ipynb) — Reducción de dimensionalidad

## Course exercises & Deliverables

-   [Base Evaluacion Practica 1](Notebooks/base-evaluacion-practica-1.ipynb) — Plantilla / entrega práctica
-   [Chonosifop](Notebooks/chonosifop.ipynb) — Entrega / ejemplo
-   [Evaluacion Practica 1](Notebooks/evaluacion-practica-1.ipynb) — Entrega práctica 1
-   [Evaluacion Practica 2](Notebooks/evaluacion-practica-2.ipynb) — Entrega práctica 2
-   [Evaluacion Practica 2 (v2)](Notebooks/evaluacion-practica-2-v2.ipynb) — Variante
-   [Evaluacion Practica 22](Notebooks/evaluacion-practica-22.ipynb) — Entrega práctica 22
-   [Evaluacion Practica 33](Notebooks/evaluacion-practica-33.ipynb) — Entrega práctica 33
-   [Evaluacion Practica 44](Notebooks/evaluacion-practica-44.ipynb) — Entrega práctica 44
-   [Evaluacion Practica 55](Notebooks/evaluacion-practica-55.ipynb) — Entrega práctica 55
-   [Evaluacion Practica 66](Notebooks/evaluacion-practica-66.ipynb) — Entrega práctica 66
-   [Evaluacion Practica 77](Notebooks/evaluacion-practica-77.ipynb) — Entrega práctica 77
-   [Evaluacion Practica 88](Notebooks/evaluacion-practica-88.ipynb) — Entrega práctica 88
-   [Evaluacion Practica 99](Notebooks/evaluacion-practica-99.ipynb) — Entrega práctica 99
-   [Tarea Grupal Fundamentos Ml](Notebooks/tarea-grupal-fundamentos-ml.ipynb) — Proyecto grupal
-   [Tarea Grupal Fundamentos Ml (v2)](Notebooks/tarea-grupal-fundamentos-ml-v2.ipynb) — Variante
-   [Tarea1plataformasml](Notebooks/tarea1plataformasml.ipynb) — Tarea 1 plataformas ML
-   [Tarea2fundamentos Ml](Notebooks/tarea2fundamentos-ml.ipynb) — Tarea 2 Fundamentos ML

## Others

-   [.b](Notebooks/2.b.ipynb) — 2B
-   [Crossvalidation Desicion Tree](Notebooks/crossvalidation-desicion-tree.ipynb) — Ejemplo Decision Tree
-   [Implementacion De Modelos Con Scikit Learn](Notebooks/3.1-implementacion-de-modelos-con-scikit-learn.ipynb) — Implementación con scikit-learn
-   [Untitled0](Notebooks/untitled0.ipynb) — Instalación / Kaggle setup
-   [Untitled1](Notebooks/untitled1.ipynb) — untitled
-   [Untitled2](Notebooks/untitled2.ipynb) — untitled
-   [Untitled4](Notebooks/untitled4.ipynb) — untitled

---

Si quieres que agrupe diferente (por ejemplo: nivel — introductorio/intermedio/avanzado—, o por semestre/proyecto), indícamelo y lo ajusto.

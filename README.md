# Clasificador de vacas en imágenes de cámaras trampa

Este repositorio contiene el código y la metodología para entrenar y evaluar un modelo de visión computacional basado en aprendizaje profundo, diseñado para detectar y aislar la presencia de ganado bovino en imágenes de cámaras trampa. 

El objetivo principal de este proyecto es actuar como un filtro automatizado para las bases de datos de monitoreo de biodiversidad de la CONABIO. Al identificar y separar las imágenes de vacas invadiendo ecosistemas naturales, se optimiza el tiempo de análisis de los investigadores, permitiéndoles enfocarse exclusivamente en la fauna nativa. Además, este modelo busca superar el sesgo geográfico que presentan los modelos globales cuando se aplican a los fenotipos locales y la densa vegetación de los bosques y selvas de México.

## Organización

El repositorio está estructurado de manera modular para separar la configuración, el procesamiento de datos, la arquitectura del modelo y la inferencia:

* `configs/`: Directorio que contiene los archivos `.yaml` con los hiperparámetros y configuraciones de los experimentos.
* `src/`: Directorio principal del código fuente, dividido en submódulos:
    * `algorithms/` : Define la lógica del ciclo del entrenamiento usando PyTorch Lightning (`LightningModule`). Configura los optimizadores (SGD), *schedulers* (StepLR) y los pasos de entrenamiento, validación y evaluación.
    * `datasets/`: Implementa las clases de conjuntos de datos (`Dataset`) y los `DataLoaders`. Aquí se define el pipeline de *data augmentation* (recortes aleatorios, *flips* horizontales/verticales, *ColorJitter* y normalización)  para mejorar la robustez del modelo frente a variaciones de iluminación en los ecosistemas.
    * `models/`: Se define la arquitectura computacional. Contiene la clase `PlainResNetClassifier`, encargada de instanciar el *backbone* (ResNet18 o ResNet50), descargar los pesos pre-entrenados de ImageNet y adaptar la capa clasificadora final (`nn.Linear`) para el problema binario.
    * `utils/`: Funciones auxiliares. Incluye el cálculo riguroso de métricas de desempeño (Macro Accuracy, Micro Accuracy y precisión por clase) mediante matrices de confusión.
* `main.py`: Script principal, lleva acabo la organización del entrenamiento y evaluación. Maneja la inicialización de los submódulos de `src/`, configuración de *loggers* (CSV, TensorBoard, Wandb, Comet) y ejecución del `Trainer`.

* `detection_only.py` / `main_detector_classifier.py`: Herramientas de inferencia que integran los modelos base de PyTorch Wildlife (como YOLOv9 o RtDetr) para generar las detecciones (*bounding boxes*) previas a la clasificación.

## Metodología

El flujo de trabajo aborda el problema en dos etapas principales, optimizando el uso de recursos mediante *transfer learning*:

1. **Detección y extracción:** Se utilizan detectores pre-entrenados (integrados a través de  [PyTorch Wildlife](https://microsoft.github.io/CameraTraps/model_zoo/megadetector/) como MegaDetector/YOLO), para identificar cualquier animal en la imagen original y generar recortes (*bounding boxes*). 
2. **Clasificación:** Dado que los modelos suelen confundir al ganado local con fauna nativa debido a las condiciones de la vegetación, los recortes obtenidos se procesan con nuestro clasificador. Para garantizar la robustez del modelo frente a las condiciones variables de las cámaras trampa, durante el entrenamiento se aplica un riguroso pipeline de aumento de datos (*data augmentation*). Este proceso incluye recortes y escalados aleatorios (*Random Resized Crop*), inversiones espaciales (*Horizontal/Vertical Flips*) y variaciones controladas de iluminación y color (*Color Jitter* en brillo, contraste, saturación y matiz). Finalmente, mediante el *fine-tuning* de la arquitectura **ResNet50**, la red utiliza estas imágenes para adaptar sus pesos finales y discernir las características específicas del ganado frente a los fondos complejos del país.

## Resultados

Los siguientes resultados y métricas de rendimiento fueron obtenidos al evaluar la clasificación sobre los siguientes conjuntos de datos:

* **Propio (CONABIO):** Conjunto de imágenes representativas de ecosistemas mexicanos (bosques y selvas). Este conjunto fue fundamental para adaptar el modelo a las condiciones locales de iluminación y vegetación. Se integraron directorios y recortes provenientes de:
    * **SIPECAM:** Sistema Integrado de Monitoreo Fotográfico y Acústico para México.
    * **IBUNAM / Instituto de Biología:** Colecciones y fototrampeo de la Universidad Nacional Autónoma de México.
    * **Pronatura (Veracruz):** Registros de conservación de fauna en la región.

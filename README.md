# Reconocimiento automático del habla en español con Common Voice

> **Proyecto final de Deep Learning** **Fuente principal:** Mozilla Common Voice Scripted Speech 27.0 - Spanish (`es`).

## Resumen del proyecto

Este proyecto busca entrenar y evaluar un sistema de reconocimiento automático del habla (Automatic Speech Recognition, ASR) que convierta grabaciones reales de voz en español a texto. Se utilizará Common Voice, un corpus de grabaciones aportadas por hablantes voluntarios junto con sus transcripciones. El estudio se centrará en tres preguntas:

- Si el ajuste fino de un encoder Wav2Vec 2.0 mejora el reconocimiento frente al mismo modelo entrenado desde cero

- Si el error cambia según el acento declarado

- Cuánto empeora la transcripción cuando el audio contiene ruido.

La métrica principal será **Word Error Rate (WER)**. Como análisis complementario, se reportará **Character Error Rate (CER)**.

## Planteamiento del problema y justificación

Los sistemas ASR pueden cometer errores cuando cambia la pronunciación, el acento, la velocidad del habla o la calidad de la grabación. Un modelo con buen desempeño promedio también podría presentar errores mayores en algunos grupos de hablantes o cuando el audio está degradado. Por eso, además de medir el rendimiento global, es importante analizar la generalización y las condiciones en las que falla.

**Problema de investigación:** ¿En qué medida el ajuste fino de un modelo Wav2Vec 2.0 con grabaciones de Common Voice mejora la transcripción del español frente al entrenamiento desde cero, y cómo varía su error entre grupos de acento y bajo condiciones controladas de ruido?

El proyecto es relevante para aplicaciones de dictado, accesibilidad, interfaces de voz y transcripción. El objetivo académico no es construir un producto listo para producción, sino comparar estrategias de aprendizaje y explicar sus resultados, errores y limitaciones.

### Título

**Evaluación de Wav2Vec 2.0 para el reconocimiento del habla en español: Ajuste fino, variación por acento y robustez al ruido**

### Objetivo general

Evaluar el desempeño de un sistema de reconocimiento automático del habla en español entrenado con Common Voice mediante la comparación entre un encoder Wav2Vec 2.0 preentrenado y una arquitectura equivalente inicializada desde cero, utilizando WER como métrica principal y analizando las diferencias de error por acento declarado y por nivel de ruido.

### Objetivos específicos

1. Preparar un subconjunto reproducible de Common Voice y comparar el WER de Wav2Vec 2.0 con encoder preentrenado ajustado para transcribir español frente al mismo diseño de modelo entrenado desde una inicialización aleatoria.

2. Medir y comparar el WER en grupos de acento de España y América Latina cuando los metadatos disponibles y el número de hablantes por grupo permitan una comparación razonable.

3. Evaluar la robustez del modelo con versiones del conjunto de prueba a las que se añade ruido en niveles controlados, cuantificando el cambio de WER respecto al audio limpio.

## Dataset y procedencia

### Fuente principal: Common Voice Spanish

- **Repositorio/ficha oficial:** [Common Voice Scripted Speech 27.0 - Spanish en Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmu5kitx700pemi073feoiqqu).

- **Proyecto y descripción general:** [Mozilla Common Voice](https://commonvoice.mozilla.org/es/about).

- **Versión elegida:** `cv-corpus-27.0-2026-09-11`, idioma español (`es`). Se fija esta versión para que los resultados sean reproducibles.

- **Licencia declarada:** CC0-1.0. La ficha oficial también indica que está prohibido intentar identificar a los hablantes y volver a alojar o redistribuir el dataset. No se subirán los audios ni sus metadatos al repositorio del proyecto.

- **Tamaño de referencia de la ficha oficial:** 1.683.737 clips, unas 2.282,72 horas de audio grabado y 597,6 horas validadas, provenientes de 26.960 hablantes. El archivo comprimido completo se anuncia en aproximadamente **48,38 GB**. La ficha registra 360.691 clips de entrenamiento, 15.907 de desarrollo y 15.907 de prueba en los splits de modelado. Estas cifras corresponden a la ficha consultada y deben confirmarse en el manifiesto descargado antes del análisis.

- **Tipo de datos:** Clips de voz en MP3, transcripciones y metadatos. Entre los campos descritos por Common Voice se encuentran `client_id`, `path`, `sentence`, `sentence_id`, votos de validación y, cuando el hablante lo declara, `accents`, `age` y `gender`.

- **Importante:** No es necesario ni recomendable entrenar con todo el corpus para este proyecto. Se usará primero un subconjunto controlado de audio validado y se registrarán su duración, número de clips, hablantes y distribución por acento.

La versión 27.0 es grande para descargar y procesar localmente. Se recomienda ejecutar el entrenamiento en un entorno con GPU en la nube (por ejemplo, Kaggle Notebooks o Google Colab) y mantener en el equipo local el código, las configuraciones, las métricas y las gráficas. El dataset se descarga desde la ficha oficial y se acepta allí la licencia/condiciones correspondientes.

### Conjunto externo para evaluar generalización

Se propone utilizar **FLEURS, configuración `es_419`**, solo para la evaluación externa y sin usar su partición `test` durante entrenamiento o selección de hiperparámetros:

- [FLEURS en Hugging Face](https://huggingface.co/datasets/google/fleurs/tree/main/es_419)

- Idioma/configuración: Español de América Latina (`es_419`).

- Licencia declarada: CC-BY-4.0; debe respetarse la atribución indicada en la ficha.

FLEURS es una fuente diferente a Common Voice y, por ello, ayuda a comprobar si el sistema se comporta de manera similar ante grabaciones que no provienen del corpus de entrenamiento. Antes de evaluarlo, se revisarán posibles coincidencias de transcripciones normalizadas y se documentará que estas muestras no participaron en el ajuste del modelo. Su estilo también es principalmente de lectura, así que este experimento no demuestra por sí solo rendimiento en conversaciones espontáneas.

## **Preguntas experimentales y problemas a resolver**

### Problema 1. ¿El ajuste fino mejora el reconocimiento frente al entrenamiento desde cero?

**Pregunta:** con el mismo subconjunto de entrenamiento y la misma arquitectura, ¿Usar pesos acústicos preentrenados reduce el WER frente a inicializar aleatoriamente todos los pesos?

**Experimentos:**

- **Modelo A - ajuste fino:** Encoder `facebook/wav2vec2-base` preentrenado con audio sin transcripciones, una cabeza CTC y vocabulario construido para las transcripciones en español. La ficha del modelo aclara que el checkpoint base no incluye un tokenizer de ASR, por lo que deben configurarse el vocabulario español y la cabeza de salida.

- **Modelo B - desde cero:** Misma configuración de Wav2Vec 2.0, misma cabeza CTC, vocabulario, datos, particiones y protocolo de entrenamiento, pero con pesos inicializados aleatoriamente.

- **Baseline clásico opcional:** Un sistema GMM-HMM. Se implementará solo si el tiempo disponible permite preparar el pipeline adicional de características acústicas, pronunciación y alineamiento. No es requisito para completar la comparación principal si resulta inviable.

**Métrica principal:** WER global sobre el mismo conjunto de prueba. Se reportará también CER y la diferencia absoluta de WER entre modelos. La comparación no debe presentar el checkpoint base de Wav2Vec 2.0 como un reconocedor español listo para usar: Es un encoder preentrenado que necesita una cabeza de salida y ajuste fino para esta tarea.

### Problema 2. ¿El error cambia según el acento?

**Pregunta:** ¿Hay diferencias observables de WER entre grabaciones clasificadas en grupos de acento de España y América Latina?

**Método:** Utilizar el campo de acento declarado por participantes, verificar las categorías reales en la versión descargada y definir explícitamente el mapeo a grupos antes de calcular métricas. Por ejemplo, las etiquetas peninsulares y canaria pueden agruparse como España; las etiquetas mexicana, andina, caribeña, centroamericana, chilena y rioplatense pueden agruparse como América Latina. La asignación exacta dependerá de las categorías disponibles y quedará documentada.

**Métricas:** WER y CER por grupo, número de clips y número de hablantes por grupo, además de la diferencia de WER entre los grupos. Se informará cuántas muestras carecen de metadatos de acento.

**Precauciones:** los datos de acento son autodeclarados, no cubren a todos los participantes y pueden estar muy desbalanceados. No se afirmará que un grupo es peor si su tamaño es insuficiente. Siempre que sea posible se compararán cantidades semejantes de hablantes y se dará prioridad a particiones por hablante para evitar que clips del mismo hablante aparezcan en entrenamiento y prueba.

### Problema 3. ¿Cuánto empeora el sistema cuando el audio tiene ruido?

**Pregunta:** ¿Cuánto aumenta el WER cuando se evalúa una misma muestra en condiciones limpias y bajo ruido añadido controladamente?

**Método:** Mantener intacto el conjunto de prueba limpio y crear copias de evaluación con ruido ambiental a relaciones señal-ruido (SNR) predefinidas, por ejemplo 20 dB, 10 dB y 5 dB. Las versiones degradadas serán únicamente condiciones experimentales derivadas del audio real; no se presentarán como grabaciones reales adicionales. El tipo de ruido, semilla aleatoria y SNR se registrarán para poder repetir el experimento.

**Métricas:** WER limpio, WER por nivel de SNR y cambio absoluto `ΔWER = WER_ruidoso - WER_limpio`. También se puede reportar el cambio relativo en porcentaje, siempre indicando claramente la fórmula.

## Métricas

La métrica principal será **Word Error Rate**:

\[ WER = \\frac\{S + D + I\}\{N\} \]

Donde `S` representa sustituciones, `D` eliminaciones, `I` inserciones y `N` el número de palabras de referencia. Un WER menor indica menos errores de transcripción. El WER puede superar 1 (o 100 %) si hay muchos errores; por eso no debe tratarse como una exactitud convencional.

Como métrica secundaria se usará **Character Error Rate (CER)**, que calcula una proporción análoga a nivel de caracteres. Se definirá una única rutina de normalización para referencias e hipótesis (por ejemplo, pasar a minúsculas, normalizar espacios y aplicar un tratamiento consistente de puntuación). La política sobre tildes y `ñ` se fijará antes de comparar modelos; no se cambiará entre experimentos.

## Preprocesamiento y prevención de fuga de datos

1. Utilizar solo clips validados para la comparación principal y documentar las reglas para descartar audios vacíos, corruptos o con transcripciones faltantes.

2. Convertir/resamplear la señal a **16 kHz y mono** para el modelo elegido; comprobar duración y valores no finitos antes del entrenamiento.

3. Normalizar de manera consistente las transcripciones y construir el vocabulario CTC con los caracteres que realmente aparezcan en el español del subconjunto, incluidos los caracteres acentuados necesarios.

4. Fijar el subconjunto de entrenamiento antes de ejecutar ambos experimentos; registrar la semilla, los identificadores de clips y los tiempos totales seleccionados.

5. Usar `train` para ajustar pesos, `dev`/validación para seleccionar hiperparámetros y `test` una sola vez para el resultado final. Comprobar duplicados de audio/texto y posibles solapamientos de hablantes (`client_id`) entre particiones. Si hay solapamientos, preparar una partición agrupada por hablante para la evaluación principal.

6. No seleccionar hiperparámetros con FLEURS `test` ni utilizar las transcripciones de prueba para mejorar el vocabulario o el modelo.

7. Generar versiones ruidosas únicamente después de fijar el conjunto de prueba limpio. La evaluación limpia y las condiciones ruidosas deben usar exactamente las mismas frases.

## Arquitectura y entrenamiento

**Arquitectura propuesta:** Wav2Vec 2.0 encoder + proyección de salida CTC (Connectionist Temporal Classification). Wav2Vec 2.0 aprende representaciones acústicas a partir de audio; la cabeza CTC transforma las representaciones en secuencias de caracteres sin exigir alinear manualmente cada carácter con un instante del audio.

**Referencia del modelo:** [Wav2Vec 2.0, Baevski et al. (2020)](https://arxiv.org/abs/2006.11477).  
**Checkpoint candidato:** [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base). Se cambiará/configurará la salida CTC para el alfabeto español y se explicará esta modificación en la presentación.

Los hiperparámetros se elegirán mediante un experimento piloto y se registrarán en un archivo de configuración. Como mínimo, reportar:

- Tasa de aprendizaje, tamaño de batch efectivo y acumulación de gradientes;

- Número de épocas, máximo de duración por clip y estrategia de congelación/descongelación del encoder;

- Optimizador, scheduler, weight decay, precisión mixta si se utiliza y semilla;

- Función de pérdida CTC y tratamiento de padding;

- Mejor época de validación, criterio de parada y tiempo de entrenamiento.

Para mantener la comparación justa, los dos modelos deben usar las mismas particiones, preprocesamiento, vocabulario y presupuesto de datos. Se debe reconocer que el entrenamiento desde cero puede necesitar más datos y cómputo que el ajuste fino. Por eso, ambos modelos comenzarán con un subconjunto limitado; solo se ampliará si el hardware y el calendario lo permiten.

## Recursos computacionales y reproducibilidad

El proyecto no requiere incluir datos ni rutas específicas de un computador en el repositorio. Se recomienda esta estructura:

```
.
├── README.md
├── requirements.txt
├── configs/
│   └── experiment.yaml
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_finetune_wav2vec2.ipynb
│   ├── 03_train_from_scratch.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data.py
│   ├── model.py
│   ├── metrics.py
│   └── noise.py
├── outputs/
│   ├── figures/
│   └── results/
└── data/
    └── external/   # no subir los datos al control de versiones
```

**Entorno recomendado:** Python, PyTorch, Transformers, Datasets (si corresponde al formato cargado), Accelerate, `jiwer`, `pandas`, `soundfile` y `matplotlib`. Se debe registrar la versión exacta de librerías y del checkpoint. El dataset de Common Voice debe descargarse por separado desde su fuente oficial.

**Hardware que se documentará tras ejecutar los experimentos:** CPU, modelo y memoria de GPU, RAM, entorno (local/Kaggle/Colab), duración de entrenamiento de cada modelo, uso aproximado de memoria y tamaño del subconjunto usado. No se deben inventar estos valores de antemano.

## Resultados que se deben presentar

Los resultados aún no se han medido. Al terminar los experimentos, completar al menos esta tabla:

| Experimento | Datos de entrenamiento | WER en test limpio | CER en test limpio | Tiempo de entrenamiento |
| - | -: | -: | -: | -: |
| Wav2Vec 2.0 ajustado | Por registrar | Por medir | Por medir | Por medir |
| Wav2Vec 2.0 desde cero | Por registrar | Por medir | Por medir | Por medir |
| GMM-HMM (opcional) | Por registrar | Por medir | Por medir | Por medir |


Añadir gráficas de loss de entrenamiento y validación, WER por acento, WER según SNR y ejemplos de predicciones correctas e incorrectas. Analizar qué errores aparecen (sustituciones, inserciones o eliminaciones), si hay sobreajuste y si las conclusiones se mantienen en FLEURS `es_419`. No se debe concluir que un modelo es mejor basándose solo en la gráfica de loss.

### Criterios de interpretación

- Un menor WER favorece la transcripción, siempre que los conjuntos y reglas de normalización sean comparables.

- Una diferencia por acento no demuestra por sí misma una causa lingüística: Puede estar relacionada con el tamaño de muestra, el contenido, la edad, la calidad de grabación u otras variables.

- Un aumento de WER al bajar el SNR cuantifica sensibilidad al ruido, pero solo para los ruidos y niveles evaluados.

- Si el resultado externo es peor que el de Common Voice, describirlo como una diferencia de dominio/corpus que requiere análisis, no como prueba automática de una única causa.

## Limitaciones y uso responsable

- Common Voice contiene habla leída y no representa toda la variación del habla espontánea.

- Los metadatos demográficos/acento pueden faltar y son autodeclarados; los grupos pueden ser desbalanceados.

- Los resultados de un subconjunto y una GPU limitada no se generalizan automáticamente al corpus completo.

- El checkpoint preentrenado incorpora conocimiento acústico previo; el experimento mide la utilidad de la inicialización y no solo la arquitectura.

- La prueba con ruido añadido es controlada y no equivale a cubrir todos los ruidos reales.

- No intentar identificar a los hablantes, inferir identidades ni publicar los clips o metadatos de Common Voice. No cargar los archivos del dataset a GitHub.

## Entregables alineados con los lineamientos del curso

Los documentos del curso requieren que la sustentación comience con problema, justificación y objetivos; explique la arquitectura y sus referencias, la base de datos y el preprocesamiento, las curvas de aprendizaje, los hiperparámetros, los recursos de cómputo, los resultados sobre datos no vistos y las conclusiones. Todos los integrantes deben poder responder preguntas sobre el proyecto.

- **Entrega preliminar:** Título, objetivo general y entre dos y tres objetivos específicos.

- **Presentación:** Máximo 15 minutos, más 5 minutos para preguntas. Debe explicar y analizar resultados, no limitarse a mostrar gráficas.

- **Informe final:** PDF de máximo 3 páginas, con objetivos, dataset y preprocesamiento, arquitectura y metodología, resultados con métricas/visuales y conclusiones basadas en evidencia.

- **Fechas indicadas en `README(2).md`:** Preliminar, jueves 29 de octubre de 2026; presentación e informe final, miércoles 11 de noviembre de 2026. Verificar la programación de sustentaciones con el curso.

## Referencias y enlaces

1. Mozilla Common Voice, versión seleccionada: [Common Voice Scripted Speech 27.0 - Spanish](https://mozilladatacollective.com/datasets/cmu5kitx700pemi073feoiqqu).

2. Ardila, R. et al. (2020). *Common Voice: A Massively-Multilingual Speech Corpus*. [LREC 2020](https://aclanthology.org/2020.lrec-1.520/).

3. Baevski, A. et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477).

4. Meta/Hugging Face. [facebook/wav2vec2-base - model card](https://huggingface.co/facebook/wav2vec2-base).

5. Google. [FLEURS - dataset card en Hugging Face](https://huggingface.co/datasets/google/fleurs/tree/main/es_419).


**Nota de alcance:** la línea principal del proyecto es la comparación de ajuste fino frente a entrenamiento desde cero. El análisis de acento y de ruido se realizará en el alcance disponible; si el tiempo o el cómputo son insuficientes, el baseline GMM-HMM se presentará como trabajo opcional y se justificarán las decisiones de alcance.


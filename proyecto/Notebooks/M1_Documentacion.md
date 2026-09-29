# Módulo M1: Fine-tuning de Modelo Base para Tutoría de Matemáticas

**Universidad EAFIT | SI4006 — Tópicos Especiales y Aplicaciones en IA**
**Fecha de entrega:** Sesión 4 | Agosto 2026

---

## RESUMEN EJECUTIVO

Se construyó un tutor de matemáticas en español capaz de resolver problemas
enunciados en lenguaje natural y operaciones directas, generando el
procedimiento paso a paso y la respuesta final.

Se compararon **cuatro arquitecturas** mediante fine-tuning sobre un corpus
propio de 165 ejemplos, con particiones y métricas idénticas para las cuatro.

**Modelo base elegido:** `Qwen/Qwen2.5-1.5B-Instruct` con LoRA
**Arquitectura de despliegue:** el modelo único, **sin** pipeline de clasificación

**El hallazgo principal no es la exactitud, sino qué enseñó el fine-tuning.**
Qwen ya resolvía el 90.91% de los problemas *antes* de entrenarlo; después
resolvía el 93.94%, una diferencia de un solo ejemplo que no es
estadísticamente significativa (McNemar exacto, p = 1.000). Lo que sí cambió de
forma radical fue la **forma** de la respuesta: el cumplimiento del formato
pedagógico pasó del **12.12% al 100%** y la verbosidad se redujo de 82.5 a 23.5
palabras por respuesta.

Conclusión honesta: **con 132 ejemplos, el fine-tuning enseñó a explicar, no a
calcular.** Los cuatro experimentos sostienen esa afirmación y el documento la
argumenta en detalle.

---

## ESTRUCTURA DE ENTREGA

```
ZIP 1 — Notebooks de entrenamiento (código)
ZIP 2 — Artefactos, datos y resultados (salidas)
```

Cada sección indica de qué ZIP proviene la información.

---

# 1. EL PROBLEMA Y EL DISEÑO EXPERIMENTAL

## 1.1 Tarea

Un estudiante escribe una pregunta en lenguaje natural ("María tiene 48
caramelos y quiere repartirlos entre 6 amigos") o una operación directa
("(45 - 9) ÷ 6 + 11"). El sistema debe devolver el **procedimiento y la
respuesta**, no solo el número: un tutor que solo da resultados no enseña.

## 1.2 Qué hace válida la comparación

Las cuatro arquitecturas comparten, por construcción:

- **El mismo corpus**, verificado con hash SHA-256 en cada notebook.
- **Las mismas particiones** (132 entrenamiento / 33 validación), estratificadas
  y **fijadas en el archivo de datos**, no recalculadas por cada notebook.
- **Las mismas métricas**, definidas en una única celda compartida.
- **La misma semilla** (42) y decodificación *greedy* en toda evaluación.

Si cada notebook hubiera hecho su propio `train_test_split`, las diferencias
entre modelos podrían venir de los datos y no de la arquitectura.

## 1.3 Métricas y por qué esas

**Generación** (Qwen, pipeline, FLAN-T5)

| Métrica | Qué mide | Por qué está |
|---|---|---|
| **Exactitud** | ¿El número final es correcto? | La única que responde "¿sirve como tutor?". Un procedimiento bonito con resultado equivocado es un fracaso. |
| **Formato válido** | ¿Usó `Paso N:` y `Respuesta final:`? | Separa *aprender a resolver* de *aprender a formatear*. **Sin esta métrica habríamos concluido que el fine-tuning no sirvió para nada**, cuando en realidad hizo el trabajo más visible del experimento. |
| **ROUGE-L** | Solapamiento de la explicación con la referencia | Aproxima si el procedimiento se parece al esperado. Métrica débil: premia coincidencia léxica, no razonamiento. |
| **Longitud media** | Palabras generadas | Detecta divagación. Resultó ser el indicador más limpio del cambio de estilo. |

**Clasificación** (BERT, FLAN-T5)

Accuracy, precision/recall/F1 macro y ponderado, más matriz de confusión. La
métrica principal es **F1-macro**: con 11 clases, si el modelo ignora una clase
entera el F1-macro se hunde aunque la accuracy apenas se mueva.

## 1.4 Extracción justa de la respuesta

Para medir la exactitud se busca el número que sigue al marcador
`Respuesta final:` y, **si no aparece, el último número del texto**. Ese
respaldo es decisivo para la honestidad del experimento: sin él estaríamos
castigando al modelo sin entrenar por desconocer un formato que todavía no le
hemos enseñado, y la mejora aparente del fine-tuning habría sido enorme y falsa.

Es exactamente lo que ocurrió: el baseline de Qwen cumplió el formato solo el
12.12% de las veces, pero resolvió correctamente el 90.91% de los problemas.

---

# 2. EL CORPUS

**Fuente:** ZIP 2 → `data/math_tutor_dataset.jsonl`

165 ejemplos en español, **11 categorías × 15 ejemplos**, particiones
estratificadas de 132 entrenamiento / 33 validación (semilla 42).

## 2.1 Distribución real

| Categoría | Total | Train | Validación |
|---|---|---|---|
| suma | 15 | 12 | 3 |
| resta | 15 | 12 | 3 |
| multiplicacion | 15 | 12 | 3 |
| division | 15 | 12 | 3 |
| operaciones_combinadas | 15 | 12 | 3 |
| potencias_raices | 15 | 12 | 3 |
| fracciones | 15 | 12 | 3 |
| porcentajes | 15 | 12 | 3 |
| ecuaciones | 15 | 12 | 3 |
| geometria | 15 | 12 | 3 |
| estadistica_probabilidad | 15 | 12 | 3 |
| **TOTAL** | **165** | **132** | **33** |

Otras distribuciones:

- **Tipo:** `operacion` 90 · `problema` (lenguaje natural) 75
- **Nivel:** `basico` 93 · `intermedio` 72
- **Origen:** `base` 52 (de los 50 ejemplos originales, algunos divididos) · `nuevo` 113

El **balance perfecto entre clases es deliberado**: el corpus original estaba
muy sesgado (19 ejemplos de operaciones combinadas frente a 2 de estadística),
lo que hacía imposible entrenar un clasificador de 11 clases.

## 2.2 Estructura de un ejemplo

```json
{
  "id": "div-01",
  "split": "train",
  "categoria": "division",
  "categoria_id": 3,
  "tipo": "problema",
  "nivel": "basico",
  "origen": "base",
  "entrada": "María tiene 48 caramelos y quiere repartirlos por igual entre 6 amigos. ¿Cuántos caramelos recibirá cada amigo?",
  "salida": "Paso 1: Repartir en partes iguales es dividir.\nPaso 2: 48 ÷ 6 = 8.\nRespuesta final: 8 caramelos",
  "respuesta_final": "8 caramelos",
  "valor": "8",
  "es_demo": false
}
```

El campo `valor` es la clave de la evaluación automática: la respuesta en forma
canónica, sin unidades ni texto.

## 2.3 Curación aplicada

| Principio | Qué se hizo |
|---|---|
| **Formato uniforme** | La `salida` se genera mecánicamente a partir de una lista de pasos. Ningún ejemplo puede desviarse del formato porque nadie lo escribe a mano. |
| **Consistencia** | Un esquema de campos único, una plantilla de prompt única y un marcador `Respuesta final:` que hace posible la evaluación automática. |
| **Representatividad** | 11 categorías balanceadas a 15 ejemplos, corrigiendo el sesgo del corpus original. |
| **Diversidad** | Mezcla de problemas en lenguaje natural y operaciones directas, dos niveles de dificultad, contextos variados. |
| **Limpieza** | **Verificación aritmética automática de los 165 ejemplos**, deduplicación exacta y normalizada, y separación de los ítems que hacían dos preguntas a la vez. |
| **Trazabilidad** | Cada ejemplo declara su `origen`, distinguiendo lo heredado de lo añadido. |

## 2.4 Esquema de etiquetado

Las categorías se solapan por naturaleza ("el 25% de 420" es un porcentaje *y*
una multiplicación). Se aplicó una **cascada de precedencia** determinista:

```
1. ¿Área, perímetro, ángulos, volumen?             -> geometria
2. ¿Hay que despejar una incógnita?                -> ecuaciones
3. ¿Promedio, mediana, moda, rango, probabilidad?  -> estadistica_probabilidad
4. ¿Interviene un porcentaje?                      -> porcentajes
5. ¿Los operandos son fracciones?                  -> fracciones
6. ¿Hay potencias o raíces?                        -> potencias_raices
7. ¿Dos o más operaciones distintas / paréntesis?  -> operaciones_combinadas
8. En otro caso, la única operación presente       -> suma|resta|multiplicacion|division
```

## 2.5 Una limitación del corpus que los resultados revelaron

El baseline de Qwen resolvió **30 de 33** problemas sin haber sido entrenado.
Eso significa que **el corpus es demasiado fácil para discriminar entre
modelos generativos competentes**: no queda margen de mejora que medir. Un
corpus con problemas de varios pasos, cifras mayores o enunciados ambiguos
habría dado una comparación mucho más informativa. Es la conclusión más
accionable del experimento de cara a M2.

---

# 3. LAS CUATRO ARQUITECTURAS

**Fuente:** ZIP 1 → `ProyectoIA/`

## 3.1 Qwen2.5 — decoder-only (generación)

`S04_Lab_Fine_tuning_Qwen.ipynb`

- **Modelo base:** `Qwen/Qwen2.5-1.5B-Instruct` — 1 543 714 304 parámetros
- **Método:** LoRA r=16, alpha=32 sobre `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Entrenables:** 4 358 144 (**0.28%** del total)
- **Hiperparámetros:** lr 2e-4, 8 épocas, mejor época **4**
- **Tiempo:** 129.4 s en GPU T4

Decisión técnica relevante: la pérdida se calcula **solo sobre la respuesta**
(los tokens del enunciado se enmascaran con `-100`). Sin eso, el modelo gastaría
capacidad aprendiendo a predecir el enunciado, que en producción escribe el
usuario.

## 3.2 BETO — encoder-only (clasificación)

`S04_Lab_Fine_tuning_BERT.ipynb`

- **Modelo base:** `dccuchile/bert-base-spanish-wwm-cased` — 109 859 339 parámetros
- **Método:** fine-tuning completo (**100%** de los parámetros)
- **Hiperparámetros:** lr 3e-5, 15 épocas, mejor época **8**
- **Tiempo:** 299.5 s en GPU T4

Se usó fine-tuning completo y no LoRA de forma deliberada: con 110 M de
parámetros el modelo cabe entero en una T4, su cabeza de clasificación es nueva
y hay que entrenarla igualmente, y con datasets pequeños permitir que se ajusten
todas las capas suele dar mejor resultado en clasificación.

## 3.3 Pipeline BETO → Qwen — cascada modular

`S04_Lab_Fine_tuning_Qwen_BERT.ipynb`

```
Usuario
   ↓
[BETO]   Etapa 1 · clasifica el tipo de problema (11 categorías) + confianza
   ↓
[Selección de estrategia]   categoría → instrucciones didácticas especializadas
   ↓
[Qwen + LoRA]   Etapa 2 · genera la solución condicionada por esa estrategia
   ↓
Paso 1: ... / Paso 2: ... / Respuesta final: ...
```

No entrena nada nuevo: compone los modelos de 3.1 y 3.2. Se evaluaron **tres
configuraciones** para poder atribuir el resultado:

| | Qué es |
|---|---|
| **A** | Qwen fine-tuned sin enrutar — el baseline del pipeline |
| **B** | Pipeline real, con la categoría que predice BETO |
| **C** | Oráculo: la categoría verdadera del corpus. El techo teórico del pipeline |

> **Nota de arquitectura.** Esto es una **cascada** de un encoder-only y un
> decoder-only: dos redes independientes, con dos tokenizadores distintos,
> entrenadas por separado. **No es un encoder-decoder.** El encoder-decoder de
> este proyecto es FLAN-T5 (3.4), donde ambas mitades están unidas por
> *cross-attention* y se entrenan con una única pérdida.

## 3.4 FLAN-T5 — encoder-decoder (integrado)

`S04_Lab_Fine_tuning_FLAN_T5.ipynb`

- **Modelo base:** `google/flan-t5-base` — 247 577 856 parámetros
- **Método:** LoRA r=16, alpha=32 sobre `q` y `v`
- **Entrenables:** 1 769 472 (**0.71%** del total)
- **Hiperparámetros:** lr 3e-4, 15 épocas, mejor época **15**
- **Tiempo:** 105.7 s en GPU T4, en fp32 (fp16 desborda en T5)

Se entrenó para escribir la categoría antes del procedimiento, resolviendo
clasificación y generación en una sola pasada:

```
Categoria: porcentajes
Paso 1: ...
Respuesta final: 105 estudiantes
```

## 3.5 Comparación de arquitecturas

`S04_Comparacion_Arquitecturas.ipynb` — agrega los cuatro `resultados/*.json`.

---

# 4. RESULTADOS

**Fuente:** ZIP 2 → `resultados/`

## 4.1 El resultado central: baseline vs fine-tuned

### Qwen2.5-1.5B + LoRA

| Métrica | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| **Exactitud** | 90.91% (30/33) | 93.94% (31/33) | +3.03% |
| **Formato válido** | **12.12% (4/33)** | **100% (33/33)** | **+87.88%** |
| **ROUGE-L** | 0.270 | 0.824 | +0.554 |
| **Longitud media** | 82.5 palabras | 23.5 palabras | −71% |

**Cómo hay que leer esta tabla.** La primera fila parece el titular y no lo es.
La diferencia de exactitud es de **un solo ejemplo**: tres problemas pasaron de
mal a bien (`mul-08`, `geo-06`, `est-07`) y dos pasaron de bien a mal
(`fra-12`, `est-01`). Aplicando la prueba de McNemar exacta a ese 3 contra 2,
**p = 1.000**: no hay ninguna evidencia de mejora en la capacidad de cálculo.

Las otras tres filas sí son concluyentes y describen lo que el fine-tuning hizo
de verdad:

- **El formato pasó de 4 de 33 a 33 de 33.** El modelo aprendió la estructura
  pedagógica de forma completa y sin excepciones.
- **La respuesta se acortó un 71%.** El baseline divagaba —82 palabras de
  media, a menudo inventando preguntas adicionales o repitiéndose hasta agotar
  el presupuesto de tokens—. El modelo entrenado responde en 23 palabras.
- **ROUGE-L se triplicó**, coherente con lo anterior: el procedimiento ahora se
  parece al que le enseñamos.

**Conclusión: el fine-tuning enseñó a explicar, no a calcular.** Es un resultado
legítimo y, con 132 ejemplos, el esperable. Presentarlo como "el modelo aprendió
matemáticas" sería falso.

### FLAN-T5-base + LoRA

| Métrica | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Exactitud | 0.00% (0/33) | 3.03% (1/33) | +3.03% |
| **Formato válido** | 0.00% | **100%** | **+100%** |
| ROUGE-L | 0.156 | 0.504 | +0.348 |
| Clasificación implícita | 0.00% | 21.21% | +21.21% |
| Longitud media | 11.3 palabras | 26.6 palabras | +136% |

El mismo patrón, llevado al extremo: **aprendió el formato a la perfección
(100%) y sigue sin saber calcular (1 acierto de 33)**. No es un fallo del
notebook: es un modelo de 248 M que produce la forma correcta con el contenido
equivocado. Es la demostración más nítida de por qué la métrica de formato tenía
que estar separada de la de exactitud.

### BETO — clasificación

| Sistema | Accuracy | F1-macro |
|---|---|---|
| **BETO fine-tuned** | **72.73%** (24/33) | **0.7223** |
| Baseline TF-IDF + regresión logística | 48.48% | 0.4479 |
| TF-IDF con validación cruzada 5-fold | — | 0.4134 ± 0.028 |
| BETO sin fine-tuning (cabeza aleatoria) | 9.09% | 0.0278 |
| Baseline clase mayoritaria | 9.09% | 0.0152 |

El punto de comparación que importa es **TF-IDF, no la clase mayoritaria**. Un
modelo lineal sobre n-gramas entrenado en dos segundos es la prueba de si vale
la pena pagar el costo de un transformer. BETO le saca **+27 puntos de
F1-macro** (0.7223 frente a 0.4479, y frente a 0.4134 con validación cruzada,
que es la estimación más estable). Eso sí justifica el transformer.

## 4.2 El pipeline: las tres configuraciones

| Configuración | Exactitud | IC 95% | Formato | ROUGE-L |
|---|---|---|---|---|
| **A · Qwen solo** | **93.94%** (31/33) | [85.8%, 100%] | 100% | 0.825 |
| **B · Pipeline real** | 87.88% (29/33) | [76.7%, 99.0%] | 100% | 0.724 |
| **C · Oráculo** | 90.91% (30/33) | [81.1%, 100%] | 100% | 0.736 |

**Beneficio del enrutamiento perfecto (C − A) = −3.03%**
**Aporte neto del pipeline real (B − A) = −6.06%**

Este es el resultado que decide la arquitectura de despliegue, y hay que leerlo
con cuidado porque es contraintuitivo.

**C está por debajo de A.** Es decir: incluso dándole al pipeline la categoría
verdadera —un clasificador perfecto, imposible en producción— el sistema es
peor que Qwen trabajando solo. **No hay ningún beneficio latente que un mejor
clasificador pudiera desbloquear.** El cuello de botella no es la accuracy del
72.73% de BETO: es que condicionar la generación por categoría no ayuda.

Las diferencias son de 1 y 2 ejemplos, dentro del margen de ruido para n=33, así
que la afirmación fuerte no es "el pipeline empeora" sino la más segura y
suficiente: **no hay evidencia de que aporte nada, ni siquiera en su mejor caso
teórico.**

### Por qué el enrutamiento no ayudó

Dos explicaciones, y conviene distinguirlas porque tienen implicaciones
distintas:

**1. Redundancia.** Qwen fine-tuned ya infiere la estrategia del enunciado.
Decirle "esto es de porcentajes, convierte el porcentaje a decimal y multiplica"
le informa de algo que ya dedujo de los 12 ejemplos de porcentajes que vio en
entrenamiento.

**2. Desajuste entre entrenamiento e inferencia.** *Esta es una limitación real
del diseño experimental y hay que declararla.* El adaptador LoRA se entrenó con
el prompt **sin** el bloque de estrategia. Las configuraciones B y C añaden ese
bloque al mensaje de sistema, así que se evalúan **fuera de la distribución con
la que el modelo fue afinado**. Parte de la degradación puede deberse a eso y no
a que el enrutamiento sea inútil en general.

Una prueba justa exigiría reentrenar Qwen con los prompts ya condicionados por
categoría. Queda como el experimento natural siguiente.

### Propagación de error y latencia

| | n | Exactitud del generador |
|---|---|---|
| Clasificación correcta | 24 | 91.67% |
| Clasificación incorrecta | 9 | 77.78% |

Equivocarse de categoría sí penaliza (−14 puntos), lo que confirma que la
estrategia inyectada **tiene efecto** sobre la generación — solo que el efecto
neto no es positivo. Con n=9 en el grupo de errores, la cifra es indicativa, no
una estimación.

**Latencia:** clasificación 3.5 ms frente a 2183 ms de generación. El enrutador
consume el **0.16%** del tiempo total: es esencialmente gratis. Ese dato importa
para la sección 6.

## 4.3 Tiempos y eficiencia del entrenamiento

| Modelo | Tiempo | Parámetros entrenados | % del total | Mejor época |
|---|---|---|---|---|
| FLAN-T5 | 105.7 s | 1 769 472 | 0.71% | 15 de 15 |
| Qwen2.5 | 129.4 s | 4 358 144 | 0.28% | **4 de 8** |
| BETO | 299.5 s | 109 859 339 | 100% | 8 de 15 |

Dos lecturas:

**BETO es el más lento pese a ser el más pequeño**, porque entrena *todos* sus
parámetros mientras los otros dos ajustan adaptadores. Es el argumento
cuantitativo a favor de LoRA: Qwen tiene 14 veces más parámetros que BETO y
entrena en menos de la mitad de tiempo.

**Qwen tocó fondo en la época 4 de 8.** Agotó lo aprendible del corpus en la
mitad del presupuesto. Eso no se arregla entrenando más épocas: se arregla con
más datos. **FLAN-T5, en cambio, seguía mejorando en la época 15 de 15**: se
quedó sin presupuesto de entrenamiento. Su 3.03% está por tanto medido en
condiciones algo desfavorables frente a Qwen, y es justo decirlo.

## 4.4 Análisis de tokenizadores

| | Qwen2.5 | BETO | FLAN-T5 |
|---|---|---|---|
| Algoritmo | BPE | WordPiece | SentencePiece |
| Vocabulario | 151 643 | 31 002 | 32 100 |
| Tokens/palabra | 2.189 | **1.489** | 2.298 |
| Tokens/ejemplo | 79.6 | 54.1 | 83.5 |
| Tasa `<unk>` | **0.000%** | 2.553% | 2.764% |

*(Medido sobre `entrada + salida` en crudo. Los notebooks individuales reportan
cifras distintas —Qwen 124.6 tokens/ejemplo, BETO 19.1— porque miden ámbitos
distintos: el prompt completo con plantilla de chat y solo el enunciado,
respectivamente. No es una contradicción.)*

Tres lecturas:

**El tamaño del vocabulario no es lo que importa; importa a qué idioma está
dedicado.** BETO tiene un vocabulario cinco veces menor que Qwen y aun así parte
el español en menos piezas (1.489 frente a 2.189 tokens por palabra), porque sus
31 k tokens son todos españoles.

**Qwen es el único sin pérdida de información.** Tasa de `<unk>` exactamente
cero: su vocabulario multilingüe cubre `÷`, `×`, `√`, `²` y las tildes. Ventaja
directa para este dominio y una de las razones de su elección.

**FLAN-T5 pierde caracteres imprescindibles.** El diagnóstico del notebook 4
identificó los que su tokenizador no puede representar:

```
¿  ¡  í  ú  ñ  Á  Í  Ó  Ú  Ñ  ×  ÷  √  π  Δ  ≈
```

Un `<unk>` no es una aproximación: es información destruida antes de llegar al
modelo. Si `÷` se convierte en `<unk>`, el modelo no puede distinguir
"864 ÷ 12" de "864 × 12".

**La normalización funcionó, pero no bastó.** Reescribiendo la notación
(`÷`→`/`, `×`→`x`, `√`→`raiz`, `²`→`^2`, sin tildes) la tasa de `<unk>` bajó de
**2.764% a 0.180%**, un 93% de reducción. Aun así FLAN-T5 obtuvo 3.03% de
exactitud. **El tokenizador no era su limitación principal**, y conviene decirlo
para no atribuirle al tokenizador un fracaso que tiene otras causas: tamaño del
modelo, preentrenamiento matemático débil y presupuesto de épocas insuficiente.

---

# 5. MODELO BASE ELEGIDO: Qwen2.5-1.5B-Instruct

## 5.1 La decisión

**Se elige `Qwen/Qwen2.5-1.5B-Instruct` con fine-tuning mediante LoRA, desplegado
como modelo único.**

## 5.2 Por qué, en orden de importancia

**1. Es el único que trae la competencia matemática.**
Este es el argumento decisivo, y sale de comparar los *baselines*, no los
resultados finales: Qwen resolvía el **90.91%** de los problemas antes de que lo
tocáramos; FLAN-T5, el **0%**. Ningún fine-tuning con 132 ejemplos crea
capacidad de cálculo desde cero — lo demuestra FLAN-T5, que tras entrenar sigue
en 3%. La capacidad tiene que venir del preentrenamiento, y solo Qwen la tiene.

**2. El fine-tuning aportó exactamente lo que faltaba: la forma.**
Formato del 12.12% al 100%, verbosidad reducida un 71%, ROUGE-L triplicado. La
combinación "modelo que ya sabe matemáticas + fine-tuning que le enseña a
explicar" es la que produce el tutor.

**3. Su tokenizador es el adecuado para el dominio.**
Cero `<unk>` sobre un corpus en español lleno de notación matemática. Ningún
entrenamiento recupera la información que el tokenizador destruye, así que esto
es un criterio de selección de primer orden y no un detalle de implementación.

**4. El costo de entrenamiento es bajo pese al tamaño.**
1 543 M de parámetros, pero LoRA entrena solo 4.36 M (0.28%). Entrena en 129
segundos en una T4 gratuita y el adaptador pesa ~20 MB, no el modelo entero.

**5. Escala sin cambiar el código.**
Existen variantes de 3B, 7B y 14B que se cargan con la misma línea. Es un camino
de mejora claro que ni BETO ni flan-t5-base ofrecen.

## 5.3 Tarjeta técnica

- **Nombre:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Licencia:** Apache 2.0
- **Entidad:** Alibaba Cloud (equipo Qwen)
- **Arquitectura:** decoder-only, 28 capas, dimensión oculta 1536, atención GQA
- **Vocabulario:** 151 643 tokens (BPE)
- **Adaptación:** LoRA r=16, alpha=32, dropout 0.05, lr 2e-4, 8 épocas
- **Pérdida de validación:** 0.1914 (mejor época: 4)
- **Artefacto:** ZIP 2 → `adaptadores/qwen-lora/` (~20 MB)

## 5.4 Limitaciones declaradas

- **No aprendió a calcular, y no hay evidencia de que pueda con este corpus.**
  La exactitud no cambió de forma significativa (p = 1.000). Sigue sin garantía
  de corrección aritmética: puede escribir un procedimiento impecable y
  equivocarse en la última operación.
- **Falla de forma consistente en `estadistica_probabilidad`**: 66.67% antes y
  después del fine-tuning, la única categoría que no llegó al 100%.
- **Es el más caro en inferencia** de los cuatro: 2.2 segundos por respuesta,
  generando token a token.
- **La medición tiene un margen amplio.** 33 ejemplos de validación: IC 95% de
  [85.8%, 100%]. Un ejemplo vale 3 puntos porcentuales.

## 5.5 Por qué no los otros tres

| Descartado | Razón |
|---|---|
| **BETO solo** | No genera texto. Un encoder-only no tiene cabeza de lenguaje autoregresiva: no puede escribir un procedimiento. Sirve como enrutador, no como tutor. |
| **FLAN-T5** | 3.03% de exactitud con 100% de formato: produce la forma correcta con el contenido equivocado. Conceptualmente es la arquitectura más elegante —clasificación y generación con una sola pérdida—, pero este modelo concreto no tiene la competencia matemática necesaria. |
| **Pipeline BETO → Qwen** | Descartado **por los datos**: C − A = −3.03%. Ni siquiera con un clasificador perfecto el enrutamiento mejora al modelo único. Ver sección 6. |

---

# 6. QUÉ HACER CON BERT

El pipeline no se recomienda como arquitectura de despliegue: el experimento del
oráculo (configuración C) demuestra que no hay beneficio que recuperar mejorando
el clasificador. Añadir un segundo modelo, un segundo tokenizador y un punto de
fallo silencioso —la alineación de etiquetas entre ambos— no se justifica con
estos números.

Eso **no** convierte a BETO en trabajo descartado. Aporta tres cosas medidas:

**1. Demuestra que la tarea de clasificación es aprendible.** 72.73% de accuracy
sobre 11 clases con 12 ejemplos de entrenamiento por clase, y +27 puntos de
F1-macro sobre el baseline clásico.

**2. Cuesta el 0.16% de la latencia.** 3.5 ms frente a los 2183 ms de la
generación. Como **instrumento de observabilidad** —saber qué tipos de problema
llegan, en cuáles falla el sistema, cuándo la confianza es baja— es
prácticamente gratis, aunque no toque la generación.

**3. Su patrón de errores enseña algo sobre el dominio.** Y es el hallazgo más
interesante del notebook 2.

## 6.1 Análisis de los errores de clasificación

BETO acertó **perfectamente (3/3)** en cuatro categorías y falló sobre todo en
las operaciones básicas:

| Perfectas (3/3) | Marcador léxico distintivo |
|---|---|
| `potencias_raices` | `√`, "al cuadrado", "elevar" |
| `porcentajes` | `%`, "porcentaje", "descuento" |
| `geometria` | "área", "perímetro", "radio", "ángulo" |
| `estadistica_probabilidad` | "probabilidad", "promedio", "mediana" |

| Peores (1/3) | Por qué |
|---|---|
| `multiplicacion` | Sin palabra clave propia |
| `ecuaciones` | Los enunciados verbales no dicen "ecuación" |

Los 9 errores lo confirman:

- `sum-08` "Camila ahorró 15000 el lunes y 22500 el martes" → predijo **division**
- `mul-04` "ahorra 12000 cada semana, ¿cuánto tras 8 semanas?" → predijo **division**
- `mul-09` "8500 por persona, ¿cuánto pagan 6?" → predijo **porcentajes**
- `div-09` "240 galletas en 16 paquetes" → predijo **multiplicacion**
- `ecu-01` "La suma de un número y 15 es igual a 42" → predijo **estadistica_probabilidad**
- `ecu-08` "El doble de un número menos 6 es igual a 18" → predijo **estadistica_probabilidad**

**El patrón es inequívoco: BETO clasifica bien cuando hay una palabra clave y
mal cuando hay que entender qué operación pide el enunciado.** Con 12 ejemplos
por clase aprendió atajos léxicos, no la semántica de la operación. Los casos
`ecu-01` y `ecu-08` son especialmente claros: son ecuaciones expresadas en
palabras, sin ningún `x` ni signo `=`, y el modelo no tiene de dónde agarrarse.

Es exactamente la limitación que se anticipó al diseñar el notebook, ahora
confirmada con datos. Y señala dónde invertir: más ejemplos de las cuatro
operaciones básicas expresadas en lenguaje natural.

---

# 7. LIMITACIONES DEL EXPERIMENTO

Declaradas explícitamente, porque condicionan cómo debe leerse todo lo anterior:

- **33 ejemplos de validación.** Un ejemplo vale 3 puntos porcentuales. El IC
  95% para una exactitud del 90% es aproximadamente [81%, 100%]. **Ninguna
  diferencia menor a ~10 puntos entre sistemas generativos es concluyente.**
- **El corpus es demasiado fácil.** Con un baseline del 90.91% no queda margen
  para medir mejora en capacidad de cálculo. Es la limitación que más condiciona
  las conclusiones.
- **3 ejemplos de validación por clase.** El recall por clase se mueve en saltos
  de 33 puntos. Sirve para detectar clases con recall 0, no para comparar clases
  entre sí.
- **El pipeline se evaluó fuera de distribución.** El adaptador de Qwen se
  entrenó sin el bloque de estrategia que las configuraciones B y C añaden al
  prompt. Parte de la degradación observada puede ser un artefacto de ese
  desajuste.
- **FLAN-T5 no agotó su presupuesto de entrenamiento** (mejor época 15 de 15),
  mientras Qwen convergió en la 4 de 8. La comparación le es algo desfavorable.
- **Una sola semilla.** No hay estimación de la varianza entre ejecuciones.
- **Las métricas no miden calidad pedagógica.** Un procedimiento correcto y uno
  *bien explicado* no son lo mismo, y nada aquí mide lo segundo.

## 7.1 Qué haría falta para un tutor desplegable

1. **Un corpus más difícil y más grande.** El proyecto está preparado para
   escalar: basta reemplazar `data/math_tutor_dataset.jsonl`.
2. **Verificación simbólica del cálculo.** Que el modelo genere el procedimiento
   y una herramienta externa (SymPy) valide la aritmética. Es la vía realista
   para el problema que el fine-tuning no resolvió.
3. **Evaluación humana** de la calidad de la explicación.
4. **Reentrenar el pipeline con prompts condicionados**, para saber si el
   enrutamiento aporta cuando se evalúa dentro de distribución.

---

# 8. REPRODUCIBILIDAD

## 8.1 Usar el modelo elegido

**ZIP 2 → `adaptadores/qwen-lora/`**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTADOR = "./ZIP2/adaptadores/qwen-lora"

tokenizer = AutoTokenizer.from_pretrained(ADAPTADOR)
modelo = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda")
modelo = PeftModel.from_pretrained(modelo, ADAPTADOR).merge_and_unload()
modelo.eval()

INSTRUCCION = ("Eres un tutor de matemáticas. Resuelve el siguiente problema "
               "explicando el procedimiento paso a paso y termina con la respuesta final.")

pregunta = "Un tanque contiene 250 litros y se utilizan 88. ¿Cuántos litros quedan?"
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": INSTRUCCION},
     {"role": "user", "content": pregunta}],
    tokenize=False, add_generation_prompt=True)

entrada = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
salida = modelo.generate(**entrada, max_new_tokens=200, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True))
```

El adaptador LoRA **no contiene el modelo base**: solo las matrices A y B. Por
eso hay que descargar Qwen y aplicar el adaptador encima. El prompt debe usar
`apply_chat_template` con esa misma instrucción: es la plantilla con la que se
entrenó, y cambiarla degrada el resultado.

## 8.2 Usar el clasificador

**ZIP 2 → `modelos/bert-clasificador/`**

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RUTA = "./ZIP2/modelos/bert-clasificador"
modelo = AutoModelForSequenceClassification.from_pretrained(RUTA)
tokenizer = AutoTokenizer.from_pretrained(RUTA)

# El modelo se entrenó SOLO con el enunciado del problema.
entrada = "Resuelve la ecuación: 2x + 5 = 19."
tokens = tokenizer(entrada, return_tensors="pt", truncation=True, max_length=64)
prediccion = modelo(**tokens).logits.argmax(-1).item()
print(modelo.config.id2label[prediccion])   # -> "ecuaciones"
```

## 8.3 Reentrenar desde cero

1. Abrir en Colab `ZIP1/ProyectoIA/S04_Lab_Fine_tuning_Qwen.ipynb`
2. Activar GPU T4
3. Ejecutar todas las celdas. **El corpus va embebido comprimido en el
   notebook**, con verificación SHA-256; no hace falta subir ningún archivo.

Orden de ejecución:

```
1 · Qwen  ──┐
            ├──> 3 · Pipeline ──┐
2 · BERT  ──┘                   ├──> 5 · Comparación
                                │
4 · FLAN-T5 ────────────────────┘
```

---

# 9. NOTAS TÉCNICAS

Tres incidencias encontradas durante la ejecución, documentadas porque afectan a
la reproducibilidad:

**`peft` + `torchao`.** Colab trae `torchao` preinstalado y `peft` comprueba su
versión con una función que lanza `ImportError` en vez de devolver `False`. El
resultado es que `get_peft_model()` falla con un error que no menciona LoRA por
ninguna parte. Solución aplicada: `pip uninstall -y torchao`.

**Qwen — los parámetros LoRA deben estar en fp32.** Con los pesos base en fp16 y
entrenamiento en precisión mixta, mantener los parámetros entrenables en fp16
provoca subdesbordamiento del gradiente y la pérdida diverge a `NaN`.

**FLAN-T5 — no admite fp16.** T5 fue preentrenado en bfloat16; sus activaciones
desbordan el rango de fp16 y la pérdida diverge. La T4 no soporta bfloat16, así
que se entrenó en fp32.

---

# 10. REFERENCIAS DE FUENTES

| Dato | Fuente |
|---|---|
| Corpus (165 ejemplos) | ZIP 2 → `data/math_tutor_dataset.jsonl` |
| Baseline y fine-tuned de Qwen | ZIP 2 → `resultados/qwen.json` |
| Métricas y errores de BETO | ZIP 2 → `resultados/bert.json` |
| Configuraciones A/B/C del pipeline | ZIP 2 → `resultados/pipeline_qwen_bert.json` |
| Diagnóstico del tokenizador de FLAN-T5 | ZIP 2 → `resultados/flan_t5.json` |
| Comparación de tokenizadores | ZIP 2 → `resultados/comparacion_final.json` |
| Adaptador LoRA de Qwen | ZIP 2 → `adaptadores/qwen-lora/` |
| Clasificador entrenado | ZIP 2 → `modelos/bert-clasificador/` |
| Logs de experimentos | ZIP 2 → `wandb/` (4 runs) |
| Código | ZIP 1 → `ProyectoIA/*.ipynb` |

# Plantilla de definición del proyecto integrador
Completa esta plantilla con tu equipo (3–4 personas). Es el documento que hace
crecer el proyecto módulo a módulo durante todo el semestre.
> **En la Sesión 1 solo se exigen los campos 1 y 2.** El resto se completa para
> la Sesión 2. No borres los campos vacíos: déjalos con su encabezado para
> irlos llenando.
Reemplaza cada bloque `> _...` con tu respuesta. Los ejemplos van entre
comentarios HTML y no se renderizan.
---
## 1. Dominio
<!-- Ejemplo: Atención al ciudadano en una alcaldía municipal. -->
> Tutor de matemáticas para estudiantes de secundaria (grados 9° a 11°), enfocado en álgebra y geometría básica. El sistema no reemplaza al docente: revisa el ejercicio que el estudiante ya resolvió (a mano o digitado) y da retroalimentación sobre el paso específico donde se equivocó, en lugar de solo decir si el resultado final está bien o mal.
---
## 2. Usuario + decisión
> **Usuario 1 — Estudiante** de secundaria (9°-11°) que resuelve ejercicios de álgebra o geometría en casa, sin el profesor presente para resolver dudas en el momento.
>
> **Decisión que cambia:** qué debe corregir primero en su procedimiento (por ejemplo: "el error no está en el resultado final, sino en cómo despejaste la variable en el paso 2"), en vez de solo saber que la respuesta quedó mal sin entender en qué parte del razonamiento falló. Hoy, sin el profesor al lado, o repite el ejercicio a ciegas o espera hasta la siguiente clase para preguntar; el sistema le permite corregir el paso exacto donde falló, en el momento en que lo necesita.
>
> **Usuario 2 — Profesor** que revisa los ejercicios de todo un curso (30+ estudiantes).
>
> **Decisión que cambia:** a qué estudiantes agrupar para un refuerzo puntual esta semana, según el tipo de error que más se repite en el curso (por ejemplo, despeje de variables vs. signos vs. jerarquía de operaciones), en vez de corregir ejercicio por ejercicio sin poder ver el patrón general y sin tiempo para planear una intervención grupal.
>
> Los dos usuarios comparten el mismo motor de análisis (detectar en qué paso del procedimiento está el error), pero lo usan para decisiones distintas: el estudiante corrige su propio ejercicio en el momento; el profesor prioriza a quién reforzar y con qué tema, a nivel de curso.
<!--
Ejemplo:
- Usuario: un funcionario de la ventanilla de atención.
- Decisión que cambia: a qué dependencia enrutar un trámite y qué documentos
  pedirle al ciudadano en el momento, en vez de mandarlo a averiguar y volver.
-->
---
## 3. Tarea del modelo (M1)
<!-- Ejemplo: clasificar el tipo de trámite a partir de la descripción libre
del ciudadano. -->
> Clasificación de texto multiclase. El modelo recibe el enunciado del ejercicio más el paso específico que escribió el estudiante (junto con el paso anterior como contexto), y devuelve una etiqueta con el tipo de error en ese paso, o "correcto" si no lo hay. Categorías propuestas: error de signo, error al despejar la variable, error de jerarquía de operaciones, error de simplificación, error conceptual (por ejemplo, mal teorema aplicado en geometría).
> 
> BETO — dccuchile/bert-base-spanish-wwm-cased: BERT entrenado desde cero en español, licencia CC BY 4.0, liviano para fine-tuning en Colab gratis. Opción principal. https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased
>
> PlanTL-GOB-ES/roberta-base-bne: RoBERTa en español entrenado por la Biblioteca Nacional de España, buena alternativa si quieren comparar resultados. https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne
---
## 4. Dataset + licencia
<!-- Ejemplo: 1.200 solicitudes históricas anonimizadas; licencia de uso
interno con permiso de la entidad. -->
> Dataset propio: ejercicios de álgebra y geometría de 9°-11° resueltos por estudiantes reales, anonimizados, recolectados con permiso de un profesor o estudiante conocido. Licencia: uso académico interno del curso, sin redistribución pública. Como referencia complementaria para construir la taxonomía de errores se consultó el dataset de la competencia *Eedi – Mining Misconceptions in Mathematics* (Kaggle), aunque está en inglés y en formato de opción múltiple, por lo que no se usa como fuente directa de entrenamiento.
---
## 5. Métrica de éxito
<!-- Ejemplo: F1 macro > 0.80 en enrutamiento; y que el funcionario acepte la
sugerencia en ≥ 70% de los casos en la prueba con usuarios. -->
> F1-macro > 0.75 en la clasificación del tipo de error por paso, contra un baseline de "clase mayoritaria". Como señal de valor real: que el estudiante confirme que el error señalado era el correcto (o corrija bien el paso después) en ≥ 70% de los casos en pruebas con usuarios, y que el profesor esté de acuerdo con la agrupación de errores que sugiere el sistema en ≥ 70% de los grupos propuestos en una prueba piloto con un curso.
---

## 6. Componente visual (M4)

<!-- Ejemplo: leer el documento escaneado que adjunta el ciudadano y verificar
que corresponde al trámite. -->

> _¿Qué aporta el componente multimodal/visual del Módulo 4 al sistema?_

---

## 7. Riesgos éticos

<!-- Ejemplo: sesgo contra solicitudes mal redactadas; riesgo de negar un
trámite por un error del modelo. Mitigación: el sistema sugiere, el funcionario
decide. -->

> _¿Qué puede salir mal para una persona real? ¿Cómo lo mitigas?_

---

## 8. Compromisos del equipo

<!-- Ejemplo: reuniones los martes; repositorio compartido; cada integrante es
dueño de un módulo pero todos revisan. -->

> _¿Cómo se organizan? ¿Quién responde por qué? ¿Cómo se comunican?_

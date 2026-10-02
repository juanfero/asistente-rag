# M10 — Evaluación con preguntas de prueba

**Estado:** ⬜ Pendiente · **Estimado:** 2.5 h · **Depende de:** M7 · **Requisito:** R11 (sección 4.4)

## Objetivo
Cumplir y superar el mínimo del caso (3 preguntas: contestable, parcial, no contestable) con un **set de 9 preguntas** (3 de cada tipo), ejecutarlas contra el sistema real y registrar **pregunta, respuesta generada, si fue correcta o no, y observación**.

## Set de preguntas (`evaluacion/preguntas.yaml`)
| ID | Tipo | Pregunta | Respuesta esperada / criterio |
|---|---|---|---|
| Q1 | Contestable | ¿Cuántos días de vacaciones tengo por año trabajado y con cuánta anticipación debo solicitarlas? | 15 días hábiles; 15 días calendario de anticipación en NexaPeople |
| Q2 | Contestable | ¿Cuál es el tope diario de alimentación en viajes nacionales e internacionales? | COP 120.000 nacional; USD 90 internacional |
| Q3 | Contestable | ¿Qué requisitos debe cumplir mi contraseña corporativa? | ≥12 caracteres, cambio cada 90 días, no reutilizar las últimas 5 |
| Q4 | Parcial | ¿Cuántos días de permiso me dan por matrimonio y cómo se pagan las horas extra? | 5 días hábiles; horas extra **no** están en los documentos |
| Q5 | Parcial | ¿Qué VPN debo usar y me prestan un celular corporativo? | FortiClient (`vpn.nexalogistica.co`); celular **no** aparece |
| Q6 | Parcial | ¿Cuál es el tope de hospedaje en Medellín y cuánto reconocen por kilómetro con vehículo propio? | COP 420.000/noche; kilometraje **no** aparece |
| Q7 | No contestable | ¿Cuál es el salario promedio de un analista en Nexa? | Mensaje de "no encontré" |
| Q8 | No contestable | ¿Cuántos días a la semana puedo trabajar remoto? | Mensaje de "no encontré" |
| Q9 | No contestable | ¿Quién es el CEO de Nexa Logística? | Mensaje de "no encontré" |

## Diseño
- `evaluacion/preguntas.yaml`: id, tipo, pregunta, `expected_facts` (lista de cadenas que deben aparecer, normalizando puntos/mayúsculas), `expected_sources`, `must_flag_missing` (parciales) / `expect_not_found` (no contestables).
- `scripts/run_evaluacion.py`: ingiere el corpus si el índice está vacío, ejecuta cada pregunta con `RAGEngine`, aplica una **verificación automática** y genera:
  - `evaluacion/resultados.json` (todo el detalle: respuesta, fuentes, scores, latencia, tokens).
  - `evaluacion/resultados.md`: tabla **Pregunta | Tipo | Respuesta generada | ¿Correcta? | Observación** + resumen (aciertos por tipo, latencia media).
- Verificación automática (heurística, documentada como tal):
  - Contestable: todos los `expected_facts` presentes **y** cita a la fuente esperada.
  - Parcial: hechos presentes **y** la respuesta señala la parte faltante (frases como "no se encuentra", "no aparece", "no hay información").
  - No contestable: `grounded=False` o contiene `NOT_FOUND_MESSAGE`.
- **Revisión humana obligatoria:** la columna "¿Correcta?" tiene valor automático y una columna "Revisión manual" que Juan Felipe confirma/corrige, con observación propia. Así se cumple el registro pedido por el caso y se demuestra criterio.
- Si algún caso falla: analizar causa (recuperación vs. generación), ajustar (umbral, top_k, prompt, chunking), **documentar el antes/después** en el resultado y en ADR.

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M10-01 | `preguntas.yaml` válido: 9 preguntas, 3 por tipo, campos obligatorios | `test_eval_dataset` |
| M10-02 | Las funciones de verificación automática clasifican bien casos sintéticos (FakeLLM) | `test_eval_checks` |
| M10-03 | El runner genera `resultados.json` y `resultados.md` con las columnas exigidas por el caso | `test_eval_report` (FakeLLM, `tmp_path`) |
| M10-04 *(integration)* | Ejecución real: ≥ 8/9 correctas automáticamente y **3/3 no contestables** sin alucinación | `scripts/run_evaluacion.py` |
| M10-05 *(manual)* | Columna de revisión manual y observaciones completadas | revisión |

## Verificación
```bash
pytest tests/unit/test_evaluacion.py -q
python scripts/run_evaluacion.py
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| | | | |

**Resultado final:** _(x/9 — contestables x/3, parciales x/3, no contestables x/3)_

# M10 — Evaluación con preguntas de prueba

**Estado:** ✅ Completado · **Estimado:** 2.5 h · **Depende de:** M7 · **Requisito:** R11 (sección 4.4)

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

### Decisiones del autor (2026-10-03)
- **A. Ejecución:** motor real (`RAGEngine` + Gemini, configuración efectiva) sobre el índice de `data/docs` (16 chunks); antes de evaluar se verifica que el índice tenga **solo** el corpus (si no, re-ingesta con reset).
- **B. Repeticiones:** `--repeticiones N` (evidencia con N=3); una pregunta es correcta solo si **todas** las repeticiones lo son; se reporta la consistencia (3/3, 2/3…). Pausa entre llamadas configurable (`--pausa`, 4 s); `LLMRateLimitError` → espera 60 s y reintenta una vez; `LLMQuotaExhaustedError` → se detiene y guarda lo avanzado.
- **C. Verificación automática** (`src/rag/evaluation.py`, reglas documentadas en el reporte): normalización (minúsculas, sin tildes, números sin separadores de miles); hechos con variantes; contestable = hechos + cita a la fuente (y página en PDF; Q2 exige págs. 1 y 2); parcial = hechos + frase estándar + palabra clave del tema faltante (Q4 `horas extra`, Q5 `celular`, Q6 `kilometr`); no contestable = `grounded=False` y `sources` vacío.
- **D. Revisión manual obligatoria (la hace el autor):** `evaluacion/revision_manual.yaml` (`correcta_manual: null`, `observacion_manual: ""`; el script nunca la sobrescribe); `--solo-reporte` regenera `resultados.md/json` sin llamar a Gemini. Columnas: ID | Tipo | Pregunta | Respuesta generada | ¿Correcta? (auto) | ¿Correcta? (manual) | Observación. Resumen: aciertos por tipo, consistencia, latencia p50/p95, tokens y nº de llamadas al LLM.
- **E. Bloque de robustez** (`evaluacion/preguntas_extra.yaml`, tabla aparte, N=1, no cuenta en M10-04): 3 paráfrasis, 1 fuera de dominio (debe cortarse por el umbral sin LLM) y 1 inyección en la pregunta.
- **F.** Si un caso falla: analizar recuperación vs. generación y **proponer** el ajuste (no aplicarlo sin visto bueno); validar todo ajuste también con las paráfrasis.
- **G. Archivos:** `evaluacion/preguntas.yaml`, `preguntas_extra.yaml`, `resultados.json`, `resultados.md`, `revision_manual.yaml`; `scripts/run_evaluacion.py`; `evidencias/M10_ejecucion.txt`.
- Los tokens se miden envolviendo el cliente LLM (`TokenTrackingLLM`) sin modificar el motor.

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
| 2026-10-03 | `pytest tests/unit/test_evaluacion.py -q` | 19 passed + 1 failed → 20 passed | El fallo era de la prueba: con FakeEmbedder el orden de recuperación no deja las págs. 1 y 2 del PDF en [1]/[2]; se reescribió con Q8 (regla independiente del orden) |
| 2026-10-03 | Verificación previa del índice | OK | 16 chunks, solo el corpus (sin subidas): no hizo falta re-ingerir |
| 2026-10-03 | `python scripts/run_evaluacion.py --repeticiones 3` | exit 0 | `evidencias/M10_ejecucion.txt`: 9/9 automáticas, consistencia 3/3 en todas, robustez 5/5; 31 llamadas a Gemini; 3 reintentos automáticos del SDK (uno con 38 s en Q3) |
| 2026-10-03 | Verificación propia de citas (fragmentos citados vs. texto del corpus) | OK | En las 9 preguntas con hechos (6 + 3 paráfrasis), cada fragmento citado contiene el dato que respalda (Q1 cita [3][4] para "15 días hábiles": ambos lo contienen por el solapamiento) |
| 2026-10-03 | `pytest -m "not integration" -v` | 389 passed, 12 deselected | `evidencias/M10_pytest_ruff.txt` |
| 2026-10-03 | `ruff check src tests && ruff format --check src tests` | All checks passed! / 43 files already formatted | |
| 2026-10-04 | Revisión manual del autor + `python scripts/run_evaluacion.py --solo-reporte` | 9/9 manual | Se añadió al resumen el total manual (`aciertos_manual`), que antes quedaba vacío |

**Estado de criterios:** M10-01 ✅ · M10-02 ✅ · M10-03 ✅ · M10-04 ✅ (9/9, no contestables 3/3) · M10-05 ✅ (revisión manual del autor: 9/9 correctas en `evaluacion/revision_manual.yaml`; reporte regenerado con `--solo-reporte`).

**Resultado final (automático):** 9/9 — contestables 3/3, parciales 3/3, no contestables 3/3 · consistencia 3/3 en las 9 · robustez 5/5 · latencia p50 7,7 s / p95 12,5 s · 26.118 tokens de entrada y 864 de salida · 31 llamadas al LLM. Detalle: `evaluacion/resultados.md`.

**F (ajustes):** no aplica — ningún caso falló, así que no se propone ningún cambio.

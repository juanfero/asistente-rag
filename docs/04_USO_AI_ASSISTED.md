# Uso de herramientas AI-assisted development

> Requisito 4.3 del caso: documentar **qué herramienta se usó, para qué, qué partes se revisaron manualmente y qué se aprendió.** Se completa al cierre de cada módulo.

## Herramientas
| Herramienta | Uso |
|---|---|
| Claude (claude.ai) | Análisis del caso técnico, documento general del proyecto y plan de módulos |
| Claude Code | Desarrollo módulo por módulo: generación de pruebas e implementación, ejecución de pytest/ruff, documentación |

## Forma de trabajo
1. Se definió primero la documentación (este repositorio `docs/`) con criterios de aceptación por módulo; Claude Code trabaja contra esos criterios, no "libremente".
2. Cada módulo: la IA propone pruebas + implementación → yo reviso el diff, ejecuto las pruebas, cuestiono decisiones y corrijo → solo se avanza con todo en verde.
3. Toda decisión relevante queda en `docs/03_DECISIONES.md`.

## Registro por módulo
| Módulo | Para qué se usó la IA | Qué revisé / corregí manualmente | Qué aprendí |
|---|---|---|---|
| Planeación | Lectura del caso, propuesta de arquitectura y plan de 12 módulos | Elección de LLM (Grok), embeddings locales, interfaz y SO; trazabilidad requisito → módulo | |
| M0 | Claude Code: lectura completa de la documentación y reporte de inconsistencias (endpoint `/stats` sin definir en M8, `pyyaml` no declarado, torch con CUDA por defecto, convención de nombres); generación de `pyproject.toml`, `Settings`, logging, `check_xai.py`, pruebas y registro de ejecución | Aprobé/decidí: torch CPU-only y cómo generar `requirements.txt` (ADR-006), atributos snake_case (ADR-007), quitar `/stats` del plan, `pyyaml` explícito, que M0-08 quede pendiente sin inventar una key. Revisé que `requirements.txt` no tenga `nvidia-*` y que la key no salga en `repr`/logs | `pydantic-settings` mapea variables en MAYÚSCULAS a atributos en minúscula; `SecretStr` oculta la key en `repr`/`str`/JSON; `logging.basicConfig(force=True)` borra el handler de `caplog` (la IA lo detectó por un test fallido y lo corrigió); un pin `+cpu` necesita `--extra-index-url` para reinstalarse |
| M1 | Claude Code: redacción del corpus ficticio (3 documentos con los hechos clave del plan), script reproducible del PDF, `Document`/loaders/`normalize_text` y 101 pruebas (hechos clave, temas excluidos, reproducibilidad del PDF) | Decidí: `path` relativo a la raíz del proyecto, NFC y `\xa0` en `normalize_text`, fecha fija en el PDF y verificación por SHA-256; aprobé dataclass, omitir `page` en md/txt y el tratamiento de ocultos. Revisé que el corpus no mencione los temas de las preguntas parciales/no contestables | La extracción de `pypdf` parte líneas según el ancho de la página (una cifra puede quedar "COP\n420.000"), así que conviene redactar para evitarlo y comparar con espacios normalizados; fpdf2 genera PDFs reproducibles si se fija la fecha de creación; una tilde puede venir "descompuesta" (letra + acento combinante) y NFC la unifica. La IA cometió un error en su propio test NFC (acento en la letra equivocada) que el test mismo detectó |
| M2 | Claude Code: chunker recursivo propio (separadores, fusión, solapamiento, ids sha1), detección y pegado de encabezados, script de inspección, 48 pruebas nuevas y evidencias | Definí: tests paramétricos con 800/120 y 500/80, fragmentar por página, prohibir chunks de solo encabezado y exigir la tabla Markdown completa en un chunk. Revisé el volcado completo de chunks (`evidencias/M2_chunks_corpus.txt`) | Un splitter recursivo puede dejar títulos huérfanos y hay que tratarlos explícitamente; conservar el separador en cada parte garantiza que cada chunk sea una subcadena contigua del original (sirve para probar que no se mezclan páginas). Como las pruebas pasaron al primer intento, la IA inyectó 3 defectos a propósito para comprobar que las pruebas los detectan |
| M3 | Claude Code: interfaz `Embedder`, embedder real con carga perezosa, `FakeEmbedder` con sha1, pruebas unitarias e integración, script de medición de tokens y diagnóstico de similitud con las 9 preguntas de M10 | Definí: no usar `hash()`, CPU explícito, sin prefijos, regla de elección del tamaño (mayor con ≤ 5 % truncado, consultar si < 400) y el diagnóstico temprano. Revisé la tabla de tokens y el chunk truncado | El límite real del modelo son 128 tokens (con 800 caracteres se truncaba el 62,5 %); el texto en español rinde ~4,1 caracteres por token en este tokenizer. La similitud sola no separa preguntas contestables de no contestables (Q8 0,59 > Q2 0,53), y una pregunta doble (Q2) puede dejar fuera del top-k la segunda mitad de la respuesta. La IA detectó y corrigió un aviso de API obsoleta (`get_sentence_embedding_dimension`) |
| M4 | | | |
| M5 | | | |
| M6 | | | |
| M7 | | | |
| M8 | | | |
| M9 | | | |
| M10 | | | |
| M11 | | | |

## Conclusiones (completar en M11)
- Beneficios observados:
- Riesgos / errores de la IA detectados:
- Prácticas que funcionaron (p. ej. pruebas primero, criterios de aceptación explícitos):

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
| M1 | | | |
| M2 | | | |
| M3 | | | |
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

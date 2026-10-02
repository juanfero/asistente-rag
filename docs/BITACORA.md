# Bitácora del proyecto

Una entrada por sesión de trabajo / cierre de módulo (más reciente arriba).

| Fecha | Módulo | Qué se hizo | Pruebas (pasan/total) | Horas | Próximo paso |
|---|---|---|---|---|---|
| 2026-10-02 | M0 | `git init`; venv con torch CPU (ADR-006); `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`; `Settings` snake_case (ADR-007) con validaciones y `SecretStr`; `setup_logging`; `scripts/check_xai.py`; tests M0; `evaluacion/.gitkeep`. **Corrección de documentación:** se quitó `/stats` de la fila M8 en `01_PLAN_MODULOS.md` (`/health` da el conteo de chunks y `GET /documents` el detalle por fuente; el comando `stats` de la CLI en M5 se mantiene). Se añadió `pyyaml` a las dependencias de M0 | 15/15 unitarias; integración 0/1 (1 skip: sin API key, M0-08 ⏸) | _(completar)_ | Proponer commit `feat(M0)` + tag `M0-ok`; luego M1 |
| 2026-10-01 | Planeación | Lectura del caso, documento general (`00_PROYECTO.md`), plan de 12 módulos con criterios de aceptación, CLAUDE.md, ADR-001/002 | — | 1 | Iniciar M0 |

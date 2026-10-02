# M12 — Opcional: Docker + despliegue cloud

**Estado:** ⬜ Opcional · **Estimado:** +3 h · **Depende de:** M11

> El caso dice: *"El uso de servicios cloud es deseable, pero no obligatorio."* Solo se ejecuta si M0–M11 están completos y sobra tiempo.

## Objetivo
Empaquetar la solución en contenedores y, opcionalmente, desplegarla en un servicio cloud.

## Tareas
1. `Dockerfile` (python:3.11-slim, pre-descarga del modelo de embeddings en build) y `docker-compose.yml` con servicios `api` (8000) y `ui` (8501), volumen para `data/`.
2. Variables por `env_file: .env`.
3. Opción de despliegue (elegir una y documentar en ADR): Render / Railway / Google Cloud Run / AWS App Runner. Considerar que Chroma persistente requiere volumen.
4. Documentar en README la sección "Despliegue".

## Criterios de aceptación
| ID | Criterio |
|---|---|
| M12-01 | `docker compose up --build` levanta API y UI; `/health` responde 200 |
| M12-02 | La evaluación (M10) produce los mismos resultados dentro del contenedor |
| M12-03 | *(si se despliega)* URL pública funcionando y enlazada en el README |

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| | | | |

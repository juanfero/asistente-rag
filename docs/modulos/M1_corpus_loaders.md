# M1 — Corpus de documentos y carga (loaders)

**Estado:** ⬜ Pendiente · **Estimado:** 3 h · **Depende de:** M0 · **Requisitos:** R1, R5

## Objetivo
1. Crear el corpus de la empresa ficticia **Nexa Logística S.A.S.**: 3 documentos internos (Markdown, PDF, TXT) con hechos concretos y verificables.
2. Implementar loaders que conviertan cada archivo en objetos `Document` con texto limpio y metadatos (fuente, tipo, página).

## Parte A — Corpus (`data/docs/`)
Los hechos clave de abajo son la **verdad de referencia** para la evaluación (M10). Redactar cada documento en español, tono de documento corporativo, ~1–2 páginas, con encabezados/secciones. Incluir todos los hechos listados (se pueden añadir detalles coherentes).

### `politica_vacaciones_y_permisos.md` (Markdown) — versión 2.1, vigente desde 2026-01-15
- 15 días hábiles de vacaciones por año trabajado; se pueden acumular máximo 2 periodos.
- Solicitud en el portal **NexaPeople** con mínimo **15 días calendario** de anticipación; aprueba el jefe inmediato en máximo 3 días hábiles.
- Se pueden fraccionar, pero un bloque debe ser de al menos 6 días hábiles.
- Permisos remunerados: matrimonio 5 días hábiles; calamidad doméstica hasta 5 días; mudanza 1 día por año.
- Licencia de paternidad 2 semanas; licencia de maternidad 18 semanas (según ley colombiana).
- Día de cumpleaños libre (debe tomarse en el mes del cumpleaños).
- **No** menciona: horas extra, trabajo remoto, salarios.

### `manual_reembolso_gastos.pdf` (PDF, ≥2 páginas) — generado con `scripts/generar_pdf_ejemplo.py` (fpdf2)
- Página 1: alcance; plazo para legalizar: **10 días hábiles** después de terminado el viaje; formato **FR-021**; soportes con factura electrónica a nombre de Nexa (NIT ficticio 900.123.456-7).
- Topes diarios de viáticos nacionales: alimentación **COP 120.000**, transporte local **COP 80.000**; hospedaje máximo **COP 350.000/noche** (Bogotá y Medellín **COP 420.000**).
- Página 2: viajes internacionales con tope de **USD 90/día** para alimentación; anticipos máximo 70% del presupuesto; aprobación: jefe inmediato hasta COP 2.000.000, gerente de área por encima; pago del reembolso en la siguiente quincena; gastos no reembolsables: bebidas alcohólicas, multas de tránsito, minibar.
- **No** menciona: kilometraje de vehículo propio, tarjetas corporativas.

### `guia_onboarding_ti.txt` (texto plano)
- Primer día: recoger portátil en Mesa de Ayuda piso 3; credenciales llegan al correo personal.
- Contraseña: mínimo **12 caracteres**, cambio cada **90 días**, no reutilizar las últimas 5.
- MFA obligatorio con **Microsoft Authenticator**.
- VPN: **FortiClient**, servidor `vpn.nexalogistica.co`, obligatoria fuera de la oficina.
- Soporte: correo `soporte@nexalogistica.co`, extensión **4040**, horario L–V 7:00–18:00; tiempo de respuesta 4 horas hábiles para incidentes normales, 1 hora para críticos.
- Prohibido instalar software no autorizado; solicitudes por el portal **NexaDesk**.
- **No** menciona: préstamo de equipos para casa, celulares corporativos.

## Parte B — Código
- `src/rag/models.py` (dataclasses o pydantic):
  - `Document(text: str, metadata: dict)` — metadata: `source` (nombre de archivo), `doc_type` (`md|pdf|txt`), `page` (int, solo PDF; 1-based), `path`.
- `src/rag/loaders.py`:
  - `load_file(path) -> list[Document]`: despacha por extensión (`.txt`, `.md`, `.markdown`, `.pdf`; case-insensitive).
    - TXT/MD: lectura UTF-8 (fallback `latin-1`), 1 `Document` por archivo.
    - PDF: `pypdf`, 1 `Document` por página con texto; páginas vacías se omiten con `warning`.
  - `load_directory(dir) -> list[Document]`: recorre el directorio (no recursivo por defecto, parámetro `recursive`), ignora archivos ocultos y no soportados (log `warning`), orden determinista por nombre.
  - Limpieza mínima `normalize_text()`: normaliza saltos de línea, colapsa espacios múltiples, elimina caracteres de control; **conserva** la estructura de párrafos.
  - Errores: `UnsupportedFileTypeError` para extensiones no soportadas en `load_file`; `FileNotFoundError` si no existe; archivo vacío → lista vacía + warning.

## Criterios de aceptación
| ID | Criterio | Test |
|---|---|---|
| M1-01 | Existen los 3 documentos en `data/docs/` y el PDF tiene ≥2 páginas con texto extraíble | `test_corpus_exists` |
| M1-02 | Cada hecho clave aparece literalmente en el texto cargado (p. ej. "15 días hábiles", "COP 120.000", "FortiClient", "4040") | `test_corpus_key_facts` (parametrizado) |
| M1-03 | `load_file` de `.md` y `.txt` devuelve 1 Document con `source` y `doc_type` correctos | `test_load_text_and_markdown` |
| M1-04 | `load_file` de PDF devuelve 1 Document por página con `page` 1-based | `test_load_pdf_pages` |
| M1-05 | Extensión no soportada → `UnsupportedFileTypeError`; ruta inexistente → `FileNotFoundError` | `test_errors` |
| M1-06 | Archivo vacío → lista vacía, sin excepción | `test_empty_file` |
| M1-07 | `load_directory` ignora `.docx`/ocultos y devuelve en orden determinista | `test_load_directory` (usa `tmp_path`) |
| M1-08 | `normalize_text` colapsa espacios y conserva párrafos | `test_normalize_text` |
| M1-09 | Archivo con codificación latin-1 se carga sin error | `test_latin1_fallback` |

## Verificación
```bash
python scripts/generar_pdf_ejemplo.py
pytest tests/unit/test_loaders.py tests/unit/test_corpus.py -q
pytest -m "not integration" -q && ruff check src tests
```

## Registro de ejecución
| Fecha | Comando / acción | Resultado | Notas |
|---|---|---|---|
| | | | |

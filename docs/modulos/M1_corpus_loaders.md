# M1 — Corpus de documentos y carga (loaders)

**Estado:** ✅ Completado · **Estimado:** 3 h · **Depende de:** M0 · **Requisitos:** R1, R5

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

### `manual_reembolso_gastos.pdf` (PDF, ≥2 páginas) — generado con `scripts/generar_pdf_ejemplo.py` (fpdf2, fuente Helvetica latin-1, fecha de creación fija → reproducible byte a byte)
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
  - *Decisiones aprobadas (2026-10-02):* `Document` es `@dataclass`; la clave `page` se **omite** (no `None`) en md/txt; `path` es la ruta **relativa a la raíz del proyecto** (p. ej. `data/docs/x.md`), nunca absoluta; si el archivo está fuera del proyecto se usa solo el nombre del archivo.
- `src/rag/loaders.py`:
  - `load_file(path) -> list[Document]`: despacha por extensión (`.txt`, `.md`, `.markdown`, `.pdf`; case-insensitive).
    - TXT/MD: lectura UTF-8 (fallback `latin-1`), 1 `Document` por archivo.
    - PDF: `pypdf`, 1 `Document` por página con texto; páginas vacías se omiten con `warning`.
  - `load_directory(dir) -> list[Document]`: recorre el directorio (no recursivo por defecto, parámetro `recursive`), ignora archivos ocultos y no soportados (log `warning`), orden determinista por nombre.
  - Limpieza mínima `normalize_text()`: normaliza saltos de línea, colapsa espacios múltiples, elimina caracteres de control; **conserva** la estructura de párrafos. Además (decisión 2026-10-02): `unicodedata.normalize("NFC")`, `\xa0` → espacio normal, recorte de espacios por línea y máximo una línea en blanco entre párrafos; no se elimina la sintaxis Markdown.
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
| 2026-10-02 | Redacción de `politica_vacaciones_y_permisos.md` y `guia_onboarding_ti.txt` | OK | Encabezado "Documento ficticio para prueba técnica." Se evitó "conexión remota" en la guía de TI para no dar pistas falsas a Q8 (trabajo remoto) |
| 2026-10-02 | `python scripts/generar_pdf_ejemplo.py` | OK, 2 páginas | 1ª versión partía "COP\n420.000" entre líneas; se reescribió esa viñeta para que ninguna cifra quede cortada (verificado buscando líneas que terminan en COP/USD/NIT/FR-) |
| 2026-10-02 | `sha256sum` → regenerar → `sha256sum` | Idéntico | `e45d3ece5d70818a1120b274e4abe4bef1ee71ec5caf2dd1921a214e5e3a9992` antes y después (`set_creation_date` = 2026-01-15T00:00:00Z). Cubierto también por `test_pdf_is_reproducible` |
| 2026-10-02 | Verificación PDF con `pypdf` | 2 páginas con texto | Página 1: 1156 caracteres; página 2: 824 |
| 2026-10-02 | `load_directory("data/docs")` | 4 Documents | txt (2624 chars), pdf p.1 (1156), pdf p.2 (824), md (3188); `path` relativo (`data/docs/...`); `.gitkeep` ignorado |
| 2026-10-02 | `pytest -m "not integration" -v` | 101 passed, 1 deselected | 1ª ejecución: 1 fallo en `test_normalize_text_nfc` por un error del propio test (acento combinante después de la "i" en lugar de la "o"); corregido y reescrito con escapes `\u` explícitos |
| 2026-10-02 | `ruff check src tests && ruff format --check src tests` | All checks passed! / 10 files already formatted (12 incluyendo `scripts/`) | `ruff format` reformateó 2 archivos de test (líneas > 100) |

**Estado de criterios:** M1-01 ✅ · M1-02 ✅ (56 hechos parametrizados: 21 md, 18 pdf, 17 txt) · M1-03 ✅ · M1-04 ✅ · M1-05 ✅ · M1-06 ✅ · M1-07 ✅ · M1-08 ✅ · M1-09 ✅.

**Pruebas adicionales:** `test_path_metadata` (decisión c), `test_normalize_text_nfc` y `test_normalize_text_nbsp` (decisión d), `test_pdf_is_reproducible` (decisión g), `test_corpus_excluded_topics` (los temas de las preguntas parciales/no contestables de M10 no aparecen en el corpus), `test_pdf_facts_by_page` (los hechos quedan en la página documentada) y `test_extensions_case_insensitive`.

**SHA-256 del PDF:** `e45d3ece5d70818a1120b274e4abe4bef1ee71ec5caf2dd1921a214e5e3a9992` (fpdf2 2.8.9)

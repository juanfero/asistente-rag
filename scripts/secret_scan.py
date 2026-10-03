"""Detecta claves de API en archivos del repositorio (versionados o en stage).

Patrones: claves de Google "AIza…" (39 caracteres), tokens "AQ.…" y líneas
`GEMINI_API_KEY=<valor no vacío>` (incluido .env.example, que debe quedar vacío).
Los hallazgos nunca muestran la clave: solo archivo, línea y tipo.

Uso: python scripts/secret_scan.py            # archivos versionados (git ls-files)
     python scripts/secret_scan.py --staged   # archivos en stage (hook de pre-commit)
     git log --all -p | python scripts/secret_scan.py --stdin   # historial (solo cuenta)
Código de salida: 0 = limpio, 1 = se encontró al menos una clave.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PATTERNS: dict[str, re.Pattern[str]] = {
    "clave Google (AIza…)": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "token (AQ.…)": re.compile(r"AQ\.[0-9A-Za-z_-]{20,}"),
}
ENV_LINE = re.compile(r"^\s*(?:export\s+)?GEMINI_API_KEY\s*=(.*)$")


def _env_value(raw: str) -> str:
    """Valor de una asignación tipo .env sin comentario ni comillas."""
    return raw.split("#", 1)[0].strip().strip("'\"").strip()


def find_secrets(text: str) -> list[tuple[int, str]]:
    """(línea, tipo) de cada posible clave en el texto; nunca devuelve la clave."""
    findings: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append((number, kind))
        match = ENV_LINE.match(line)
        if match and _env_value(match.group(1)):
            findings.append((number, "GEMINI_API_KEY con valor"))
    return findings


def _git(args: list[str], root: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def tracked_files(root: Path) -> dict[str, str]:
    """Contenido de los archivos versionados (texto; binarios decodificados con reemplazo)."""
    files = {}
    for name in _git(["ls-files"], root).splitlines():
        path = root / name
        if path.is_file():
            files[name] = path.read_bytes().decode("utf-8", errors="replace")
    return files


def staged_files(root: Path) -> dict[str, str]:
    """Contenido en stage (índice) de los archivos añadidos o modificados."""
    names = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"], root).splitlines()
    files = {}
    for name in names:
        blob = subprocess.run(["git", "show", f":{name}"], cwd=root, capture_output=True)
        files[name] = blob.stdout.decode("utf-8", errors="replace")
    return files


def scan(files: dict[str, str]) -> list[str]:
    """Hallazgos como 'archivo:línea: tipo'."""
    return [
        f"{name}:{line}: {kind}"
        for name, text in sorted(files.items())
        for line, kind in find_secrets(text)
    ]


def count_stdin(stream) -> int:
    """Nº de coincidencias en un flujo de texto (p. ej. el historial de git)."""
    return sum(len(find_secrets(line)) for line in stream)


def main() -> int:
    """Punto de entrada: imprime hallazgos y devuelve 1 si hay alguno."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true", help="revisa solo los archivos en stage")
    mode.add_argument(
        "--stdin", action="store_true", help="lee texto de stdin y solo informa el conteo"
    )
    args = parser.parse_args()
    if args.stdin:
        found = count_stdin(sys.stdin)
        print(f"{found} coincidencias")
        return 1 if found else 0
    root = Path(_git(["rev-parse", "--show-toplevel"], Path.cwd()).strip())
    findings = scan(staged_files(root) if args.staged else tracked_files(root))
    if findings:
        print("BLOQUEADO: posibles claves de API (no se muestran los valores):", file=sys.stderr)
        for item in findings:
            print(f"  {item}", file=sys.stderr)
        print(
            "Quita la clave (va solo en .env, ignorado por git) y vuelve a intentar.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

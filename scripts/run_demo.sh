#!/usr/bin/env bash
# Levanta la API (puerto tomado de API_URL) y la UI Streamlit; Ctrl+C detiene ambas.
# Uso: scripts/run_demo.sh            (desde la raíz del proyecto o cualquier carpeta)
#      UI_PORT=8502 scripts/run_demo.sh   (si el puerto 8501 está ocupado)
# Los PID se guardan al lanzar cada proceso y se detienen con `kill <PID>` (sin pkill).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

API_URL="$(python -c 'from rag.api_client import UISettings; print(UISettings().api_url)')"
API_PORT="$(python -c 'import sys; from urllib.parse import urlparse; u = urlparse(sys.argv[1]); print(u.port or 8000)' "$API_URL")"
UI_PORT="${UI_PORT:-8501}"
LOG_DIR="${TMPDIR:-/tmp}/rag_demo_logs"
mkdir -p "$LOG_DIR"

API_PID=""
UI_PID=""
cleanup() {
  trap - INT TERM EXIT
  echo ""
  echo "Deteniendo la demo…"
  for pid in "$UI_PID" "$API_PID"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  echo "Listo: API y UI detenidas."
}
trap cleanup INT TERM EXIT

port_busy() {
  python -c 'import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(0)
sys.exit(1)' "$1"
}
if port_busy "$API_PORT"; then
  echo "ERROR: el puerto ${API_PORT} (API) ya está en uso (otra API u otro servicio)."
  echo "Cambia API_URL en .env (p. ej. http://localhost:8011) o detén ese servicio."
  exit 1
fi
if port_busy "$UI_PORT"; then
  echo "ERROR: el puerto ${UI_PORT} (UI) ya está en uso. Usa otro: UI_PORT=8502 scripts/run_demo.sh"
  exit 1
fi

echo "Levantando la API en http://localhost:${API_PORT} (logs: ${LOG_DIR}/api.log)…"
uvicorn rag.api:app --workers 1 --port "${API_PORT}" >"${LOG_DIR}/api.log" 2>&1 &
API_PID=$!

for _ in $(seq 1 120); do
  if curl -s -m 2 "http://localhost:${API_PORT}/health" 2>/dev/null | grep -q '"llm_model"'; then
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "ERROR: la API no arrancó. Revisa ${LOG_DIR}/api.log"
    exit 1
  fi
  sleep 1
done

echo "Levantando la UI en http://localhost:${UI_PORT} (logs: ${LOG_DIR}/ui.log)…"
API_URL="$API_URL" streamlit run src/rag/ui_streamlit.py --server.headless true \
  --server.port "${UI_PORT}" >"${LOG_DIR}/ui.log" 2>&1 &
UI_PID=$!

echo ""
echo "  API (Swagger): http://localhost:${API_PORT}/docs   [PID ${API_PID}]"
echo "  UI:            http://localhost:${UI_PORT}         [PID ${UI_PID}]"
echo "  Presiona Ctrl+C para detener ambas."
wait

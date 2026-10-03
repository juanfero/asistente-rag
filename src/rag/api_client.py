"""Cliente HTTP de la API para la UI (la UI no importa el motor: separación cliente/servidor)."""

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from pydantic_settings import BaseSettings, SettingsConfigDict

TIMEOUT_HEALTH = 5
TIMEOUT_ASK = 60
TIMEOUT_DOCUMENTS = 120  # /documents (listar, subir, borrar) e /ingest


class UISettings(BaseSettings):
    """Configuración de la UI: solo la URL de la API (no lee GEMINI_API_KEY)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_url: str = "http://localhost:8000"


class APIClientError(Exception):
    """Error de la API con su código (campo "error"), mensaje y estado HTTP."""

    def __init__(self, code: str, detail: str, status: int | None = None) -> None:
        super().__init__(detail)
        self.code, self.detail, self.status = code, detail, status


@dataclass
class UploadResult:
    """Resultado de subir un archivo (una petición por archivo para informar por separado)."""

    name: str
    ok: bool
    report: dict[str, Any] | None = None
    code: str | None = None
    detail: str | None = None


def start_command(base_url: str) -> str:
    """Comando para levantar la API en el puerto de `base_url`."""
    port = urlparse(base_url).port or 8000
    return f"uvicorn rag.api:app --workers 1 --port {port}"


class APIClient:
    """Llamadas a la API con timeouts y errores traducidos a `APIClientError`."""

    def __init__(self, base_url: str, session: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, timeout: float, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=timeout, **kwargs)
        except requests.Timeout as exc:
            raise APIClientError(
                "API_TIMEOUT",
                f"La API no respondió en {timeout} s. Intenta de nuevo en unos segundos.",
            ) from exc
        except requests.ConnectionError as exc:
            raise APIClientError(
                "API_UNAVAILABLE",
                f"No se pudo conectar con la API en {self.base_url}. Levántala con: "
                f"{start_command(self.base_url)}",
            ) from exc
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.status_code >= 400:
            if isinstance(data, dict) and "error" in data:
                raise APIClientError(
                    data["error"], str(data.get("detail", "")), response.status_code
                )
            raise APIClientError(
                "API_ERROR",
                f"La API respondió con un error (HTTP {response.status_code}).",
                response.status_code,
            )
        return data

    def health(self) -> dict[str, Any]:
        """Estado del servicio; detecta si la URL responde pero no es la API del asistente."""
        data = self._request("GET", "/health", TIMEOUT_HEALTH)
        if not isinstance(data, dict) or not {"llm_model", "chunks", "top_k"} <= set(data):
            raise APIClientError(
                "API_UNEXPECTED",
                f"{self.base_url} respondió, pero no es la API del asistente (¿otro servicio en "
                f"ese puerto?). Revisa API_URL en .env o levanta la API con: "
                f"{start_command(self.base_url)}",
            )
        return data

    def list_documents(self) -> list[dict[str, Any]]:
        """Documentos indexados con su origen."""
        return self._request("GET", "/documents", TIMEOUT_DOCUMENTS)

    def upload(self, files: list[tuple[str, bytes]]) -> list[UploadResult]:
        """Sube cada archivo por separado y devuelve un resultado por archivo."""
        results = []
        for name, data in files:
            try:
                report = self._request(
                    "POST", "/documents", TIMEOUT_DOCUMENTS, files=[("files", (name, data))]
                )
                results.append(UploadResult(name=name, ok=True, report=report))
            except APIClientError as exc:
                results.append(UploadResult(name=name, ok=False, code=exc.code, detail=exc.detail))
        return results

    def ingest(self, reset: bool = False) -> dict[str, Any]:
        """Re-indexa data/docs + data/uploads."""
        return self._request("POST", "/ingest", TIMEOUT_DOCUMENTS, json={"reset": reset})

    def delete_document(self, source: str, delete_file: bool = True) -> dict[str, Any]:
        """Quita un documento del índice (y el archivo si es una subida)."""
        return self._request(
            "DELETE",
            f"/documents/{source}",
            TIMEOUT_DOCUMENTS,
            params={"delete_file": str(delete_file).lower()},
        )

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """Pregunta al asistente."""
        payload: dict[str, Any] = {"question": question}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request("POST", "/ask", TIMEOUT_ASK, json=payload)


def get_client(base_url: str | None = None) -> APIClient:
    """Cliente configurado con API_URL (variables de entorno o .env)."""
    return APIClient(base_url or UISettings().api_url)

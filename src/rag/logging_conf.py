"""Configuración de logging del proyecto."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"

# Librerías que registran cada petición HTTP o detalle de carga en INFO (ruido en la CLI)
NOISY_LOGGERS = (
    "httpx",
    "httpx2",
    "httpcore",
    "openai",
    "huggingface_hub",
    "sentence_transformers",
    "urllib3",
)


# El SDK openai registra los reintentos (p. ej. tras un 503) en INFO desde este logger
OPENAI_RETRY_LOGGER = "openai._base_client"


class RetryAsWarning(logging.Filter):
    """Deja pasar solo los avisos de reintento del SDK openai y los eleva a WARNING."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.getMessage().startswith("Retrying request"):
            return record.levelno >= logging.WARNING
        record.levelno, record.levelname = logging.WARNING, "WARNING"
        return True


def setup_logging(level: str | int = "INFO") -> None:
    """Configura el logger raíz con el formato del proyecto (reemplaza handlers previos).

    Las librerías ruidosas quedan en WARNING; los reintentos del SDK openai (p. ej. ante un 503)
    se muestran como WARNING.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    retry_logger = logging.getLogger(OPENAI_RETRY_LOGGER)
    retry_logger.setLevel(logging.INFO)
    if not any(isinstance(f, RetryAsWarning) for f in retry_logger.filters):
        retry_logger.addFilter(RetryAsWarning())

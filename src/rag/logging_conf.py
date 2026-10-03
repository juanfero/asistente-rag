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


def setup_logging(level: str | int = "INFO") -> None:
    """Configura el logger raíz con el formato del proyecto (reemplaza handlers previos).

    Las librerías ruidosas quedan en WARNING para que la salida muestre solo lo relevante.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

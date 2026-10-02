"""Configuración de logging del proyecto."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    """Configura el logger raíz con el formato del proyecto (reemplaza handlers previos)."""
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)

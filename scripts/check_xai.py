"""Verifica la conexión con Grok (xAI): lista modelos y hace una completion mínima.

Uso: python scripts/check_xai.py
Códigos de salida: 0 = OK, 1 = falta XAI_API_KEY o error de conexión/modelo.
"""

import sys

from openai import OpenAI, OpenAIError

from rag.config import Settings, get_settings


def check(settings: Settings) -> int:
    """Ejecuta la verificación y devuelve el código de salida."""
    if settings.xai_api_key is None or not settings.xai_api_key.get_secret_value():
        print(
            "ERROR: falta XAI_API_KEY. Defínela en el archivo .env (ver .env.example) "
            "o como variable de entorno y vuelve a ejecutar.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(
        api_key=settings.xai_api_key.get_secret_value(),
        base_url=settings.xai_base_url,
        timeout=60,
    )
    try:
        model_ids = sorted(m.id for m in client.models.list())
        print(f"Modelos disponibles ({len(model_ids)}):")
        for model_id in model_ids:
            print(f"  - {model_id}")

        if settings.xai_model not in model_ids:
            print(
                f"ERROR: XAI_MODEL='{settings.xai_model}' no está entre los modelos disponibles. "
                "Elige uno de la lista y actualiza .env.",
                file=sys.stderr,
            )
            return 1

        response = client.chat.completions.create(
            model=settings.xai_model,
            messages=[{"role": "user", "content": "Responde solo: OK"}],
            temperature=0,
            max_tokens=20,
        )
    except OpenAIError as exc:
        print(
            f"ERROR: no se pudo conectar con xAI ({type(exc).__name__}). "
            "Revisa la key, el saldo y la conexión.",
            file=sys.stderr,
        )
        return 1

    text = (response.choices[0].message.content or "").strip()
    print(f"Modelo '{settings.xai_model}' respondió: {text!r}")
    return 0 if text else 1


def main() -> int:
    """Punto de entrada del script."""
    return check(get_settings())


if __name__ == "__main__":
    sys.exit(main())

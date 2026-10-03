"""Verifica la conexión con Gemini (endpoint compatible con OpenAI).

1. Lista los modelos disponibles para la key y confirma que GEMINI_MODEL está entre ellos.
2. Hace una completion corta (max_tokens bajo) y otra con LLM_MAX_TOKENS, y revisa que la
   respuesta no venga vacía ni con finish_reason="length" (los modelos que "piensan" gastan
   max_tokens en razonamiento).

Nunca imprime la API key: solo "****" + sus últimos 4 caracteres.
Uso: python scripts/check_gemini.py      Códigos de salida: 0 = OK, 1 = error.
"""

import sys

from openai import APIStatusError, OpenAI, OpenAIError

from rag.config import Settings, get_settings

SHORT_MAX_TOKENS = 20
PROMPT = "Responde solo: OK"


def mask(secret: str) -> str:
    """Enmascara una clave dejando visibles solo sus últimos 4 caracteres."""
    return "****" + secret[-4:] if len(secret) >= 8 else "****"


def safe_error(exc: Exception, secret: str) -> str:
    """Tipo, código HTTP y mensaje del error, sin la clave."""
    status = getattr(exc, "status_code", None)
    message = str(getattr(exc, "message", "") or exc)
    if secret:
        message = message.replace(secret, mask(secret))
    code = f" HTTP {status}" if status else ""
    return f"{type(exc).__name__}{code}: {message[:300]}"


def model_ids(client: OpenAI) -> list[str]:
    """Ids de los modelos disponibles, sin el prefijo 'models/'."""
    return sorted(m.id.split("/")[-1] for m in client.models.list())


def completion_report(client: OpenAI, model: str, max_tokens: int) -> dict:
    """Hace una completion mínima y devuelve texto, finish_reason y uso de tokens."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return {
        "max_tokens": max_tokens,
        "text": (choice.message.content or "").strip(),
        "finish_reason": choice.finish_reason,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
    }


def print_report(r: dict) -> None:
    """Imprime el resultado de una completion."""
    print(
        f"  max_tokens={r['max_tokens']}: texto={r['text']!r} · finish_reason={r['finish_reason']}"
        f" · prompt_tokens={r['prompt_tokens']} · completion_tokens={r['completion_tokens']}"
        f" · reasoning_tokens={r['reasoning_tokens']}"
    )


def check(settings: Settings) -> int:
    """Ejecuta la verificación y devuelve el código de salida."""
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    if not key:
        print(
            "ERROR: falta GEMINI_API_KEY. Defínela en el archivo .env de la raíz del proyecto "
            "(ver .env.example) y vuelve a ejecutar.",
            file=sys.stderr,
        )
        return 1

    print(f"API key: {mask(key)} · base_url: {settings.gemini_base_url}")
    client = OpenAI(api_key=key, base_url=settings.gemini_base_url, timeout=60, max_retries=1)
    try:
        ids = model_ids(client)
        print(f"Modelos disponibles ({len(ids)}):")
        for model_id in ids:
            print(f"  - {model_id}")
        if settings.gemini_model not in ids:
            print(
                f"ERROR: GEMINI_MODEL='{settings.gemini_model}' no está entre los modelos "
                "disponibles para esta key.",
                file=sys.stderr,
            )
            return 1
        print(f"Modelo configurado '{settings.gemini_model}': disponible ✔")

        print("Completions de prueba:")
        short = completion_report(client, settings.gemini_model, SHORT_MAX_TOKENS)
        print_report(short)
        normal = completion_report(client, settings.gemini_model, settings.llm_max_tokens)
        print_report(normal)
    except (APIStatusError, OpenAIError) as exc:
        print(f"ERROR al llamar a Gemini: {safe_error(exc, key)}", file=sys.stderr)
        return 1

    if not short["text"] or short["finish_reason"] == "length":
        print(
            f"AVISO: con max_tokens={SHORT_MAX_TOKENS} la respuesta vino vacía o cortada "
            f"(finish_reason={short['finish_reason']}). Posible consumo de tokens de "
            "razonamiento: revisar ADR-003."
        )
    if not normal["text"] or normal["finish_reason"] == "length":
        print("ERROR: la completion con LLM_MAX_TOKENS vino vacía o cortada.", file=sys.stderr)
        return 1
    print("OK: conexión con Gemini verificada.")
    return 0


def main() -> int:
    """Punto de entrada del script."""
    return check(get_settings())


if __name__ == "__main__":
    sys.exit(main())

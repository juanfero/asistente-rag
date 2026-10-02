"""Genera data/docs/manual_reembolso_gastos.pdf (2 páginas) con fpdf2.

La fecha de creación es constante para que el PDF sea reproducible byte a byte.
Uso: python scripts/generar_pdf_ejemplo.py [ruta_salida]
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF

OUTPUT = Path("data/docs/manual_reembolso_gastos.pdf")
CREATION_DATE = datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

HEADER = [
    "Nexa Logística S.A.S. - Gerencia Financiera",
    "Versión 3.0 - Vigente desde 2026-01-15 - Código FIN-MAN-007",
    "Documento ficticio para prueba técnica.",
]

# Cada sección: (título, [párrafos]). Los párrafos que empiezan por "- " son viñetas.
PAGE_1 = [
    (
        "1. Alcance",
        [
            "Este manual define las reglas para solicitar anticipos y legalizar gastos de viaje "
            "de los colaboradores de Nexa Logística S.A.S. Aplica a viajes nacionales e "
            "internacionales realizados por motivos de trabajo.",
        ],
    ),
    (
        "2. Legalización de gastos",
        [
            "- El plazo para legalizar los gastos es de 10 días hábiles después de terminado "
            "el viaje.",
            "- La legalización se presenta en el formato FR-021 (Legalización de gastos de "
            "viaje), firmado por el colaborador.",
            "- Todos los soportes deben ser factura electrónica a nombre de Nexa Logística "
            "S.A.S., NIT 900.123.456-7. No se aceptan recibos de caja menor ni cuentas de "
            "cobro sin factura.",
        ],
    ),
    (
        "3. Topes diarios de viáticos nacionales",
        [
            "- Alimentación: COP 120.000 por día.",
            "- Transporte local: COP 80.000 por día.",
            "- Hospedaje: máximo COP 350.000 por noche.",
            "- Hospedaje en Bogotá y Medellín: máximo COP 420.000 por noche.",
            "Los valores que superen estos topes son asumidos por el colaborador, salvo "
            "autorización previa y escrita del gerente de área.",
        ],
    ),
]

PAGE_2 = [
    (
        "4. Viajes internacionales",
        [
            "- Para viajes internacionales el tope de alimentación es de USD 90 por día.",
            "- El hospedaje internacional se reserva a través del área Administrativa según "
            "las tarifas del convenio vigente.",
        ],
    ),
    (
        "5. Anticipos",
        [
            "Se pueden solicitar anticipos por un máximo del 70% del presupuesto total del "
            "viaje. El anticipo se legaliza junto con el resto de gastos en el formato FR-021.",
        ],
    ),
    (
        "6. Aprobaciones",
        [
            "- Gastos hasta COP 2.000.000: aprueba el jefe inmediato.",
            "- Gastos por encima de COP 2.000.000: aprueba el gerente de área.",
        ],
    ),
    (
        "7. Pago del reembolso",
        [
            "Una vez aprobada la legalización, el reembolso se paga en la siguiente quincena "
            "mediante transferencia a la cuenta de nómina del colaborador.",
        ],
    ),
    (
        "8. Gastos no reembolsables",
        [
            "- Bebidas alcohólicas.",
            "- Multas de tránsito.",
            "- Consumos de minibar.",
            "- Gastos personales o de acompañantes.",
        ],
    ),
]


def _write_page(pdf: FPDF, sections: list[tuple[str, list[str]]], title: str | None) -> None:
    """Agrega una página con título opcional y sus secciones."""
    pdf.add_page()
    if title:
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in HEADER:
            pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    for heading, paragraphs in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        for paragraph in paragraphs:
            pdf.multi_cell(0, 6, paragraph, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        pdf.ln(3)


def build_pdf(output: Path) -> Path:
    """Construye el manual de reembolso de gastos y lo guarda en `output`."""
    pdf = FPDF(format="A4")
    pdf.set_creation_date(CREATION_DATE)
    pdf.set_title("Manual de Reembolso de Gastos de Viaje")
    pdf.set_author("Nexa Logística S.A.S. (ficticio)")
    pdf.set_auto_page_break(auto=True, margin=15)
    _write_page(pdf, PAGE_1, title="Manual de Reembolso de Gastos de Viaje")
    _write_page(pdf, PAGE_2, title=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))
    return output


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT
    print(f"PDF generado: {build_pdf(target)}")

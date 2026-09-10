from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = colors.HexColor("#12233F")
BLUE = colors.HexColor("#1F4E79")
RED = colors.HexColor("#B42318")
LIGHT = colors.HexColor("#F4F6FA")
LINE = colors.HexColor("#E4E7EC")


def build_pdf(summary: dict[str, object], notes: list[str] | None = None) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="CAPM Analytics Pro Report",
        author="Juan Carlos Muñoz",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], textColor=NAVY, fontSize=22, leading=26, spaceAfter=5)
    sub = ParagraphStyle("ReportSub", parent=styles["Heading2"], textColor=BLUE, fontSize=11, leading=14, spaceAfter=4)
    small = ParagraphStyle("Small", parent=styles["BodyText"], textColor=colors.HexColor("#667085"), fontSize=8.5, leading=11)
    body = ParagraphStyle("Body", parent=styles["BodyText"], textColor=colors.HexColor("#344054"), fontSize=8.8, leading=12)

    story = [
        Paragraph("CAPM Analytics Pro", title),
        Paragraph("Equity risk and required-return research summary", sub),
        Paragraph("Universidad del Cauca · Author: Juan Carlos Muñoz", small),
        Paragraph(datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC"), small),
        Spacer(1, 7),
    ]

    data = [["Metric", "Value"]] + [[str(k), str(v)] for k, v in summary.items()]
    table = Table(data, colWidths=[72 * mm, 103 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [table, Spacer(1, 9), Paragraph("Core equation: E(Ri) = Rf + beta × [E(Rm) − Rf]", body)]

    if notes:
        story += [Spacer(1, 6), Paragraph("Methodology and controls", sub)]
        for note in notes:
            story.append(Paragraph(f"• {note}", body))

    story += [Spacer(1, 10), Paragraph("Research disclaimer: review benchmark selection, currency, country risk, sample period, beta stability and data availability before incorporating the estimate into an investment or valuation decision.", small)]
    doc.build(story)
    return buffer.getvalue()

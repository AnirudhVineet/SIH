"""
One-page PDF briefing note for an officer.

ReportLab rather than WeasyPrint: WeasyPrint needs GTK/Pango system libraries,
which are awkward on Windows and add weight to the deploy image. ReportLab is
pure Python, so the same code path works locally and in the container.

Contents mirror the dashboard, in the order an officer reads it: what the
stress picture is, what the forecast says, how much early warning the spike
model gives, and what the optimizer recommends releasing -- with the input
caveats printed on the page, not just in the UI, so a PDF that leaves the
room still carries them.
"""

from __future__ import annotations

import datetime as dt
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#1c5cab")
INK = colors.HexColor("#14141a")
MUTED = colors.HexColor("#5d5d68")
RULE = colors.HexColor("#d7d7de")
WARN_BG = colors.HexColor("#fdf6e3")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=17, leading=20,
            textColor=INK, alignment=TA_LEFT, spaceAfter=2,
        ),
        "sub": ParagraphStyle(
            "sub", parent=base["Normal"], fontSize=8.5, textColor=MUTED, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=10.5, leading=13,
            textColor=ACCENT, spaceBefore=9, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=8.8, leading=12, textColor=INK,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=7.4, leading=9.5, textColor=MUTED,
        ),
    }


def _table(data: list[list[str]], widths: list[float], align_right: list[int] = ()) -> Table:
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f6f9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def build_brief(
    *,
    commodity: str,
    centre: str,
    horizon: int,
    as_of: str,
    forecast: dict,
    sentence: str,
    stress_rows: list[dict],
    plan,
    release_date: dt.date | None,
    procurement_rows: list[dict],
    spike_metrics: dict,
    accuracy: dict | None,
    input_labels: dict,
) -> bytes:
    """Returns the PDF as bytes, ready for st.download_button."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=13 * mm, bottomMargin=12 * mm,
        title=f"PSS01 brief — {commodity} {centre}",
    )
    S = _styles()
    story = []

    story.append(Paragraph("Price Intelligence &amp; Buffer Stock Brief", S["title"]))
    story.append(
        Paragraph(
            f"{commodity.title()} · {centre} · {horizon}-day outlook &nbsp;|&nbsp; "
            f"data as of {as_of} &nbsp;|&nbsp; generated "
            f"{dt.datetime.now():%d %b %Y %H:%M} &nbsp;|&nbsp; "
            "Price Monitoring Division, Dept. of Consumer Affairs",
            S["sub"],
        )
    )

    # 1 — assessment
    # sentence comes from app/data/sentences.parquet, built for the dashboard's
    # HTML rendering (real rupee glyph). ReportLab's base-14 fonts don't carry
    # U+20B9 -- same issue the LABELS dict already routes around -- so it is
    # swapped for the ASCII form only on this PDF path.
    story.append(Paragraph("1. Assessment", S["h2"]))
    story.append(Paragraph(sentence.replace("₹", "Rs "), S["body"]))

    if forecast:
        story.append(Spacer(1, 5))
        story.append(
            _table(
                [
                    ["Current", f"{horizon}d P50", "80% band", "Change", "Spike prob."],
                    [
                        f"Rs {forecast['current_price']:,.0f}",
                        f"Rs {forecast['p50']:,.0f}",
                        f"Rs {forecast['p10']:,.0f} - {forecast['p90']:,.0f}",
                        f"{forecast['pct_change']:+.1f}%",
                        f"{forecast['spike_proba']:.0%}"
                        if forecast.get("spike_proba") is not None else "-",
                    ],
                ],
                [30 * mm, 30 * mm, 45 * mm, 25 * mm, 28 * mm],
            )
        )
        story.append(Paragraph("Wholesale mandi modal price, Rs per quintal.", S["small"]))

    # 2 — stress
    story.append(Paragraph("2. Price Stress Index", S["h2"]))
    rows = [["Centre", "State", "Stress", "Band"]]
    for r in stress_rows[:6]:
        rows.append([
            r["centre"], r.get("state") or "-",
            f"{r['stress']:.0f}", r.get("band", ""),
        ])
    story.append(_table(rows, [34 * mm, 46 * mm, 20 * mm, 26 * mm], align_right=[2]))
    story.append(
        Paragraph(
            "0-100 composite: 14-day forecast vs the centre's own trailing 1-year "
            "median (45%), spike probability (35%), conformal band width (20%).",
            S["small"],
        )
    )

    # 3 — early warning + accuracy
    story.append(Paragraph("3. Model performance (walk-forward backtest)", S["h2"]))
    perf = [["Metric", "Value"]]
    perf.append(["Median spike early warning",
                 f"{spike_metrics['median_lead_days_per_episode']:.0f} days per episode"])
    perf.append(["Spike episodes caught", f"{spike_metrics['episode_recall']:.0%}"])
    perf.append(["Spike precision / recall",
                 f"{spike_metrics['precision']:.0%} / {spike_metrics['recall']:.0%}"])
    if accuracy:
        perf.append([f"Forecast MAPE ({commodity}, {horizon}d)",
                     f"{accuracy['lgbm_mape']:.1f}% (LightGBM) vs "
                     f"{accuracy['sarimax_mape']:.1f}% (SARIMAX)"])
    story.append(_table(perf, [62 * mm, 88 * mm]))

    # 4 — recommendation
    story.append(Paragraph("4. Recommended release", S["h2"]))
    if plan and plan.rows:
        rel = [["Centre", "State", "Stress", "Release (t)", "% cap", "Transport (Rs)"]]
        for r in plan.rows:
            rel.append([
                r["centre"], r.get("state") or "-", f"{r['stress']:.0f}",
                f"{r['release_tonnes']:,.0f}", f"{r['pct_of_capacity']:.0f}%",
                f"{r['transport_cost_inr']:,.0f}",
            ])
        rel.append([
            "TOTAL", "", "", f"{plan.total_released:,.0f}", "",
            f"{plan.total_cost:,.0f}",
        ])
        t = _table(rel, [28 * mm, 38 * mm, 18 * mm, 26 * mm, 18 * mm, 32 * mm],
                   align_right=[2, 3, 4, 5])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE", (0, -1), (-1, -1), 0.7, ACCENT),
        ]))
        story.append(t)
        release_when = (
            f"Proposed release date: {release_date:%d %b %Y}. " if release_date else ""
        )
        story.append(
            Paragraph(
                f"{release_when}Linear program over centres scoring >= 30 stress, "
                f"available stock {plan.available_stock:,.0f} t, solver status "
                f"{plan.status}.",
                S["small"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No centre currently meets the stress threshold for release. "
                "Recommendation: hold stock.",
                S["body"],
            )
        )

    # 5 — procurement (buy-side)
    story.append(Paragraph("5. Procurement — rebuild the buffer", S["h2"]))
    open_rows = [r for r in procurement_rows if r.get("band") == "Open"]
    if open_rows:
        prows = [["Centre", "State", "Score", "Months to next harvest"]]
        for r in open_rows:
            prows.append([
                r["centre"], r.get("state") or "-",
                f"{r['procurement_score']:.0f}",
                f"{r.get('months_to_next_harvest', '-')}",
            ])
        story.append(_table(prows, [34 * mm, 46 * mm, 20 * mm, 50 * mm], align_right=[2]))
        story.append(
            Paragraph(
                "Harvest-driven price lows worth buying into to rebuild the buffer, "
                "scored from the same trailing-median baseline as the stress index.",
                S["small"],
            )
        )
    else:
        months_out = min(
            (r.get("months_to_next_harvest", 99) for r in procurement_rows), default=None
        )
        story.append(
            Paragraph(
                "No centre is currently in a harvest-driven procurement window for "
                f"{commodity}."
                + (f" Nearest harvest in ~{months_out} month(s)." if months_out is not None else ""),
                S["body"],
            )
        )

    # 6 — caveats, on the page not just on screen
    story.append(Spacer(1, 7))
    caveat = Table(
        [[Paragraph(
            "<b>Input provenance.</b> Stress, forecasts, bands and spike probabilities "
            "are real fitted-model outputs on Agmarknet wholesale mandi data. "
            f"Available stock is an {input_labels['available_stock']}. "
            f"Absorption capacity is a {input_labels['capacity']}. "
            f"Transport cost is a {input_labels['transport']}. "
            "No buffer-stock, capacity or freight feed is ingested by this system, and "
            "mandi arrivals and retail prices are absent from the source data. "
            "<b>This is decision support for a human officer, not a procurement order.</b>",
            S["small"],
        )]],
        colWidths=[152 * mm],
    )
    caveat.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0c85a")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether(caveat))

    doc.build(story)
    return buf.getvalue()

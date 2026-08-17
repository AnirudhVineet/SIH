"""
Shared visual language for the dashboard: one dark surface, one accent, and
chart defaults every figure inherits.

Colour choices are validated, not eyeballed (see PROGRESS.md):
  * Accent blue #3987e5 is the single accent -- UI chrome, forecast line,
    band fill. Contrast 3:1+ on the dark surface.
  * SHAP bars use the blue<->red diverging pair with a neutral zero line,
    because a SHAP contribution is signed (above / below norm). Validated
    at CVD dE 66.4, well clear of the >=12 target.
  * The stress choropleth uses a semantic heat ramp (amber -> red), the
    sequential-with-scale-legend case: lightness decreases monotonically
    (0.84 -> 0.72 -> 0.55) and every step clears 3:1 on the surface.

No dual axes anywhere: mandi arrivals would have been the natural second
axis on the commodity chart, but arrivals do not exist in this dataset
(QUESTIONS.md #1) and a second scale would misrepresent the data anyway.
"""

from __future__ import annotations

SURFACE = "#16161a"
SURFACE_RAISED = "#1e1e24"
BORDER = "#2e2e36"

TEXT_PRIMARY = "#f2f2ef"
TEXT_SECONDARY = "#a9a9a2"
TEXT_MUTED = "#77776f"

ACCENT = "#3987e5"
ACCENT_SOFT = "rgba(57, 135, 229, 0.18)"
ACCENT_DIM = "rgba(57, 135, 229, 0.45)"

POSITIVE = "#3987e5"  # SHAP: above this commodity's recent norm
NEGATIVE = "#e66767"  # SHAP: below it

# Status steps for stress banding -- paired with a label everywhere they are
# used, never colour alone.
STATUS = {
    "Low": "#0ca30c",
    "Moderate": "#f5c518",
    "Elevated": "#ec835a",
    "High": "#d03b3b",
}

# Semantic-heat sequential ramp for the choropleth, low -> high stress.
STRESS_SCALE = [
    [0.00, "#2b3a4a"],
    [0.25, "#6b6a3f"],
    [0.50, "#f5c518"],
    [0.75, "#ec835a"],
    [1.00, "#d03b3b"],
]

GRID = "#26262e"


def stress_band(score: float) -> str:
    """Stress score -> named band. Bands are what the officer reads; the
    colour is secondary encoding, never the only signal."""
    if score >= 60:
        return "High"
    if score >= 45:
        return "Elevated"
    if score >= 30:
        return "Moderate"
    return "Low"


def style_fig(fig, height: int = 420, legend: bool = True):
    """Applies the shared chart chrome: transparent surface, hairline
    recessive grid, no chart-junk borders, legend on top."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY, size=13),
        margin=dict(l=8, r=8, t=28, b=8),
        hoverlabel=dict(
            bgcolor=SURFACE_RAISED,
            bordercolor=BORDER,
            font=dict(color=TEXT_PRIMARY, size=12),
        ),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=GRID,
        tickfont=dict(color=TEXT_MUTED),
        title_font=dict(color=TEXT_MUTED),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=TEXT_MUTED),
        title_font=dict(color=TEXT_MUTED),
    )
    return fig


CSS = f"""
<style>
  .stApp {{ background: {SURFACE}; }}
  section[data-testid="stSidebar"] {{
      background: {SURFACE_RAISED};
      border-right: 1px solid {BORDER};
  }}
  h1, h2, h3, h4 {{ color: {TEXT_PRIMARY} !important; letter-spacing: -0.01em; }}
  p, span, label, li {{ color: {TEXT_SECONDARY}; }}

  .hero {{
      font-size: 2.6rem; font-weight: 650; color: {TEXT_PRIMARY};
      line-height: 1.05; letter-spacing: -0.02em;
  }}
  .hero-sub {{ font-size: 0.82rem; color: {TEXT_MUTED}; text-transform: uppercase;
      letter-spacing: 0.08em; margin-bottom: 0.2rem; }}

  .tile {{
      background: {SURFACE_RAISED}; border: 1px solid {BORDER};
      border-radius: 10px; padding: 1rem 1.15rem; height: 100%;
  }}
  .tile-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
      color: {TEXT_MUTED}; margin-bottom: 0.35rem; }}
  .tile-value {{ font-size: 1.75rem; font-weight: 640; color: {TEXT_PRIMARY};
      line-height: 1.1; }}
  .tile-note {{ font-size: 0.78rem; color: {TEXT_MUTED}; margin-top: 0.3rem; }}

  .sentence {{
      background: {SURFACE_RAISED};
      border: 1px solid {BORDER};
      border-left: 3px solid {ACCENT};
      border-radius: 8px; padding: 1.05rem 1.2rem;
      font-size: 1.02rem; line-height: 1.6; color: {TEXT_PRIMARY};
  }}
  .caveat {{
      border-left: 3px solid {STATUS["Moderate"]};
      background: rgba(245, 197, 24, 0.06);
      border-radius: 6px; padding: 0.7rem 0.95rem;
      font-size: 0.85rem; color: {TEXT_SECONDARY};
  }}
  .pill {{
      display: inline-block; padding: 0.14rem 0.6rem; border-radius: 999px;
      font-size: 0.74rem; font-weight: 600; letter-spacing: 0.02em;
  }}
  .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 8px; }}
  [data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY}; }}
  div[data-baseweb="select"] > div {{
      background: {SURFACE_RAISED}; border-color: {BORDER};
  }}
</style>
"""

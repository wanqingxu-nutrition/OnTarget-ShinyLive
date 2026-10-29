"""Plotly visualizations — nutrient ring chart, HEI-2020 spider, calorie gauge.

Nutrient chart: green ring at r=0.5 marks the adequate zone. Adequate points sit
                on the ring (green). Both deficient (blue) and excessive (red)
                points push outward toward the edge; color encodes direction,
                radial distance encodes severity.
HEI-2020 spider: r = component_score / max_score. Center = 0, edge = max.
                 Five equal radial divisions (0, 20, 40, 60, 80, 100% of max),
                 covering both 5-point and 10-point components on a common axis.
Calorie gauge: needle on a semicircle; green band around the calorie target.
"""

from __future__ import annotations
import plotly.graph_objects as go

CAL_TARGET = 2330
# Dot colors chosen to stay distinguishable for the three main types of color
# blindness (protanopia, deuteranopia, tritanopia), checked by simulation:
#   * red, green and blue differ from one another for every vision type, and
#   * the green -> red fade changes in LIGHTNESS as well as hue (light
#     yellow-green -> dark crimson), so neighbouring shades stay apart even for
#     viewers who cannot tell red from green by hue alone.
DEFICIENT_COLOR = "#0072B2"   # blue
EXCESSIVE_COLOR = "#9E0142"   # red (dark crimson)
OK_COLOR = "#9ACD32"          # green (light yellow-green)
DOT_OUTLINE = dict(color="rgba(0,0,0,0.55)", width=1)   # keeps light dots visible on white
# Behind the NAME of an element that is already on target, as a highlighter pen
# would. Pale enough to leave the black label fully readable.
ON_TARGET_FILL = "#cfeab4"


def fade_color(r: float) -> str:
    """Color on the green -> red fade. r = 0 gives OK_COLOR, r = 1 EXCESSIVE_COLOR.

    Interpolates in linear RGB so the lightness changes smoothly along the fade.
    """
    t = min(1.0, max(0.0, float(r)))

    def to_lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def to_srgb(v):
        s = 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
        return int(round(min(1.0, max(0.0, s)) * 255))

    a = [int(OK_COLOR[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(EXCESSIVE_COLOR[i:i + 2], 16) for i in (1, 3, 5)]
    rgb = [to_srgb(to_lin(x) * (1 - t) + to_lin(y) * t) for x, y in zip(a, b)]
    return "#%02X%02X%02X" % tuple(rgb)
HEI_FILL_COLOR = "#27AE60"    # green polygon for HEI score area
# Shared by the three round (polar) charts so they render at identical size.
# The side margins fit the longest axis label ("Animal source foods").
ROUND_CHART_MARGIN = dict(l=120, r=120, t=75, b=40)
ROUND_CHART_HEIGHT = 520
# Grey rings dividing each round chart into 5 equal distances (center -> edge).
RING_STEPS = [0.2, 0.4, 0.6, 0.8, 1.0]
RING_COLOR = "rgba(0,0,0,0.22)"
RING_INNER = 0.45
RING_OUTER = 0.55


def _fmt(x, unit=""):
    if x is None:
        return "—"
    if abs(x) >= 100:
        return f"{x:.0f}{' ' + unit if unit else ''}"
    if abs(x) >= 10:
        return f"{x:.1f}{' ' + unit if unit else ''}"
    return f"{x:.2f}{' ' + unit if unit else ''}"


def nutrient_anti_spider(adequacy_rows: list[dict]) -> go.Figure:
    """22-axis nutrient chart. Center = adequate (in-band).

    Adequate points sit at r=0 (green, at center). Off-band points push outward
    toward the edge; color encodes direction (blue = deficient, red = excessive).
    Radial distance from center = severity of the deviation.
    """
    thetas = [row["nutrient"] for row in adequacy_rows]
    rs = [row["r"] for row in adequacy_rows]
    colors = []
    for row in adequacy_rows:
        if row["direction"] == "deficient":
            colors.append(DEFICIENT_COLOR)
        elif row["direction"] == "excessive":
            colors.append(EXCESSIVE_COLOR)
        else:
            colors.append(OK_COLOR)

    # Nutrients inside their range. app.py passes these to the page so the
    # chart's own labels can be highlighted (see ON_TARGET_FILL).
    on_target = [row["nutrient"] for row in adequacy_rows if row["direction"] == "ok"]

    hover = []
    for row in adequacy_rows:
        sev = row.get("severity", 0.0)
        txt = (f"<b>{row['nutrient']}</b><br>"
               f"Intake: {_fmt(row['intake'], row['unit'])}<br>"
               f"Lower (RDA or AI): {_fmt(row['lower'], row['unit'])}<br>"
               f"Upper (UL or CDRR): {_fmt(row['upper'], row['unit'])}<br>"
               f"Status: <b>{'Adequate' if row['direction'] == 'ok' else str(row['direction']).capitalize()}</b>")
        hover.append(txt)

    fig = go.Figure()
    closed_thetas = thetas + [thetas[0]]
    n_close = len(closed_thetas)

    # Dots only, one per nutrient — no line connecting them.
    fig.add_trace(go.Scatterpolar(
        r=rs, theta=thetas,
        mode="markers", cliponaxis=False,
        marker=dict(size=12, color=colors,
                     line=DOT_OUTLINE),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="intake", showlegend=False,
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1.0], showticklabels=False, showline=False, ticks="",   # no ruler or tick marks
                showgrid=True, gridcolor=RING_COLOR, gridwidth=1,   # grey rings, 5 equal steps
                tickvals=RING_STEPS,
            ),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        meta=dict(ontarget=on_target),
        showlegend=False,
        # No chart title: the color legend is shown as text above the chart.
        # Same margins + height as the HEI and HDB charts, so all three round
        # charts are drawn at exactly the same size.
        margin=ROUND_CHART_MARGIN, height=ROUND_CHART_HEIGHT,
    )
    return fig


def hei_anti_spider(component_scores: dict[str, float], bounds: dict[str, dict]) -> go.Figure:
    """13-axis HEI-2020 anti-spider: r = 1 − score/max_score. Center=full marks, edge=0.

    Five equal radial divisions (100%, 80%, 60%, 40%, 20%, 0% of max as you move
    out from center). A perfect component collapses to a dot at center; gaps push
    its dot outward. Shows DOTS ONLY — no connecting polygon or filled area.
    """
    order = ["Total Fruits", "Whole Fruits", "Total Vegetables", "Greens & Beans",
             "Whole Grains", "Dairy", "Total Protein", "Seafood & Plant Protein",
             "Fatty Acids", "Refined Grains", "Sodium", "Added Sugars", "Sat Fat"]
    thetas = []
    rs = []
    hover = []
    on_target = []
    for name in order:
        max_sc = bounds[name]["max_score"]
        sc = float(component_scores.get(name, 0.0))
        r = 1.0 - (sc / max_sc) if max_sc else 1.0  # center = full marks
        label = f"{name} ({int(max_sc)}pt)"
        thetas.append(label)
        rs.append(r)
        if r <= 1e-9:                      # full score: this component is on target
            on_target.append(label)
        hover.append(f"<b>{name}</b><br>Score: {sc:.2f} / {int(max_sc)}<br>"
                     f"Direction: {bounds[name]['direction']}<br>"
                     f"% of max: {100*(1-r):.0f}%")

    fig = go.Figure()
    closed_thetas = thetas + [thetas[0]]
    n_close = len(closed_thetas)


    # Dots only — each component's gap to max. Center = full marks (dot at center);
    # a dot pushed outward = that component fell short. NO connecting line / fill.
    # Color fades gradually: green at full score -> red at 0 points.
    dot_colors = [fade_color(r) for r in rs]
    fig.add_trace(go.Scatterpolar(
        r=rs, theta=thetas,
        mode="markers", cliponaxis=False,
        marker=dict(size=12, color=dot_colors,
                     line=DOT_OUTLINE),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="component",
    ))

    total = sum(component_scores.values()) if component_scores else 0
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1.0], showticklabels=False, showline=False, ticks="",   # no ruler or tick marks
                showgrid=True, gridcolor=RING_COLOR, gridwidth=1,   # grey rings, 5 equal steps
                tickvals=RING_STEPS,
            ),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        meta=dict(ontarget=on_target),
        showlegend=False,
        margin=ROUND_CHART_MARGIN, height=ROUND_CHART_HEIGHT,
        # Keeps the total score; the dot-color explanation is shown above the chart.
        title=dict(text=f"Total score: {total:.1f} / 100",
                   font=dict(size=14)),
    )
    return fig


def hdb_bullseye(assessment: dict) -> go.Figure:
    """6-axis Healthy Diet Basket bull's-eye, one axis per food group.

    r = 1 − adequacy, adequacy = min(1, group share of the whole diet / HDB target share).
      * center (r = 0): the group's share meets or exceeds its target
      * edge   (r = 1): no calories from the group
    Proportional in between: half the target share puts the dot halfway. The title
    shows the mean adequacy of the 6 groups as a score out of 100.
    """
    thetas, rs, hover, on_target = [], [], [], []
    for row in assessment["rows"]:
        target = row["target_kcal"]
        pct = row["pct_of_target"]
        r = 1.0 - row["adequacy"]
        thetas.append(row["group"])
        rs.append(r)
        if r <= 1e-9:                      # share meets or beats the target
            on_target.append(row["group"])
        div = "met" if row["meets_diversity"] else "below minimum"
        hover.append(
            f"<b>{row['group']}</b><br>"
            f"Share of diet: {100 * row['current_share']:.1f}% "
            f"(target {100 * row['ideal_share']:.1f}%)<br>"
            f"Calories: {row['current_kcal']:.0f} of {target:.0f} kcal target<br>"
            f"Adequacy: {100 * row['adequacy']:.0f}%<br>"
            f"Food types: {row['n_items']} of {row['min_items']} required ({div})"
        )

    fig = go.Figure()
    closed = thetas + [thetas[0]]
    # Color fades gradually: green at (or above) the group's calorie target ->
    # red at no calories from the group at all.
    dot_colors = [fade_color(r) for r in rs]
    fig.add_trace(go.Scatterpolar(
        r=rs, theta=thetas, mode="markers", cliponaxis=False,
        marker=dict(size=14, color=dot_colors, line=DOT_OUTLINE),
        text=hover, hovertemplate="%{text}<extra></extra>", name="group",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1.0], showticklabels=False, showline=False, ticks="",   # no ruler or tick marks
                showgrid=True, gridcolor=RING_COLOR, gridwidth=1,   # grey rings, 5 equal steps
                tickvals=RING_STEPS,
            ),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        meta=dict(ontarget=on_target),
        showlegend=False,
        # Wide side margins: "Animal source foods" sits on the left axis and was
        # clipped at the default margin.
        margin=ROUND_CHART_MARGIN, height=ROUND_CHART_HEIGHT,
        title=dict(text=f"Mean adequacy ratio: "
                        f"{100 * assessment['mean_adequacy']:.1f} / 100%",
                   font=dict(size=14)),
    )
    return fig


def cost_quality_frontier(series: dict[str, dict]) -> go.Figure:
    """Line chart: HEI score (x) vs cost in $/day (y), one trace per series.

    `series` is dict[label] = {target_str: {score, cost_usd, energy_kcal, n_foods, ...}}
    """
    style = {
        "Lowest-cost diet": {"color": "#2563EB", "dash": "solid", "marker": "circle"},
        "HEI only":         {"color": "#F59E0B", "dash": "dash",  "marker": "diamond"},  # amber dashed
        "HEI + DRI bounds": {"color": "#10B981", "dash": "solid", "marker": "circle"},   # emerald solid
        "DRI + HEI":        {"color": "#10B981", "dash": "solid", "marker": "circle"},   # emerald solid
        "DRI only":         {"color": "#7C3AED", "dash": "solid", "marker": "star"},     # violet star (single point)
    }
    fig = go.Figure()
    for label, data in series.items():
        keys = sorted(data.keys(), key=int)
        achieved = [data[k]["score"] for k in keys]
        costs = [data[k]["cost_usd"] for k in keys]
        kcal = [data[k]["energy_kcal"] for k in keys]
        nfoods = [data[k]["n_foods"] for k in keys]
        hover = [
            f"<b>{label}</b><br>HEI target {k} → achieved {a:.1f}<br>"
            f"Cost: ${c:.2f}/day<br>Calories: {e:.0f} kcal<br>Foods: {n}"
            for k, a, c, e, n in zip(keys, achieved, costs, kcal, nfoods)
        ]
        s = style.get(label, {"color": "#1F2937", "dash": "solid", "marker": "circle"})
        fig.add_trace(go.Scatter(
            x=achieved, y=costs, mode="lines+markers",
            line=dict(color=s["color"], width=3, dash=s["dash"]),
            marker=dict(size=12, color=s["color"], symbol=s["marker"],
                          line=dict(color="white", width=2)),
            name=label,
            text=hover, hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="HEI-2020 score (achieved)", range=[40, 105], gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="Cost ($/day)", rangemode="tozero", gridcolor="rgba(0,0,0,0.08)"),
        title=dict(text="Cost–quality frontier — least-cost diet at each HEI target",
                   font=dict(size=14)),
        height=360,
        margin=dict(l=60, r=30, t=60, b=50),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


CATEGORY_PALETTE = [
    "#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16",
    "#06B6D4", "#A855F7", "#F43F5E", "#22C55E", "#EAB308",
]


def diet_composition_bar(diet: dict[str, float], catalog) -> go.Figure:
    """Horizontal bar chart of foods in a diet, sorted by grams, colored by category."""
    name_to_cat = dict(zip(catalog["name"], catalog["food_category"]))
    items = sorted(diet.items(), key=lambda kv: kv[1])  # ascending so largest is on top in horizontal
    names = [n for n, _ in items]
    grams = [g for _, g in items]
    cats = [name_to_cat.get(n, "Other") for n in names]

    unique_cats = sorted(set(cats))
    cat_color = {c: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)] for i, c in enumerate(unique_cats)}
    bar_colors = [cat_color[c] for c in cats]

    short_names = [n if len(n) <= 50 else n[:47] + "…" for n in names]
    hover = [
        f"<b>{n}</b><br>Category: {c}<br>Grams/day: {g:.0f}"
        for n, c, g in zip(names, cats, grams)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grams, y=short_names, orientation="h",
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0.1)", width=0.5)),
        text=[f"{g:.0f}g" for g in grams],
        textposition="outside",
        textfont=dict(size=10),
        hovertext=hover, hoverinfo="text",
        showlegend=False,
    ))
    # Add invisible category legend traces (one per unique category)
    for c in unique_cats:
        fig.add_trace(go.Bar(
            x=[None], y=[None], orientation="h",
            marker=dict(color=cat_color[c]), name=c, showlegend=True,
            hoverinfo="skip",
        ))
    fig.update_layout(
        title=dict(text="Diet composition (grams/day, colored by food category)",
                   font=dict(size=14)),
        xaxis=dict(title="Grams / day", gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(automargin=True),
        height=max(320, 28 * len(names) + 80),
        margin=dict(l=20, r=80, t=60, b=50),
        plot_bgcolor="white",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                    font=dict(size=10)),
        bargap=0.25,
    )
    return fig


def hei_component_bar(component_scores: dict[str, float], bounds: dict[str, dict]) -> go.Figure:
    """Horizontal bar chart: HEI components, score / max, colored by direction."""
    order = ["Total Fruits", "Whole Fruits", "Total Vegetables", "Greens & Beans",
             "Whole Grains", "Dairy", "Total Protein", "Seafood & Plant Protein",
             "Fatty Acids", "Refined Grains", "Sodium", "Added Sugars", "Sat Fat"]
    labels, scores, maxes, fills, fracs = [], [], [], [], []
    on_target = []
    for name in order:
        max_sc = bounds[name]["max_score"]
        sc = float(component_scores.get(name, 0.0))
        labels.append(f"{name} ({int(max_sc)} pt)")
        if max_sc and sc >= max_sc - 1e-9:
            on_target.append(labels[-1])
        scores.append(sc)
        maxes.append(max_sc)
        fracs.append(sc / max_sc if max_sc else 0)
        direction = bounds[name]["direction"]
        if direction.startswith("moderation"):
            fills.append("#E67E22")  # orange
        else:
            fills.append("#27AE60")  # green
    # reverse so first item appears at top of chart
    labels = labels[::-1]
    scores = scores[::-1]
    maxes = maxes[::-1]
    fracs = fracs[::-1]
    fills = fills[::-1]

    fig = go.Figure()
    # Background "max" bars (faded)
    fig.add_trace(go.Bar(
        x=maxes, y=labels, orientation="h",
        marker=dict(color="rgba(0,0,0,0.06)", line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))
    # Actual score bars on top
    fig.add_trace(go.Bar(
        x=scores, y=labels, orientation="h",
        marker=dict(color=fills, line=dict(color="rgba(0,0,0,0.15)", width=0.5)),
        text=[f"{s:.1f}/{int(m)} ({100*f:.0f}%)" for s, m, f in zip(scores, maxes, fracs)],
        textposition="outside",
        textfont=dict(size=10),
        showlegend=False, hoverinfo="text",
    ))
    total = sum(scores)
    fig.update_layout(
        barmode="overlay",
        title=dict(text=f"HEI-2020 component scores — total {total:.1f} / 100",
                   font=dict(size=14)),
        xaxis=dict(title="Score", range=[0, 12], gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(automargin=True),
        meta=dict(ontarget=on_target),
        height=440,
        margin=dict(l=20, r=80, t=60, b=50),
        plot_bgcolor="white",
    )
    return fig


HDB_PALETTE = {
    "Starchy staples":         "#E69F00",  # orange
    "Vegetables":              "#2CA02C",  # green
    "Fruits":                  "#D62728",  # red
    "Animal source foods":     "#9467BD",  # purple
    "Legumes, nuts and seeds": "#8C564B",  # brown
    "Oils and fats":           "#1F77B4",  # blue
}


def hdb_pie(assessment: dict) -> go.Figure:
    """Side-by-side pies: ideal HDB energy share vs. current diet energy share."""
    from plotly.subplots import make_subplots

    rows = assessment["rows"]
    groups = [r["group"] for r in rows]
    colors = [HDB_PALETTE[g] for g in groups]
    ideal = [r["ideal_share"] * 100 for r in rows]
    current = [r["current_share"] * 100 for r in rows]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("Ideal (HDB target)", "Current diet"),
    )
    fig.add_trace(go.Pie(
        labels=groups, values=ideal, marker=dict(colors=colors), sort=False,
        name="Ideal", texttemplate="%{value:.1f}%",
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    ), 1, 1)
    fig.add_trace(go.Pie(
        labels=groups, values=current, marker=dict(colors=colors), sort=False,
        name="Current", texttemplate="%{value:.1f}%",
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    ), 1, 2)
    fig.update_layout(
        title=dict(text="Healthy Diet Basket — energy share by food group",
                   font=dict(size=14)),
        height=430, margin=dict(l=20, r=20, t=90, b=20),
        legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
    )
    return fig


def hdb_histogram(assessment: dict) -> go.Figure:
    """Grouped bars: ideal vs current energy share per HDB group, with
    a ⚠ marker on any group that fails the minimum item-diversity rule."""
    rows = assessment["rows"]
    groups = [r["group"] for r in rows]
    ideal = [r["ideal_share"] * 100 for r in rows]
    current = [r["current_share"] * 100 for r in rows]
    flags = ["" if r["meets_diversity"] else f" ⚠ {r['n_items']}/{r['min_items']}"
             for r in rows]
    ideal_text = [f"{v:.1f}%" for v in ideal]
    current_text = [f"{c:.1f}%{fl}" for c, fl in zip(current, flags)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=groups, y=ideal, name="Ideal (HDB)",
        marker=dict(color="rgba(0,0,0,0.25)"),
        text=ideal_text, textposition="outside",
        hovertemplate="%{x}<br>Ideal: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=groups, y=current, name="Current diet",
        marker=dict(color=[HDB_PALETTE[g] for g in groups]),
        text=current_text, textposition="outside",
        hovertemplate="%{x}<br>Current: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="HDB energy share: ideal vs current "
                        "<br><sub>⚠ n/min = group below minimum item diversity</sub>",
                   font=dict(size=14)),
        yaxis=dict(title="% of energy", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(tickangle=-20),
        height=430, margin=dict(l=50, r=20, t=70, b=110),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def calorie_gauge(total_kcal: float, target: float = CAL_TARGET) -> go.Figure:
    """Semicircle gauge. Green band = target ±10%, yellow = ±20%, red outside."""
    low_green, high_green = target * 0.9, target * 1.1
    low_yellow, high_yellow = target * 0.8, target * 1.2
    max_scale = max(target * 1.6, total_kcal * 1.1, target * 1.6)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total_kcal,
        number={"suffix": " kcal", "font": {"size": 36}},
        delta={"reference": target, "increasing": {"color": "#C0392B"},
                "decreasing": {"color": "#2E86C1"}, "suffix": " kcal"},
        gauge={
            "axis": {"range": [0, max_scale], "tickwidth": 1},
            "bar": {"color": "rgba(30, 30, 30, 0.85)", "thickness": 0.25},
            "steps": [
                {"range": [0, low_yellow], "color": "#F5B7B1"},
                {"range": [low_yellow, low_green], "color": "#FAD7A0"},
                {"range": [low_green, high_green], "color": "#ABEBC6"},
                {"range": [high_green, high_yellow], "color": "#FAD7A0"},
                {"range": [high_yellow, max_scale], "color": "#F5B7B1"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.9,
                "value": target,
            },
        },
        title={"text": f"Daily calories  •  target {target:.0f} kcal", "font": {"size": 14}},
    ))
    fig.update_layout(height=320, margin=dict(l=30, r=30, t=60, b=30))
    return fig

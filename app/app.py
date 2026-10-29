"""OnTarget - Diet Optimizer, as a Shiny for Python app that runs in the browser.

This is a UI port of the Streamlit app at github.com/AlkaneX/OnTarget-DietOptimizer
(commit b5d43a7, 14 September 2026). `data.py`, `scoring.py`, `optimize.py` and
`viz.py` are byte-for-byte copies of that repository: no guideline rule, scoring
formula or solver constraint was rewritten here. Only the page framework changed,
so every number this app shows comes from the same code as the Streamlit version.

Why a port at all: ShinyLive compiles the whole app to WebAssembly (Pyodide), so
the finished site is static files. There is no Python server, nothing to deploy to
Streamlit Cloud, and a reader can open it from a folder or any static host.

Three Streamlit mechanisms have no equivalent in the browser and are rebuilt here:
  * st.session_state   -> reactive.value (diet, solver amounts, optimizer banner)
  * st.cache_data      -> a plain dict memo on the solve, keyed the same way
  * st.context.headers -> the page reports its own width, because a browser-only
                          app cannot read the request's User-Agent to spot a phone
"""

from __future__ import annotations

import html as _html
import json
import re
import textwrap

import pandas as pd
from shiny import App, reactive, render, ui

import data
import scoring
import viz
import optimize


# plotly.js is loaded once in the page head; every figure is then embedded as
# plain HTML. This avoids shinywidgets/ipywidgets, which would pull a large extra
# stack into Pyodide for no gain here.
PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

# Tolerance for "at the bound" (binding) detection - suppresses floating-point dust.
BIND_TOL = 1e-6

# Chart toolbar: no zoom tools (the magnifier icon) on any chart. Download and
# full screen stay; in full screen the round charts resize to fit the device.
CHART_CONFIG = {
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d",
                               "zoomIn2d", "zoomOut2d", "autoScale2d",
                               "resetScale2d"],
    "displaylogo": False,
    "responsive": True,
}

# Menu labels are "full name (abbreviation)"; the values are the short internal
# names optimize.py matches on. Referenced via these constants (never re-typed as
# literals) so a label edit cannot silently desync from the lookups below.
LBL_DRI = "Dietary Reference Intakes (DRI)"
LBL_HEI = "Healthy Eating Index (HEI)"
LBL_HDB = "Healthy Diet Basket (HDB)"
GUIDELINE_CHOICES = {
    LBL_DRI: "DRI",
    LBL_HEI: "HEI 2020",
    LBL_HDB: optimize.HDB_WITH_DIVERSITY,
}
NONE_LABEL = "—"                       # an unused menu
OPT_OPTS = [NONE_LABEL] + list(GUIDELINE_CHOICES)
RANK_LABELS = ["Choose one criterion",
               "Choose another criterion (optional)",
               "Choose a third criterion (optional)"]

# Charts shown by default (and after each calculation): Comparison across Criteria
# and Food Breakdown.
CHART_KEYS = [
    ("chk_cmp", "Comparison across Criteria", True),
    ("chk_cal", "Calories", False),
    ("chk_hei", "HEI", False),
    ("chk_dri", "DRI", False),
    ("chk_hdb", "HDB", False),
    ("chk_comp", "Food Breakdown", True),
]

PHONE_MAX_PX = 640          # same breakpoint the Streamlit app's phone CSS uses


# --------------------------------------------------------------- data load ----
# Food data and the DRI / HEI guideline definitions all come from the data/
# folder via data.py. There is no upload option, exactly as in the live app.
CATALOG = data.load_catalog()
DRI = data.load_dri("female")
HEI_BOUNDS = data.load_hei_bounds()
PRICES = data.load_prices()
CAL_TARGET = float(DRI["Energy"][0])

_SOLVE_CACHE: dict = {}


def solve_criteria(ordered: tuple, cal_target_kcal: float):
    """Cheapest diet meeting a SET of criteria, memoized like st.cache_data.

    `ordered` is already normalised by optimize.canonical_criteria, so every
    ordering of the same criteria shares ONE entry and returns the identical diet.
    """
    key = (ordered, round(cal_target_kcal, 6))
    if key not in _SOLVE_CACHE:
        _SOLVE_CACHE[key] = optimize.optimize_criteria(
            list(ordered), catalog=CATALOG, prices_df=PRICES,
            hei_bounds=HEI_BOUNDS, dri=DRI, calorie_target=cal_target_kcal,
        )
    return _SOLVE_CACHE[key]


# ----------------------------------------------------------------- helpers ----
def slug(name: str) -> str:
    """Stable, HTML-id-safe token for a food name (input ids are per food)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_") or "food"


def wrap_axis_labels(fig, width: int = 13):
    """Put long axis labels onto two or three short lines."""
    for tr in fig.data:
        if getattr(tr, "theta", None) is not None:
            tr.theta = ["<br>".join(textwrap.wrap(str(t), width=width)) or str(t)
                        for t in tr.theta]
    return fig


def phone_round(fig):
    """On a phone, every round chart (full-size sections and the comparison view)
    gets exactly the same layout, so all of them are drawn at the same size and as
    wide as the screen allows: wrapped labels, small side margins, the same top
    margin whether or not the chart has a title, and the same height."""
    wrap_axis_labels(fig)
    has_title = bool(fig.layout.title.text)
    fig.update_layout(height=450, margin=dict(l=62, r=62, t=80, b=30),
                      font=dict(size=9))
    if has_title:
        fig.update_layout(title_font_size=12, title_x=0.5, title_xanchor="center")
    fig.update_polars(angularaxis_tickfont_size=8)
    return fig


def round_fig(fig, phone: bool):
    """Round chart shown full-size in its own section."""
    return phone_round(fig) if phone else fig


def compact_fig(fig, phone: bool):
    """Shrink a round chart to fit one of three side-by-side columns. All three get
    the same margins, height and label size, so they are drawn equally big. The
    chart's own title is dropped (its score is shown above it instead). On a phone
    the columns stack, so it uses the shared full-width phone layout."""
    if phone:
        fig.layout.title = {}
        return phone_round(fig)
    wrap_axis_labels(fig)
    # Reset the title completely. Leaving a styled title with no text makes
    # Plotly print the word "undefined" above the chart.
    fig.layout.title = {}
    fig.update_layout(height=320, margin=dict(l=55, r=55, t=30, b=30),
                      font=dict(size=9))
    fig.update_polars(angularaxis_tickfont_size=8)
    return fig


def with_title(fig, text: str):
    """Give a chart a centred title; line breaks in `text` become new lines."""
    fig.update_layout(title=dict(text=text.replace("\n", "<br>"), font=dict(size=14)))
    return fig


def black_text(fig):
    """Draw every piece of chart text (titles, axis and hover labels) in black."""
    fig.update_layout(font=dict(color="#000000"))
    # Color the title only if the chart HAS one. Styling an empty title makes
    # Plotly print the word "undefined" where the title would be.
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(color="#000000"))
    fig.update_polars(angularaxis_tickfont_color="#000000")
    return fig


def fig_html(fig) -> ui.HTML:
    """Plotly figure -> embeddable HTML (plotly.js comes from the page head).

    Each chart sits in its own scroll box: a round chart whose outermost axis
    labels reach past its drawing area would otherwise widen the whole page and
    leave the body scrolling sideways.
    """
    height = fig.layout.height or 420
    inner = fig.to_html(full_html=False, include_plotlyjs=False,
                        default_height=f"{height}px", config=CHART_CONFIG)
    # The names of the elements already on target, for CHART_FIT_JS to highlight.
    # Plotly has no per-tick label style, so the page draws the highlight itself.
    meta = fig.layout.meta
    labels = list(meta.get("ontarget") or []) if isinstance(meta, dict) else []
    attr = (f' data-ontarget="{_html.escape(json.dumps(labels))}"') if labels else ""
    return ui.HTML(f'<div class="chart-wrap"{attr}>{inner}</div>')


def table_html(rows, sizer_rows=None) -> ui.HTML:
    """Table whose columns are just wide enough for their contents, every cell
    aligned left. Wide tables scroll sideways instead of squeezing their text.

    `sizer_rows` (optional) are extra rows rendered INVISIBLY (visibility:
    collapse): they take no space but still size the columns. Passing the same
    sizer rows to several tables gives them identical column widths.
    """
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df = df.astype(str)
    esc = _html.escape
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>"
                   for row in df.values.tolist())
    for srow in (sizer_rows or []):
        body += ('<tr class="sizer">'
                 + "".join(f"<td>{esc(str(srow.get(c, '')))}</td>" for c in df.columns)
                 + "</tr>")
    return ui.HTML(f'<div class="fit-table"><table><thead><tr>{head}</tr></thead>'
                   f"<tbody>{body}</tbody></table></div>")


def flag(label: str, good: str) -> str:
    """Prefix a status label: target emoji when it equals `good`, warning otherwise."""
    return f"\U0001F3AF {label}" if label == good else f"⚠️ {label}"


def dri_status(direction: str) -> str:
    """DRI status label with a capital first letter: Adequate / Deficient / Excessive."""
    return "Adequate" if direction == "ok" else str(direction).capitalize()


def hei_direction(score: float, max_score: float, component_type: str) -> str:
    """What would raise this HEI-2020 component's score, from its scoring type."""
    if max_score and score >= max_score - 1e-6:
        return "On target"
    if component_type == "adequacy_ratio":
        return "Less saturated fat or more unsaturated fat"
    if component_type.startswith("moderation"):
        return "Reduce intake"
    return "Need more"


def plain_number(x, digits: int = 2) -> str:
    """Round to `digits` places; whole numbers lose the trailing '.0' (2330.0 -> 2330)."""
    v = round(float(x), digits)
    return str(int(v)) if v == int(v) else f"{v:.{digits}f}".rstrip("0").rstrip(".")


def dot_legend(items) -> ui.HTML:
    """One legend line above a chart: a colored dot before each description."""
    parts = [f'<span style="color:{c}; font-size:1.2rem; '
             f'-webkit-text-stroke:0.6px rgba(0,0,0,0.55);">&#9679;</span> {_html.escape(t)}'
             for c, t in items]
    return ui.HTML('<div class="dot-legend">' + "; ".join(parts) + "</div>")


def ontarget_note() -> ui.HTML:
    """Explains the light-green highlight drawn behind on-target chart labels."""
    return ui.HTML('<div class="dot-legend">Element names highlighted like '
                   '<span class="hl-sample">this</span> are already on target.</div>')


def metric(label: str, value: str, delta: str = "") -> ui.Tag:
    """One headline indicator: large plain value, no box, not bold (as in the app)."""
    return ui.tags.div(
        ui.tags.div(label, class_="metric-label"),
        ui.tags.div(value, class_="metric-value"),
        ui.tags.div(delta, class_="metric-delta"),
        class_="metric",
    )


def note(kind: str, *children) -> ui.Tag:
    """Streamlit's st.success / st.warning / st.info / st.error boxes."""
    return ui.tags.div(*children, class_=f"note note-{kind}")


def section_title(text: str) -> ui.Tag:
    return ui.tags.h3(text, class_="section-title")


def expander(title: str, *children, open_: bool = False) -> ui.Tag:
    """Streamlit's st.expander, as a native <details> block."""
    return ui.tags.details(
        ui.tags.summary(title), ui.tags.div(*children, class_="exp-body"),
        open="open" if open_ else None, class_="exp",
    )


# A browser-only app has no request headers, so the PAGE reports its own width and
# the server lays the round charts out for a phone below 640 px - the same
# breakpoint, and the same chart layout, the Streamlit app picks from the
# User-Agent string.
VIEWPORT_JS = """
(function () {
  function report() {
    if (window.Shiny && Shiny.setInputValue) {
      Shiny.setInputValue('viewport_w', window.innerWidth, {priority: 'event'});
    }
  }
  document.addEventListener('shiny:connected', report);
  // Belt and braces: if that event already fired (or never does), keep trying
  // briefly until Shiny is ready, so the server always learns the width.
  var tries = 0;
  var poll = setInterval(function () {
    if (window.Shiny && Shiny.setInputValue) { report(); clearInterval(poll); }
    else if (++tries > 100) { clearInterval(poll); }
  }, 100);
  var t = null;
  window.addEventListener('resize', function () {
    clearTimeout(t);
    t = setTimeout(report, 250);
  });
})();
"""

# Plotly measures its container once, when the chart is first drawn. Shiny
# inserts these charts into a layout that is still settling, so a chart could be
# drawn wider than the column it ends up in: the three round charts of
# "Comparison across Criteria" were cut off, each with its own sideways
# scrollbar. This re-fits every chart to its box after the layout settles, and
# again whenever that box changes size (window resized, sidebar opened or
# closed, an expander opened).
CHART_FIT_JS = r"""
(function () {
  var timer = null;
  function fitAll() {            // debounced: a collapsing sidebar animates,
    clearTimeout(timer);         // and only the width it settles on matters
    timer = setTimeout(doFit, 150);
  }
  function fitRepeatedly() {
    // A chart is drawn by its own inline script, so it may not exist yet when
    // Shiny says the section is in the page. Re-fit a few times while the
    // charts appear; doFit is a no-op for anything already the right size.
    [0, 150, 400, 900, 1800, 3000].forEach(function (ms) {
      setTimeout(doFit, ms);
    });
  }
  // Plotly styles every tick label of an axis the same way, so an individual
  // label cannot be given its own colour through the figure. The page draws the
  // highlight instead: a rounded rectangle behind the label's own glyphs, in the
  // tick's coordinate system, so it follows the label wherever Plotly puts it.
  var SVGNS = 'http://www.w3.org/2000/svg';
  var ON_TARGET_FILL = '__ON_TARGET_FILL__';
  function squash(s) { return String(s).replace(/\s+/g, ''); }
  function highlight(wrap) {
    var raw = wrap.getAttribute('data-ontarget');
    if (!raw) return;
    var names;
    try { names = JSON.parse(raw); } catch (e) { return; }
    var wanted = {};
    (names || []).forEach(function (n) { wanted[squash(n)] = true; });
    // Angular labels on the round charts, category labels on the bar charts.
    var labels = wrap.querySelectorAll(
      '.angular-axis text, .yaxislayer-above text, .xaxislayer-above text');
    Array.prototype.forEach.call(labels, function (t) {
      var tick = t.parentNode;
      if (!tick) return;
      var mark = tick.querySelector('rect.on-target-mark');
      // A wrapped label is several tspans; textContent joins them with no space,
      // which is why both sides are compared with the whitespace taken out.
      if (!wanted[squash(t.textContent)]) {
        // Plotly reuses tick elements, so a label that is no longer on target
        // can arrive wearing the previous one's highlight.
        if (mark) tick.removeChild(mark);
        return;
      }
      var box;
      try { box = t.getBBox(); } catch (e) { return; }
      if (!box || !box.width) return;
      if (!mark) {
        mark = document.createElementNS(SVGNS, 'rect');
        mark.setAttribute('class', 'on-target-mark');
        mark.setAttribute('rx', 3);
        mark.setAttribute('fill', ON_TARGET_FILL);
        mark.setAttribute('pointer-events', 'none');   // never swallow a hover
        tick.insertBefore(mark, t);                    // behind the text, not over it
      }
      // getBBox is in the text's OWN coordinates, and Plotly puts the placement
      // transform on the <text> element rather than on the tick group. Without
      // copying it the highlight lands at the group's origin, not on the label.
      var tf = t.getAttribute('transform');
      if (tf) { mark.setAttribute('transform', tf); }
      else { mark.removeAttribute('transform'); }
      mark.setAttribute('x', box.x - 3);
      mark.setAttribute('y', box.y - 1);
      mark.setAttribute('width', box.width + 6);
      mark.setAttribute('height', box.height + 2);
    });
  }

  function doFit() {
    if (!window.Plotly) return;
    var wraps = document.querySelectorAll('.chart-wrap');
    Array.prototype.forEach.call(wraps, function (w) {
      var d = w.querySelector('.plotly-graph-div');
      if (!d || d.offsetParent === null) return;   // inside a closed expander
      if (!d._fullLayout) return;                  // not drawn yet
      var target = w.clientWidth;
      if (!target) return;
      var now = d._fullLayout ? d._fullLayout.width : 0;
      if (Math.abs(now - target) < 1) return;
      // An explicit width, rather than Plotly's own re-measure, so the chart
      // can only ever be exactly as wide as the box it sits in.
      try { Plotly.relayout(d, {width: target}); } catch (e) { }
    });
    var marked = document.querySelectorAll('.chart-wrap[data-ontarget]');
    Array.prototype.forEach.call(marked, highlight);
  }
  var ro = window.ResizeObserver ? new ResizeObserver(fitAll) : null;
  function scan() {
    if (ro) {
      var wraps = document.querySelectorAll('.chart-wrap');
      // observe() ignores an element it is already watching.
      Array.prototype.forEach.call(wraps, function (w) { ro.observe(w); });
    }
    fitAll();
  }
  // Shiny inserts a whole section at once and each chart then draws itself, so
  // watch the document for new charts rather than trusting one event to be the
  // moment everything is ready. doFit does nothing to a chart already the right
  // size, so this settles after the last chart is drawn.
  if (window.MutationObserver) {
    new MutationObserver(scan).observe(document.documentElement,
                                       {childList: true, subtree: true});
  }
  document.addEventListener('shiny:value', fitRepeatedly);
  // <details> does not bubble its toggle event, so listen in the capture phase.
  document.addEventListener('toggle', fitAll, true);
  document.addEventListener('transitionend', fitAll, true);
  window.addEventListener('load', scan);
  window.addEventListener('resize', fitAll);
  // Last resort. A ResizeObserver only delivers while the page is rendering, so
  // a tab that was resized while in the background can come back with a chart
  // still at its old width. doFit costs a handful of property reads when
  // everything already fits, so checking occasionally is cheaper than the bug.
  setInterval(doFit, 2000);
})();
""".replace("__ON_TARGET_FILL__", viz.ON_TARGET_FILL)

CSS = """
/* Black body text everywhere, matching the Streamlit theme's textColor. */
body, .bslib-page-sidebar { color: #000000; }
.sidebar-intro { font-size: 1.05rem; line-height: 1.5; }
.sidebar-intro ul { margin: 0.35rem 0 0.2rem 0; padding-left: 1.2rem; }
.byline { color: #000000; font-size: 0.9rem; line-height: 1.5; }
.byline a { text-decoration: underline; color: #000000; }
.rule { margin: 1.4rem 0 1.2rem 0; border: none; border-top: 1px solid rgba(0,0,0,0.25); }
h1.app-title { font-size: 2.1rem; font-weight: 700; margin: 0 0 0.6rem 0; }
.section-title { font-size: 1.5rem; font-weight: 600; margin: 0.2rem 0 0.6rem 0; }
.dot-legend { color: #000000; margin-bottom: 0.3rem; }
/* The swatch in that legend line, matching viz.ON_TARGET_FILL. */
.hl-sample { background: __ON_TARGET_FILL__; border-radius: 3px; padding: 0 4px; }

/* Headline indicators: large, plain, no box (the app un-boxed these). */
.metric { padding: 0.2rem 0; }
.metric-label { font-size: 1.15rem; font-weight: 400; }
.metric-value { font-size: 2.3rem; font-weight: 400; line-height: 1.2; }
.metric-delta { font-size: 0.9rem; font-weight: 400; color: #333; }

/* Chart picker: the six criterion checkboxes. */
.chart-picker .form-check-label { font-size: 1.2rem; font-weight: 600; }
.chart-picker .form-check { padding: 4px 0 4px 1.6em; }

/* Tables size to their contents and scroll sideways rather than squeezing. */
.fit-table { overflow-x: auto; margin: 0.25rem 0 0.75rem 0; }
.fit-table table { border-collapse: collapse; width: auto; color: #000; }
.fit-table th, .fit-table td { text-align: left !important; white-space: nowrap;
  padding: 4px 14px 4px 6px; border-bottom: 1px solid rgba(0,0,0,0.12); }
.fit-table th { font-weight: 600; background: rgba(0,0,0,0.04); }
.fit-table tr.sizer { visibility: collapse; }

/* Message boxes (st.success / warning / info / error). */
.note { padding: 0.65rem 0.9rem; border-radius: 0.4rem; margin: 0.5rem 0 0.9rem 0;
        border-left: 4px solid; }
.note-success { background: #e8f5e9; border-color: #2e7d32; }
.note-warning { background: #fff8e1; border-color: #f9a825; }
.note-info    { background: #e3f2fd; border-color: #1565c0; }
.note-error   { background: #fdecea; border-color: #c62828; }
.caption { color: #000; font-size: 0.86rem; margin: 0.2rem 0 0.6rem 0; }

/* Expanders. */
.exp { border: 1px solid rgba(0,0,0,0.15); border-radius: 0.4rem;
       margin: 0.5rem 0 0.9rem 0; }
.exp > summary { cursor: pointer; padding: 0.55rem 0.8rem; font-weight: 600; }
.exp .exp-body { padding: 0 0.8rem 0.7rem 0.8rem; }

/* One row of the food cart: the gram box and its remove button. */
.gram-row { align-items: end; margin-bottom: 0.1rem; }
.gram-row .form-group, .gram-row .shiny-input-container { margin-bottom: 0.25rem; }
.cmp-head { min-height: 4.6em; color: #000; line-height: 1.5; }
/* A chart is resized to its own box (see CHART_FIT_JS), so it never needs to
   scroll; hidden also keeps a stale scrollbar from reserving a strip of grey
   under each of the three comparison charts. */
.chart-wrap { overflow-x: hidden; max-width: 100%; }
body { overflow-x: hidden; }

/* Phone: the comparison columns stack and each chart gets the full width. */
@media (max-width: 640px) {
  h1.app-title { font-size: 1.6rem; }
  .metric-value { font-size: 1.9rem; }
  .cmp-head { min-height: 0; margin-top: 0.6rem; }
}
""".replace("__ON_TARGET_FILL__", viz.ON_TARGET_FILL)


# ---------------------------------------------------------------------- UI ----
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Build a daily diet"),
        ui.h5("\U0001F3AF Manual selection or computerized choice of the lowest "
              "cost diet meeting health criteria"),
        ui.HTML(
            '<div class="sidebar-intro">'
            "This diet calculator offers a selection of 93 items sold by Stop &amp; "
            "Shop in December 2025. To build your diet, select one or more health "
            "criteria to display, and search for items to add and select the "
            "quantity desired. You can also use the optimizer to find the "
            "<b>lowest cost</b> set of foods that satisfy those criteria:"
            "<ul>"
            "<li>Essential nutrients based on <b>DRI</b></li>"
            "<li>Disease risks based on <b>HEI</b></li>"
            "<li>Dietary diversity based on <b>HDB</b></li>"
            "</ul></div>"
        ),
        # One menu per criterion. Which menu holds which criterion does NOT matter:
        # the picks are treated as an unordered set (optimize.canonical_criteria).
        *[
            ui.input_select(f"opt_rank_{i}", lbl, choices=OPT_OPTS,
                            selected=(LBL_HEI if i == 0 else NONE_LABEL))
            for i, lbl in enumerate(RANK_LABELS)
        ],
        # The HEI target only matters when the Healthy Eating Index is one of the
        # picks; the panel hides it otherwise without resetting the chosen value.
        ui.panel_conditional(
            " || ".join(f"input.opt_rank_{i} == '{LBL_HEI}'"
                        for i in range(len(RANK_LABELS))),
            ui.input_slider("opt_hei_target", "Healthy Eating Index (HEI) target score",
                            min=50, max=100, value=100, step=5),
        ),
        ui.input_action_button("solve", "Calculate the most affordable diet",
                               class_="btn-primary w-100"),
        ui.hr(),
        ui.output_ui("picker_ui"),
        ui.layout_columns(
            ui.input_action_button("add", "Add", class_="btn-outline-primary w-100"),
            ui.input_action_button("clear", "Clear all",
                                   class_="btn-outline-secondary w-100"),
            col_widths=[6, 6],
        ),
        ui.hr(),
        ui.output_ui("diet_editor"),
        width=380, open="desktop", title=None,
    ),
    ui.tags.h1("Nutritional Quality and Cost Calculator for Hypothetical Diets in Boston",
               class_="app-title"),
    ui.HTML(
        '<div class="byline">'
        "This calculator was built by Wanqing Xu, for use in teaching and research "
        "under the supervision of William Masters at the Friedman School of "
        "Nutrition at Tufts University.<br>"
        "Last revised: October 28, 2026. Contact: "
        '<a href="mailto:wanqing.xu@tufts.edu">wanqing.xu@tufts.edu</a> or '
        '<a href="mailto:william.masters@tufts.edu">william.masters@tufts.edu</a>'
        "</div>"
    ),
    ui.HTML('<hr class="rule">'),
    ui.output_ui("banner"),
    ui.output_ui("overview"),
    ui.output_ui("chart_picker"),
    ui.output_ui("sections"),
    ui.head_content(
        ui.tags.script(src=PLOTLY_JS),
        ui.tags.script(ui.HTML(VIEWPORT_JS)),
        ui.tags.script(ui.HTML(CHART_FIT_JS)),
        ui.tags.style(CSS),
    ),
    title="OnTarget - Diet Optimizer",
    window_title="Diet score & adequacy (HEI-2020)",
    fillable=False,
)


# ------------------------------------------------------------------ server ----
def server(input, output, session):
    # The diet is the single source of truth: an ordered mapping food -> grams/day.
    diet = reactive.value({})
    # Amounts that came from the solver are SHOWN rounded to whole grams; the diet
    # keeps the exact value so calories, cost and scores still match the solve.
    solver_foods = reactive.value(set())
    opt_meta = reactive.value(None)
    # Bumped whenever the diet is replaced programmatically, so the gram inputs are
    # rebuilt from the diet instead of keeping the values the browser still holds.
    ui_nonce = reactive.value(0)
    picker_nonce = reactive.value(0)
    _rm_registered: set[str] = set()

    def is_phone() -> bool:
        """Phone layout below 640 px. The probe reports a moment after the page
        connects, so treat "not reported yet" as desktop: reading the input
        unguarded would raise SilentException and leave every chart unrendered."""
        try:
            w = input.viewport_w()
        except Exception:
            return False
        return bool(w) and float(w) <= PHONE_MAX_PX

    # ---------------------------------------------------------- the criteria ---
    @reactive.calc
    def picks() -> list[str]:
        return [input[f"opt_rank_{i}"]() for i in range(len(RANK_LABELS))]

    @reactive.calc
    def ordered_priorities() -> tuple:
        """The selected criteria as an UNORDERED set: which menu a criterion sits in
        has no effect. canonical_criteria de-duplicates and sorts them into one
        fixed order, so "HEI then DRI" and "DRI then HEI" are the same request."""
        target = float(input.opt_hei_target())
        return optimize.canonical_criteria(
            (GUIDELINE_CHOICES[p], target if p == LBL_HEI else 0.0)
            for p in picks() if p != NONE_LABEL
        )

    @reactive.effect
    def _toggle_solve_button():
        # Nothing to solve with no criteria chosen.
        ui.update_action_button("solve", disabled=not ordered_priorities())

    @reactive.effect
    @reactive.event(input.solve)
    def _solve():
        crit = ordered_priorities()
        if not crit:
            return
        res = solve_criteria(crit, CAL_TARGET)
        if res.get("infeasible"):
            opt_meta.set({"infeasible": True,
                          "reason": res.get("reason", "unknown"),
                          "priorities": crit})
        else:
            solved = {k: float(v) for k, v in res["grams"].items()}
            diet.set(solved)
            solver_foods.set(set(solved))
            opt_meta.set({
                "priorities": crit,
                "applied": res.get("applied", []),
                "dropped": res.get("dropped", []),
                "notes": res.get("notes", []),
                "cost": res.get("cost", 0.0),
                "snapshot": dict(solved),
            })
            ui_nonce.set(ui_nonce.get() + 1)
        # After a solve, show the default charts again.
        for key, _lbl, default in CHART_KEYS:
            ui.update_checkbox(key, value=default)

    # ------------------------------------------------------- the food basket ---
    @render.ui
    def picker_ui():
        _ = picker_nonce()
        with reactive.isolate():
            chosen = set(diet())
        remaining = [""] + sorted(n for n in CATALOG["name"] if n not in chosen)
        return ui.TagList(
            ui.HTML(f'<div class="caption">{len(CATALOG)} foods available. '
                    "Pick items and enter grams per day.</div>"),
            ui.input_select("food_picker", "Search / add food", choices=remaining,
                            selected="", multiple=False),
        )

    @reactive.effect
    @reactive.event(input.add)
    def _add():
        try:
            pick = input.food_picker()
        except Exception:
            pick = ""
        if not pick:
            return
        with reactive.isolate():
            cur = dict(diet())
            sf = set(solver_foods())
        # New foods start at 0 g and go to the TOP of the food cart list.
        diet.set({pick: 0.0, **{k: v for k, v in cur.items() if k != pick}})
        sf.discard(pick)
        solver_foods.set(sf)
        opt_meta.set(None)                     # the diet is now user-edited
        ui_nonce.set(ui_nonce.get() + 1)
        picker_nonce.set(picker_nonce.get() + 1)

    @reactive.effect
    @reactive.event(input.clear)
    def _clear():
        diet.set({})
        solver_foods.set(set())
        opt_meta.set(None)
        ui_nonce.set(ui_nonce.get() + 1)
        picker_nonce.set(picker_nonce.get() + 1)

    def _remove(name: str):
        with reactive.isolate():
            cur = dict(diet())
            sf = set(solver_foods())
        cur.pop(name, None)
        sf.discard(name)
        diet.set(cur)
        solver_foods.set(sf)
        opt_meta.set(None)                     # the diet is now user-edited
        ui_nonce.set(ui_nonce.get() + 1)
        picker_nonce.set(picker_nonce.get() + 1)

    def _ensure_remove_effect(name: str, s: str):
        """One remove-effect per food, registered the first time its row is drawn."""
        if s in _rm_registered:
            return
        _rm_registered.add(s)

        @reactive.effect
        @reactive.event(input[f"rm_{s}"], ignore_init=True)
        def _():
            _remove(name)

    @render.ui
    def diet_editor():
        _ = ui_nonce()
        with reactive.isolate():
            cur = dict(diet())
            sf = set(solver_foods())
        if not cur:
            return note("info", "No foods selected yet. Pick one above.")
        rows = []
        for name, exact in cur.items():
            s = slug(name)
            _ensure_remove_effect(name, s)
            shown = float(int(float(exact) + 0.5)) if name in sf else float(exact)
            rows.append(ui.layout_columns(
                ui.input_numeric(f"g_{s}", name, value=shown, min=0, step=10),
                ui.input_action_button(f"rm_{s}", "✕",
                                       class_="btn-sm btn-outline-secondary",
                                       title="remove"),
                col_widths=[9, 3], class_="gram-row",
            ))
        return ui.TagList(ui.h5("Diet (grams / day)"), *rows)

    @reactive.calc
    def active() -> dict[str, float]:
        """The diet as the charts see it: foods with more than 0 g, read from the
        gram inputs so a hand edit takes effect immediately.

        Solver amounts are DISPLAYED rounded to whole grams while the diet keeps the
        exact value, so a box still holding the number it was given counts as
        untouched and the exact amount is used. Reading the rounded number back
        would quietly re-score a different diet: the optimizer's HEI-100 solution
        scores 99.9 and drifts to 2330.3 kcal once every food is rounded, and the
        optimized-diet banner would claim the user had edited it.
        """
        _ = ui_nonce()
        cur = diet()
        sf = solver_foods()
        out = {}
        for name, exact in cur.items():
            exact = float(exact)
            shown = float(int(exact + 0.5)) if name in sf else exact
            try:
                typed = float(input[f"g_{slug(name)}"]() or 0.0)
            except Exception:
                typed = shown
            value = exact if abs(typed - shown) <= 1e-9 else typed
            if value > 0:
                out[name] = value
        return out

    # --------------------------------------------------------- the main view ---
    def _diet_matches(a: dict, b: dict, tol: float = 1e-6) -> bool:
        """True if two diets have the same foods and ~equal grams."""
        if a is None or b is None or set(a) != set(b):
            return False
        return all(abs(float(a[k]) - float(b.get(k, 0.0))) <= tol for k in a)

    @render.ui
    def banner():
        meta = opt_meta()
        if not meta:
            return None
        if meta.get("infeasible"):
            return note("error", ui.HTML(
                "❌ Optimizer could not meet the selected criteria: "
                f"{_html.escape(str(meta['reason']))} "
                "(the diet below is unchanged from before)."))
        short = {"DRI": "DRI", "HEI 2020": "HEI", optimize.HDB_WITH_DIVERSITY: "HDB"}
        targets = dict(meta.get("priorities", []))
        crit_str = ", ".join(
            short.get(nm, nm) + (f" (target {int(targets[nm])})" if nm == "HEI 2020" else "")
            for nm in targets
        )
        applied = [short.get(n, n) for n in meta.get("applied", [])]
        dropped = [short.get(n, n) for n in meta.get("dropped", [])]
        msg = f"\U0001F3AF <b>Optimized diet</b> — criteria: {_html.escape(crit_str)}. "
        if dropped:
            msg += (f"They cannot all be met at once. Met: <b>{', '.join(applied)}</b>; "
                    f"not met: <b>{', '.join(dropped)}</b>. ")
        else:
            msg += "All criteria met. "
        msg += f"Cost <b>${meta.get('cost', 0.0):.2f}/day</b>."
        pristine = _diet_matches(active(), meta.get("snapshot", {}))
        box = note("success", ui.HTML(msg)) if pristine else note(
            "warning", ui.HTML(msg + "<br><i>(You've since edited it in the sidebar "
                               "— the figures below reflect your edited diet.)</i>"))
        notes = [ui.HTML(f'<div class="caption">ℹ️ {_html.escape(str(n))}</div>')
                 for n in meta.get("notes", [])]
        return ui.TagList(box, *notes)

    @render.ui
    def overview():
        act = active()
        if not act:
            return note("info", "No diet yet. Build one in the left sidebar by adding "
                                "foods, or by using the \U0001F3AF optimizer.")
        kcal = scoring.total_calories(CATALOG, act)
        cost = scoring.total_cost(CATALOG, PRICES, act)
        return ui.layout_columns(
            metric("Calories", f"{kcal:.0f} kcal", f"{kcal - CAL_TARGET:+.0f} vs target"),
            metric("Food Diversity", f"{len(act)} items"),
            metric("Cost", f"${cost:.2f} / day"),
            col_widths=[4, 4, 4],
        )

    @render.ui
    def chart_picker():
        if not active():
            return None
        return ui.TagList(
            ui.HTML('<hr class="rule">'),
            section_title("\U0001F4CA Charts to display"),
            ui.HTML('<div class="caption">Check or uncheck each criterion to show '
                    "its attributes</div>"),
            ui.layout_columns(
                *[ui.input_checkbox(key, lbl, value=default)
                  for key, lbl, default in CHART_KEYS],
                col_widths=[4, 4, 4, 4, 4, 4], class_="chart-picker",
            ),
        )

    @render.ui
    def sections():
        act = active()
        if not act:
            return None
        phone = is_phone()

        kcal = scoring.total_calories(CATALOG, act)
        totals = scoring.total_nutrients(CATALOG, act)
        adequacy = scoring.nutrient_adequacy(totals, DRI, rel_tol=BIND_TOL)
        hei_score, hei_comps, _dbg = scoring.compute_hei(CATALOG, act, HEI_BOUNDS)
        hdb = scoring.hdb_assessment(CATALOG, act)

        # DRI chart: calories are included with the nutrients. Energy is a target
        # (EER), so +/-5% counts as in-band; larger deviations push the dot out.
        cal_dev = (kcal - CAL_TARGET) / CAL_TARGET if CAL_TARGET else 0.0
        if cal_dev < -0.05:
            cal_dir, cal_sev = "deficient", min(1.0, abs(cal_dev) / 0.3)
        elif cal_dev > 0.05:
            cal_dir, cal_sev = "excessive", min(1.0, abs(cal_dev) / 0.3)
        else:
            cal_dir, cal_sev = "ok", 0.0
        dri_rows = [{
            "nutrient": "Energy", "unit": "kcal", "intake": kcal,
            "lower": CAL_TARGET, "upper": CAL_TARGET,
            "r": cal_sev, "severity": cal_sev, "direction": cal_dir,
        }] + adequacy
        # DRI score: the 21 nutrients with a lower bound (sodium has none), each
        # scored 0-100% of its lower bound and capped at 100% (as for HDB).
        dri_score = scoring.dri_mean_adequacy(adequacy)
        dri_score_text = (
            f"Adequate nutrients: {dri_score['n_adequate']} of {dri_score['n_components']}\n"
            f"Mean adequacy ratio: {100 * dri_score['mean_adequacy_ratio']:.1f} / 100%"
        )

        def checked(key: str, default: bool) -> bool:
            # The picker is drawn by another output, so on the first pass these
            # inputs may not exist yet; fall back to the default rather than
            # aborting the whole section render.
            try:
                return bool(input[key]())
            except Exception:
                return default

        show = {key: checked(key, default) for key, _l, default in CHART_KEYS}
        out = []
        if not any(show.values()):
            out.append(note("info", "Check at least one chart above to display it."))

        if show["chk_cmp"]:
            cols = []
            for title, score, fig in (
                ("Healthy Eating Index (HEI)",
                 f"Total score: {sum(hei_comps.values()):.1f} / 100",
                 viz.hei_anti_spider(hei_comps, HEI_BOUNDS)),
                ("Dietary Reference Intakes (DRI)",
                 dri_score_text,
                 viz.nutrient_anti_spider(dri_rows)),
                ("Healthy Diet Basket (HDB)",
                 f"Mean adequacy ratio: {100 * hdb['mean_adequacy']:.1f} / 100%",
                 viz.hdb_bullseye(hdb)),
            ):
                cols.append(ui.tags.div(
                    # Same header height in every column (DRI has 2 score lines), so
                    # the three charts start at the same level.
                    ui.HTML(f'<div class="cmp-head"><b>{_html.escape(title)}</b><br>'
                            + _html.escape(score).replace("\n", "<br>") + "</div>"),
                    fig_html(black_text(compact_fig(fig, phone))),
                ))
            out += [ui.HTML('<hr class="rule">'),
                    section_title("Comparison across Criteria"),
                    ontarget_note(),
                    ui.layout_columns(*cols, col_widths=[4, 4, 4])]

        if show["chk_cal"]:
            out += [ui.HTML('<hr class="rule">'), section_title("Calories"),
                    fig_html(black_text(viz.calorie_gauge(kcal, CAL_TARGET)))]

        if show["chk_hei"]:
            rows = []
            for name, max_sc, direction in data.HEI_COMPONENTS:
                sc = hei_comps.get(name, 0.0)
                mx = HEI_BOUNDS.get(name, {}).get("max_score", float(max_sc))
                rows.append({
                    "Component": name,
                    "Score": f"{sc:.1f}",
                    "Max": f"{mx:.0f}",
                    "% of max": f"{100 * sc / mx:.0f}%" if mx else "—",
                    "Direction to improve": flag(hei_direction(sc, mx, direction), "On target"),
                })
            out += [
                ui.HTML('<hr class="rule">'),
                section_title("Healthy Eating Index Components"),
                dot_legend([(viz.OK_COLOR, "Green dot: full score"),
                            (viz.EXCESSIVE_COLOR, "Red dot: 0 points")]),
                ontarget_note(),
                fig_html(black_text(round_fig(
                    viz.hei_anti_spider(hei_comps, HEI_BOUNDS), phone))),
                expander("HEI components in table view, based on HEI-2020",
                         table_html(rows)),
            ]

        if show["chk_dri"]:
            rows = []
            for r in adequacy:
                rows.append({
                    "Nutrient": f"{r['nutrient']} ({r['unit']})",
                    "Intake": plain_number(r["intake"]),
                    "Lower (RDA or AI)": "—" if r["lower"] is None else plain_number(r["lower"]),
                    "Upper (UL or CDRR)": "—" if r["upper"] is None else plain_number(r["upper"]),
                    "% of lower": "—" if not r["lower"] else f"{100 * r['intake'] / r['lower']:.0f}%",
                    "% of upper": "—" if not r["upper"] else f"{100 * r['intake'] / r['upper']:.0f}%",
                    "Status": flag(dri_status(r["direction"]), "Adequate"),
                })
            out += [
                ui.HTML('<hr class="rule">'),
                section_title("Dietary Reference Intake Components"),
                dot_legend([(viz.OK_COLOR, "Green dot: adequate"),
                            (viz.DEFICIENT_COLOR, "Blue dot: deficient"),
                            (viz.EXCESSIVE_COLOR, "Red dot: excessive")]),
                ontarget_note(),
                fig_html(black_text(round_fig(
                    with_title(viz.nutrient_anti_spider(dri_rows), dri_score_text), phone))),
                expander("DRI components in table view, based on the 2011 version",
                         table_html(rows)),
            ]

        if show["chk_hdb"]:
            out += [ui.HTML('<hr class="rule">'),
                    section_title("Healthy Diet Basket Components")]
            # Food-type diversity is enforced whenever the user picks the "with
            # diversity" variant, so the verdict is reported in every case.
            failing = [r for r in hdb["rows"] if not r["meets_diversity"]]
            if failing:
                out.append(note("warning", ui.HTML(
                    "⚠️ Minimum food-group diversity not met: "
                    + "; ".join(f"{_html.escape(r['group'])} ({r['n_items']}/"
                                f"{r['min_items']} food types)" for r in failing))))
            else:
                out.append(note("success",
                                "✅ Minimum food-group diversity met for all 6 groups."))
            if hdb["unclassified_kcal"] > 0:
                out.append(ui.HTML(
                    f'<div class="caption">Note: {hdb["unclassified_kcal"]:.0f} kcal '
                    f'({100 * hdb["excluded_share_of_diet"]:.1f}% of total diet energy) '
                    "from foods not mapped to an HDB group — shown in the "
                    "<b>Excluded foods</b> section below, but left out of the "
                    "6-group shares.</div>"))
            out += [
                dot_legend([
                    (viz.OK_COLOR, "Green dot: at or above the calorie target of this food group"),
                    (viz.EXCESSIVE_COLOR, "Red dot: no calorie intake from this food group"),
                ]),
                ontarget_note(),
                fig_html(black_text(round_fig(viz.hdb_bullseye(hdb), phone))),
            ]

            # ---- Items in each food group ----
            item_rows = {
                r["group"]: [{"Food": it["name"],
                              "Grams/day": f"{it['grams']:.1f}",
                              "kcal": f"{it['kcal']:.0f}"} for it in r["items"]]
                for r in hdb["rows"]
            }
            excl_rows = [{"Food": it["name"],
                          "Grams/day": f"{it['grams']:.1f}",
                          "kcal": f"{it['kcal']:.0f}"} for it in hdb["excluded_items"]]
            # The 3 longest entries of each column across ALL groups AND the excluded
            # foods, as invisible sizer rows, so every group's table and the
            # excluded-foods table share the same column widths.
            cols3 = ["Food", "Grams/day", "kcal"]
            all_rows = [row for rs in item_rows.values() for row in rs] + excl_rows
            longest = {c: sorted({row[c] for row in all_rows}, key=len, reverse=True)[:3]
                       for c in cols3}
            sizers = [{c: (longest[c][i] if i < len(longest[c]) else "") for c in cols3}
                      for i in range(3)]

            group_bits = [ui.HTML('<div class="caption">Diversity counts food types: '
                                  "foods with the same subgroup (e.g. two wheat foods) "
                                  "count as one type.</div>")]
            for r in hdb["rows"]:
                div = "✅" if r["meets_diversity"] else "⚠️ below min"
                group_bits.append(ui.HTML(
                    f"<p style='margin:0.6rem 0 0.2rem 0;'><b>{_html.escape(r['group'])}</b> "
                    f"— {r['current_kcal']:.0f} of {r['target_kcal']:.0f} kcal "
                    f"· {r['n_items']}/{r['min_items']} food types {div}</p>"))
                if r["items"]:
                    group_bits.append(table_html(item_rows[r["group"]], sizer_rows=sizers))
                else:
                    group_bits.append(ui.HTML('<div class="caption">— no foods in '
                                              "this group —</div>"))
            out.append(expander("Food items in each group", *group_bits, open_=True))

            # ---- Excluded foods (no HDB group) ----
            if hdb["excluded_items"]:
                ex_rows = [{
                    "Food": it["name"],
                    "Grams/day": f"{it['grams']:.1f}",
                    "kcal": f"{it['kcal']:.0f}",
                    "% of total diet energy":
                        f"{(100 * it['kcal'] / hdb['total_diet_kcal']) if hdb['total_diet_kcal'] > 0 else 0:.1f}%",
                } for it in hdb["excluded_items"]]
                excl_body = [
                    ui.HTML('<div class="caption">These foods carry an \'Exclude\' (or '
                            "unrecognized) HDB group label, so they contribute calories "
                            "to the diet but are left out of the six-group energy "
                            "shares.</div>"),
                    table_html(ex_rows, sizer_rows=sizers),
                ]
            else:
                excl_body = [note("success", "No excluded foods — every food maps "
                                             "to an HDB group.")]
            out.append(expander(
                f"Excluded foods — {hdb['unclassified_kcal']:.0f} kcal "
                f"({100 * hdb['excluded_share_of_diet']:.1f}% of total diet energy)",
                *excl_body, open_=bool(hdb["excluded_items"])))

            tbl = [{
                "Food group": r["group"],
                "Ideal % energy": f"{100 * r['ideal_share']:.1f}%",
                "Current % energy": f"{100 * r['current_share']:.1f}%",
                "Current kcal": f"{r['current_kcal']:.0f}",
                "Target kcal": f"{r['target_kcal']:.0f}",
                "% of target": f"{100 * r['pct_of_target']:.0f}%",
                "Food types": r["n_items"],
                "Min types": r["min_items"],
                "Diversity": "✅" if r["meets_diversity"] else "⚠️ below min",
            } for r in hdb["rows"]]
            out.append(expander("HDB components in table view, comparing the current "
                                "and ideal energy intake", table_html(tbl)))

        if show["chk_comp"]:
            pmap = dict(zip(PRICES["name"], PRICES["price_per_100g"]))
            rows = []
            for name, g in sorted(act.items(), key=lambda kv: -kv[1]):
                ppg = pmap.get(name, 0.0)
                rows.append({
                    "Food": name,
                    "Grams/day": f"{g:.1f}",
                    "$/100g": f"${ppg:.2f}",
                    "$/day": f"${ppg * g / 100:.2f}",
                })
            out += [ui.HTML('<hr class="rule">'), section_title("Food Breakdown"),
                    table_html(rows)]

        return ui.TagList(*out)


app = App(app_ui, server)

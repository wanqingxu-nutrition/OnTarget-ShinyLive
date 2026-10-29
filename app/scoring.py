"""HEI-2020 scoring + nutrient adequacy.

HEI components are derived directly from raw FPED columns in the catalog
(see data.py docstring). Bounds for HEI scoring are loaded from
`2. HEI Scoring Reference` via data.load_hei_bounds().

Score directions:
  - adequacy:        score = max * intake / upper          (capped at max; lower ignored)
  - adequacy_ratio:  Fatty Acids ratio (higher better)     (linear between lower=zero / upper=max)
  - moderation:      score = max if intake<=lower, 0 if intake>=upper, linear between
  - moderation_pct:  same as moderation; intake supplied as % of energy
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from data import NUTRIENT_ORDER, HEI_COMPONENTS


def total_calories(catalog: pd.DataFrame, grams: dict[str, float]) -> float:
    total = 0.0
    name_to_idx = {n: i for i, n in enumerate(catalog["name"])}
    for name, g in grams.items():
        if g <= 0 or name not in name_to_idx:
            continue
        total += float(catalog.iloc[name_to_idx[name]]["Energy"]) * g / 100.0
    return total


def total_nutrients(catalog: pd.DataFrame, grams: dict[str, float]) -> dict[str, float]:
    totals = {nutrient: 0.0 for nutrient, _ in NUTRIENT_ORDER}
    name_to_idx = {n: i for i, n in enumerate(catalog["name"])}
    for name, g in grams.items():
        if g <= 0 or name not in name_to_idx:
            continue
        r = catalog.iloc[name_to_idx[name]]
        for nutrient, _ in NUTRIENT_ORDER:
            totals[nutrient] += float(r[nutrient]) * g / 100.0
    return totals


def total_cost(catalog: pd.DataFrame, prices: pd.DataFrame, grams: dict[str, float]) -> float:
    pmap = dict(zip(prices["name"], prices["price_per_100g"]))
    total = 0.0
    for name, g in grams.items():
        if g <= 0:
            continue
        total += pmap.get(name, 0.0) * g / 100.0
    return total


def nutrient_adequacy(totals: dict[str, float], dri: dict[str, tuple],
                      rel_tol: float = 0.0) -> list[dict]:
    """Compute per-nutrient adequacy.

    r = 0 means in-band (adequate, sits at the center).
    r > 0 means off-band, pushing outward toward the edge; color encodes
       direction: deficient (blue) and excessive (red) both push outward.

    severity = r ∈ [0, 1]: 0 if adequate, 1 if maximally off-band.

    `rel_tol` (relative) gives a small band around each bound that still counts as
    in-band. This suppresses floating-point "dust" for diets that sit *exactly* on
    a bound (binding constraints), e.g. an optimized diet pressed to the UL — so a
    value 1e-14 over the limit is treated as adequate, not "excessive".
    """
    out = []
    for nutrient, unit in NUTRIENT_ORDER:
        lo, up = dri.get(nutrient, (None, None))
        intake = totals.get(nutrient, 0.0)
        lo_tol = lo * (1.0 - rel_tol) if (lo is not None) else None
        up_tol = up * (1.0 + rel_tol) if (up is not None) else None
        if lo_tol is not None and intake < lo_tol:
            direction = "deficient"
            sev = 1.0 - intake / lo if lo > 0 else 1.0
            sev = max(0.0, min(1.0, sev))
        elif up_tol is not None and intake > up_tol:
            direction = "excessive"
            sev = (intake - up) / up if up > 0 else 1.0
            sev = max(0.0, min(1.0, sev))
        else:
            direction = "ok"
            sev = 0.0
        r = sev
        out.append({
            "nutrient": nutrient, "unit": unit, "intake": intake,
            "lower": lo, "upper": up, "r": r, "severity": sev,
            "direction": direction,
        })
    return out


# ---------- HEI-2020 component intake from raw FPED, per 100 g ----------

def per100g_components(catalog: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return per-100g HEI component arrays derived from raw FPED columns.

    Treats V_LEGUMES as contributing to BOTH veg AND protein columns
    (per-food simplification of the HEI conditional rule — see QUESTIONS.md).
    """
    f_total   = catalog["F_TOTAL"].to_numpy(dtype=float)
    f_juice   = catalog["F_JUICE"].to_numpy(dtype=float)
    v_total   = catalog["V_TOTAL"].to_numpy(dtype=float)
    v_drkgr   = catalog["V_DRKGR"].to_numpy(dtype=float)
    v_legumes = catalog["V_LEGUMES"].to_numpy(dtype=float)
    g_whole   = catalog["G_WHOLE"].to_numpy(dtype=float)
    g_refined = catalog["G_REFINED"].to_numpy(dtype=float)
    d_total   = catalog["D_TOTAL"].to_numpy(dtype=float)
    pf_total  = catalog["PF_TOTAL"].to_numpy(dtype=float)
    pf_seafd_hi  = catalog["PF_SEAFD_HI"].to_numpy(dtype=float)
    pf_seafd_low = catalog["PF_SEAFD_LOW"].to_numpy(dtype=float)
    pf_soy    = catalog["PF_SOY"].to_numpy(dtype=float)
    pf_nutsds = catalog["PF_NUTSDS"].to_numpy(dtype=float)
    pf_legumes = catalog["PF_LEGUMES"].to_numpy(dtype=float)

    return {
        "tf": f_total,
        "wf": f_total - f_juice,
        "tv": v_total + v_legumes,
        "gb": v_drkgr + v_legumes,
        "wg": g_whole,
        "rg": g_refined,
        "da": d_total,
        "tp": pf_total + v_legumes,
        "sp": pf_seafd_hi + pf_seafd_low + pf_soy + pf_nutsds + pf_legumes,
    }


def _aq(intake, upper, max_score):
    return min(max_score, max_score * max(0.0, intake) / upper) if upper > 0 else 0.0


def _md(intake, lower, upper, max_score):
    if intake <= lower:
        return max_score
    if intake >= upper:
        return 0.0
    return max_score * (upper - intake) / (upper - lower)


def _aq_ratio(intake, lower, upper, max_score):
    if intake <= lower:
        return 0.0
    if intake >= upper:
        return max_score
    return max_score * (intake - lower) / (upper - lower)


def compute_hei(catalog: pd.DataFrame, grams: dict[str, float],
                hei_bounds: dict[str, dict] | None = None
                ) -> tuple[float, dict[str, float], dict[str, dict]]:
    """HEI-2020 score. Returns (total, per-component, debug)."""
    if hei_bounds is None:
        from data import load_hei_bounds
        hei_bounds = load_hei_bounds()

    x = np.zeros(len(catalog), dtype=float)
    name_to_idx = {n: i for i, n in enumerate(catalog["name"])}
    for name, g in grams.items():
        if name in name_to_idx and g > 0:
            x[name_to_idx[name]] = g
    x100 = x / 100.0

    energy = catalog["Energy"].to_numpy(dtype=float)
    tot_e = float(x100 @ energy)
    if tot_e <= 0:
        return 0.0, {}, {}

    p = 1000.0 / tot_e
    comps_per100 = per100g_components(catalog)

    intakes_per1000 = {k: float(x100 @ v) * p for k, v in comps_per100.items()}

    sfa  = float(x100 @ catalog["SFA"].to_numpy(dtype=float))
    mufa = float(x100 @ catalog["MUFA"].to_numpy(dtype=float))
    pufa = float(x100 @ catalog["PUFA"].to_numpy(dtype=float))
    sodium_g = float(x100 @ catalog["Sodium"].to_numpy(dtype=float)) / 1000.0
    add_sug_g = float(x100 @ catalog["added_sugar_g"].to_numpy(dtype=float))

    fa_ratio = (pufa + mufa) / max(sfa, 1e-9)
    sodium_per1000 = sodium_g * p
    add_sug_pct = add_sug_g * 4.0 / tot_e * 100.0
    sat_fat_pct = sfa * 9.0 / tot_e * 100.0

    s: dict[str, float] = {}
    for comp_name, _, direction in HEI_COMPONENTS:
        b = hei_bounds[comp_name]
        mx = b["max_score"]
        lo, up = b["lower"], b["upper"]
        if direction == "adequacy":
            key_map = {
                "Total Fruits": "tf", "Whole Fruits": "wf",
                "Total Vegetables": "tv", "Greens & Beans": "gb",
                "Whole Grains": "wg", "Dairy": "da",
                "Total Protein": "tp", "Seafood & Plant Protein": "sp",
            }
            s[comp_name] = _aq(intakes_per1000[key_map[comp_name]], up, mx)
        elif direction == "adequacy_ratio":
            s[comp_name] = _aq_ratio(fa_ratio, lo, up, mx)
        elif direction == "moderation":
            if comp_name == "Refined Grains":
                s[comp_name] = _md(intakes_per1000["rg"], lo, up, mx)
            elif comp_name == "Sodium":
                s[comp_name] = _md(sodium_per1000, lo, up, mx)
        elif direction == "moderation_pct":
            if comp_name == "Added Sugars":
                s[comp_name] = _md(add_sug_pct, lo, up, mx)
            elif comp_name == "Sat Fat":
                s[comp_name] = _md(sat_fat_pct, lo, up, mx)

    debug = {
        "density_intakes": {
            "Total Fruits (cup eq/1000kcal)":   intakes_per1000["tf"],
            "Whole Fruits":                     intakes_per1000["wf"],
            "Total Vegetables":                 intakes_per1000["tv"],
            "Greens & Beans":                   intakes_per1000["gb"],
            "Whole Grains (oz eq/1000kcal)":    intakes_per1000["wg"],
            "Dairy (cup eq/1000kcal)":          intakes_per1000["da"],
            "Total Protein (oz eq/1000kcal)":   intakes_per1000["tp"],
            "Seafood & Plant Protein":          intakes_per1000["sp"],
            "Fatty Acids ratio":                fa_ratio,
            "Refined Grains (oz eq/1000kcal)":  intakes_per1000["rg"],
            "Sodium (g/1000kcal)":              sodium_per1000,
            "Added Sugars (% energy)":          add_sug_pct,
            "Sat Fat (% energy)":               sat_fat_pct,
        },
        "total_energy_kcal": tot_e,
    }
    return float(sum(s.values())), s, debug


# ============================================================================
#  Constraint analysis: which bounds are MET EXACTLY vs AUTOMATICALLY satisfied
# ============================================================================
#
# Terminology used across the app:
#   - "Met exactly" (a.k.a. *binding* / active constraint): the diet sits right on
#     the limit — at the RDA floor, at the UL ceiling, or (for HDB) exactly on the
#     group-energy target. For an optimized diet these are the constraints the
#     solver had to push against; relaxing one would let the cost drop. Detected by
#     intake being within `rel_tol` of the bound.
#   - "Automatically satisfied" (a.k.a. *slack* / non-binding): the constraint holds
#     comfortably, with margin to spare — it came along for free and is not what
#     limits the diet.
#   - "Violated": outside the bound by more than the tolerance (a real shortfall or
#     excess, not floating-point dust).

def classify_dri_constraints(catalog: pd.DataFrame, grams: dict[str, float],
                             dri: dict[str, tuple], *, rel_tol: float = 1e-6) -> list[dict]:
    """Classify each DRI-bounded nutrient relative to its RDA/AI (lower) and UL/CDRR
    (upper) bounds for the given diet.

    status ∈ {"exact_lower", "exact_upper", "within", "deficient", "excessive",
              "no_bound"}.
    """
    totals = total_nutrients(catalog, grams)
    out = []
    for nutrient, unit in NUTRIENT_ORDER:
        lo, up = dri.get(nutrient, (None, None))
        intake = totals.get(nutrient, 0.0)
        if lo is not None and lo > 0 and abs(intake - lo) <= rel_tol * lo:
            status = "exact_lower"
        elif up is not None and up > 0 and abs(intake - up) <= rel_tol * up:
            status = "exact_upper"
        elif lo is not None and intake < lo:
            status = "deficient"
        elif up is not None and intake > up:
            status = "excessive"
        elif lo is not None or up is not None:
            status = "within"
        else:
            status = "no_bound"
        out.append({
            "nutrient": nutrient, "unit": unit, "intake": intake,
            "lower": lo, "upper": up, "status": status,
        })
    return out


def classify_hei_components(component_scores: dict[str, float],
                            hei_bounds: dict[str, dict], *,
                            rel_tol: float = 1e-6) -> list[dict]:
    """Classify each HEI-2020 component as maxed-out (at full marks) or below."""
    out = []
    for name, max_sc, direction in HEI_COMPONENTS:
        cap = float(hei_bounds.get(name, {}).get("max_score", max_sc))
        sc = float(component_scores.get(name, 0.0))
        at_max = (cap > 0 and abs(sc - cap) <= rel_tol * cap)
        out.append({
            "component": name, "score": sc, "max": cap,
            "direction": direction, "at_max": at_max,
        })
    return out


# ============================================================================
#  Healthy Diet Basket (HDB) assessment
# ============================================================================

def hdb_assessment(catalog: pd.DataFrame, grams: dict[str, float]) -> dict:
    """Assess a diet against the Healthy Diet Basket food-group targets.

    Returns:
      {
        "rows": [ {group, current_kcal, target_kcal, pct_of_target,
                   current_share, ideal_share, n_items, min_items,
                   meets_diversity, items}, ... ]  (6 groups),
        "total_classified_kcal": float,
        "unclassified_kcal": float,
        "total_diet_kcal": float,             # classified + excluded
        "excluded_items": [ {name, grams, kcal}, ... ],
        "excluded_share_of_diet": float,      # excluded kcal / total_diet_kcal
      }

    Shares are of the WHOLE diet, including calories from "Exclude" foods:
      current_share = group kcal / total diet kcal
      ideal_share   = HDB target share (e.g. 1160 / 2330 = 49.8% for starchy staples)
      target_kcal   = ideal_share * total diet kcal
      pct_of_target = current_share / ideal_share
      adequacy      = min(1, pct_of_target)   (0 = no intake, 1 = target met or exceeded)
    The result also carries "mean_adequacy": the average of the 6 adequacies (0-1).

    Diversity counts distinct food TYPES, not foods: foods sharing an HDB
    Subgroup (e.g. two wheat foods) are one type; a food with no subgroup is
    its own type (see data.hdb_type_of). n_items is that type count, and
    meets_diversity flags n_items >= min_items. Any chosen food (> 0 g) counts.
    Each group's `items` list holds every contributing food (name, grams, kcal,
    type); `excluded_items` holds foods that map to no HDB group ("Exclude").
    """
    from data import (HDB_GROUPS, HDB_MIN_ITEMS, HDB_MIN_ITEM_GRAMS,
                      HDB_TARGET_KCAL, hdb_group_of, hdb_type_of, hdb_ideal_shares)

    name_to_idx = {n: i for i, n in enumerate(catalog["name"])}
    energy_by_group = {g: 0.0 for g in HDB_GROUPS}
    items_by_group = {g: [] for g in HDB_GROUPS}   # list of {name, grams, kcal}
    excluded_items = []                            # foods mapped to no HDB group
    unclassified_kcal = 0.0

    # A food counts toward a group's diversity if it supplies at least
    # HDB_MIN_ITEM_GRAMS. That is 0, and 0-g foods are skipped below, so every
    # chosen food counts.
    item_floor = HDB_MIN_ITEM_GRAMS - 1e-6
    for name, g in grams.items():
        if g <= 0 or name not in name_to_idx:
            continue
        row = catalog.iloc[name_to_idx[name]]
        grp = hdb_group_of(str(row["hdb_group"]), name)
        kcal = float(row["Energy"]) * g / 100.0
        if grp is None:
            unclassified_kcal += kcal
            excluded_items.append({"name": name, "grams": g, "kcal": kcal})
            continue
        energy_by_group[grp] += kcal
        items_by_group[grp].append({
            "name": name, "grams": g, "kcal": kcal,
            "type": hdb_type_of(name, str(row.get("hdb_subgroup", "") or "")),
        })

    total = sum(energy_by_group.values())
    total_diet = total + unclassified_kcal
    ideal = hdb_ideal_shares()

    rows = []
    for grp in HDB_GROUPS:
        items = sorted(items_by_group[grp], key=lambda it: -it["kcal"])
        # Distinct TYPES among the chosen foods (same subgroup = same type).
        n_counted = len({it["type"] for it in items if it["grams"] >= item_floor})
        # Shares are of the whole diet, excluded foods included.
        share = (energy_by_group[grp] / total_diet) if total_diet > 0 else 0.0
        pct = (share / ideal[grp]) if ideal[grp] > 0 else 0.0
        rows.append({
            "group": grp,
            "current_kcal": energy_by_group[grp],
            "target_kcal": ideal[grp] * total_diet,
            "pct_of_target": pct,
            "adequacy": min(1.0, pct),
            "current_share": share,
            "ideal_share": ideal[grp],
            "n_items": n_counted,
            "min_items": HDB_MIN_ITEMS[grp],
            "meets_diversity": n_counted >= HDB_MIN_ITEMS[grp],
            "items": items,
        })

    return {
        "rows": rows,
        "total_classified_kcal": total,
        "unclassified_kcal": unclassified_kcal,
        "total_diet_kcal": total_diet,
        "excluded_items": sorted(excluded_items, key=lambda it: -it["kcal"]),
        "excluded_share_of_diet": (unclassified_kcal / total_diet) if total_diet > 0 else 0.0,
        "mean_adequacy": sum(r["adequacy"] for r in rows) / len(rows) if rows else 0.0,
    }


def dri_mean_adequacy(adequacy_rows: list[dict]) -> dict:
    """DRI mean adequacy ratio, scored the same way as HDB adequacy.

    Components: every nutrient in `adequacy_rows` that has a DRI lower bound.
    Sodium has no lower bound, so it is left out (21 nutrients with the current
    DRI sheet). Energy is not one of these rows.

    Each component's adequacy ratio runs from 0 (no intake) to 1 (intake at or
    above the lower bound), in proportion in between:
        ratio = min(1, intake / lower bound)
    Mean adequacy ratio = sum of the ratios / number of components.

    Returns {"n_components", "n_adequate", "mean_adequacy_ratio" (0-1)}.
    "n_adequate" counts the same components whose status is adequate (within
    both the lower and, where defined, the upper bound).
    """
    ratios, n_adequate = [], 0
    for r in adequacy_rows:
        lo = r.get("lower")
        if lo is None or float(lo) <= 0:
            continue
        ratios.append(min(1.0, max(0.0, float(r["intake"]) / float(lo))))
        if r.get("direction") == "ok":
            n_adequate += 1
    return {
        "n_components": len(ratios),
        "n_adequate": n_adequate,
        "mean_adequacy_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0,
    }

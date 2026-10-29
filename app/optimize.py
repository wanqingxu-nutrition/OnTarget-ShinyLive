"""LP-based optimizer: provably-minimum-cost diet at each HEI-2020 target.
***May 8 Improvement To-dos:
1) Solve the LP/MILP with FA score forced to alpha_fa. Return (z, cost) or (None, inf). --------------------------REMOVE the integer constraint for food items to make it continuous
2)   - x_i in [0, 800]                             grams of food i ----------------REMOVE in the optimization. SIMPLEX algorithm (now it's grid)
3) ENERGY_INT_SLACK_KCAL = 5.0  # actual energy will fall in [2330, 2335] ---------REMOVE energy constraints, set fixed to 2330 (SIMPLEX).
4) viz.py: the deficiency is good, for the excess, make 2X of the UL as the edge of the spider chart
***
Reformulation
-------------
At fixed total energy = calorie_target, every HEI component score is a
piecewise-linear function of x = (g1, ..., g_n). We linearize each component
with score variables s_C plus inequality constraints, giving a true LP
(except for the Fatty Acids ratio).

Variables (105 total: 93 food choice variables + 12 HEI score vars without the FA ratio-SLACK VARIABLES/constraints):
  - x_i in [0, 800]                             grams of food i ----------------REMOVE in the optimization. SIMPLEX algorithm (now it's grid)
  - s_TF, s_WF, s_TV, s_GB, s_WG, s_D, s_TP, s_SP    adequacy scores
  - s_RG, s_Na, s_AS, s_SF                          moderation scores
  (Fatty Acids score s_FA is handled separately via grid search; see below.)

Objective:
  min  sum_i (price_i / 100) * x_i

Equality:
  sum_i (energy_i / 100) * x_i = calorie_target   (kcal balance)

Per-component constraints (at fixed kcal, density is linear in x):

  Adequacy:
    s_C  <=  max_C                                                    (variable bound)
    s_C  <=  (max_C / upper_C) * density_C(x)                         (linear)

  Moderation (Refined Grains, Sodium, Added Sugars, Sat Fat):
    s_C * (upper_C - lower_C) + max_C * density_C(x)  <=  max_C * upper_C
    density_C(x)  <=  upper_C                            (keeps s_C >= 0 feasible)

HEI floor:
  sum_C s_C  >=  target  -  alpha_FA              (FA contribution is fixed = alpha_FA)

Fatty Acids ratio (NON-LINEAR (PUFA+MUFA)/SFA, grid-searched):
  For alpha_FA in {0, 1, ..., 10} (i.e., FA score in {0, 1, ..., 10}):
    add constraint  sum_i [(1.2 + 0.13 * alpha_FA) * SFA_i - PUFA_i - MUFA_i] * x_i  <=  0
  Solve LP for each alpha_FA; pick cheapest feasible.

This is exact (no relaxation gap) for our problem because (a) at the optimum the
moderation constraint density <= upper is satisfied by margin (high HEI floors
require low moderation intake anyway), and (b) the FA non-linearity is removed
by enumeration over a fine alpha grid.

Output:
  optimal_diets/diets.json    {target: {grams, score, energy_kcal, cost_usd, components, n_foods}}
"""

from __future__ import annotations
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

import data
import scoring


HEI_TARGETS = [50, 60, 70, 80, 90, 100]
OUT_DIR = Path(__file__).parent / "optimal_diets"
OUT_FILE = OUT_DIR / "diets.json"

# When grams must be integer, exact 2330 kcal is essentially infeasible —
# allow a small upper window so MILP can find feasible integer solutions.
ENERGY_INT_SLACK_KCAL = 5.0  # actual energy will fall in [2330, 2335]

# Resolution of the Fatty Acids alpha grid. 11 values (0..10 in unit steps) is
# usually sufficient — increase to 41 (0..10 in 0.25 steps) for finer optimum.
FA_ALPHA_GRID = np.linspace(0, 10, 21)  # 0, 0.5, 1.0, ..., 10.0

# Foods that the LP is forbidden from using (bounds locked to 0). Use this to
# work around source-data quality issues — e.g., the V_DRKGR inflation on raw
# kale that lets the optimizer satisfy Greens & Beans with implausibly little.
EXCLUDED_FOODS: list[str] = [
    # Raw kale's V_DRKGR (2.50 cup eq/100g) is ~3.3x USDA FPED expectation.
    # Frozen kale's value (0.61) looks correct, so it's allowed.
    "Kale, raw",
]

# Score component names in the order they appear in z (after the 93 x_i variables).
SCORE_NAMES = [
    "Total Fruits", "Whole Fruits", "Total Vegetables", "Greens & Beans",
    "Whole Grains", "Dairy", "Total Protein", "Seafood & Plant Protein",
    "Refined Grains", "Sodium", "Added Sugars", "Sat Fat",
]
ADEQUACY_KEYS_FOR_NAME = {
    "Total Fruits": "tf", "Whole Fruits": "wf",
    "Total Vegetables": "tv", "Greens & Beans": "gb",
    "Whole Grains": "wg", "Dairy": "da",
    "Total Protein": "tp", "Seafood & Plant Protein": "sp",
}

# Foods below this many grams are dropped from the displayed diet (cosmetic) —
# anything ≥ 0.02 g is shown so small but real linear contributions stay visible.
DISPLAY_TRIM_G = 0.02

# Smallest amount the optimizer gives a food it counts toward HDB diversity.
# The rule itself has NO minimum (any chosen food counts; see
# data.HDB_MIN_ITEM_GRAMS = 0), but the MILP needs a positive floor to mean
# "chosen" — at 0 it could count a food while giving it 0 g. 1 g is the smallest
# amount that still appears as a food in the cart, which rounds to whole grams.
OPT_SELECTED_MIN_G = 1.0


def _build_density_coefficients(catalog, calorie_target: float):
    """Per-x_i coefficients for each density-style quantity at fixed kcal.

    For an FPED component with per-100g value c_p100[i]:
        density_C[i] = c_p100[i] * 10 / calorie_target
    For sodium (mg per 100g, output is g/1000kcal):
        density_Na[i] = sodium_mg_p100[i] / (100 * calorie_target)
    For added sugars (g per 100g, output is % energy):
        density_AS[i] = AS_g_p100[i] * 4 / calorie_target
    For sat fat (g per 100g, output is % energy):
        density_SF[i] = SFA_p100[i] * 9 / calorie_target
    """
    cp = scoring.per100g_components(catalog)
    K = 10.0 / calorie_target  # cup/oz_eq density coefficient multiplier
    coeff = {
        "tf": cp["tf"] * K,
        "wf": cp["wf"] * K,
        "tv": cp["tv"] * K,
        "gb": cp["gb"] * K,
        "wg": cp["wg"] * K,
        "rg": cp["rg"] * K,
        "da": cp["da"] * K,
        "tp": cp["tp"] * K,
        "sp": cp["sp"] * K,
        "Na": catalog["Sodium"].to_numpy(dtype=float) / (100.0 * calorie_target),
        "AS": catalog["added_sugar_g"].to_numpy(dtype=float) * 4.0 / calorie_target,
        "SF": catalog["SFA"].to_numpy(dtype=float) * 9.0 / calorie_target,
    }
    return coeff


def _solve_lp_for_alpha(target_hei: float, alpha_fa: float, *,
                         catalog, prices_df, hei_bounds, calorie_target: float,
                         dri: dict[str, tuple] | None = None,
                         enforce_hei_floor: bool = True,
                         enforce_hdb: bool = False,
                         enforce_hdb_diversity: bool = False,
                         hdb_equal_types: bool = False,
                         integer_grams: bool = False):
    """Solve the LP/MILP with FA score forced to alpha_fa. Return (z, cost) or (None, inf).

    If `hdb_equal_types` is True (requires `enforce_hdb_diversity`), every food
    type (HDB subgroup) the diet uses within a food group must supply the SAME
    number of calories; see `_solve_with_hdb_diversity`.

    If `dri` is provided, every nutrient with a defined RDA/UL adds a linear
    constraint forcing the daily intake into [lower, upper].

    If `enforce_hei_floor` is False, the HEI total floor + the FA alpha
    constraint are dropped (used by Mode 1 = DRI only).

    If `enforce_hdb` is True, six equality rows pin each Healthy Diet Basket
    group's energy to its target (scaled to calorie_target), and foods that map
    to no HDB group are locked to 0.

    If `enforce_hdb_diversity` is True (requires `enforce_hdb`), the problem is
    solved as a MILP with one binary selection variable per HDB-eligible food.
    A linking constraint (x_i >= OPT_SELECTED_MIN_G·y_i) plus a per-group count
    row (sum of y_i in group >= HDB_MIN_ITEMS[group]) force each group to use at
    least its required number of DISTINCT foods, each ≥ OPT_SELECTED_MIN_G. The
    gram variables themselves stay free and continuous on [0, 800] (minimum 0) —
    only the selection indicators are integer — so this honors both the HDB
    energy shares and the food-item diversity rule simultaneously without
    quantizing any food's weight.

    If `integer_grams` is True, the gram variables are constrained to integers
    via `scipy.optimize.milp`, and the energy equality is relaxed to a small
    range [calorie_target, calorie_target + ENERGY_INT_SLACK_KCAL] (because
    integer grams can't usually hit a real-valued kcal target exactly).
    """
    n = len(catalog)
    n_scores = len(SCORE_NAMES)
    n_vars = n + n_scores

    # Per-x coefficients for densities
    dc = _build_density_coefficients(catalog, calorie_target)

    energy_p100 = catalog["Energy"].to_numpy(dtype=float)
    sfa_p100 = catalog["SFA"].to_numpy(dtype=float)
    mufa_p100 = catalog["MUFA"].to_numpy(dtype=float)
    pufa_p100 = catalog["PUFA"].to_numpy(dtype=float)

    pmap = dict(zip(prices_df["name"], prices_df["price_per_100g"]))
    price_p100 = np.array([pmap.get(nm, 0.0) for nm in catalog["name"]], dtype=float)

    # HDB group membership (only needed when enforce_hdb).
    names = list(catalog["name"])
    if enforce_hdb:
        raw_groups = list(catalog["hdb_group"])
        group_of = [data.hdb_group_of(str(raw_groups[i]), str(names[i])) for i in range(n)]
        raw_subs = (list(catalog["hdb_subgroup"]) if "hdb_subgroup" in catalog.columns
                    else [""] * n)
        type_of = [data.hdb_type_of(str(names[i]), str(raw_subs[i] or "")) for i in range(n)]
    else:
        group_of = [None] * n
        type_of = [None] * n

    # ---- objective: min cost = sum (price/100) * x_i ----
    c = np.zeros(n_vars)
    c[:n] = price_p100 / 100.0

    # ---- bounds ----
    # Each food's weight is a free continuous variable from 0 g up to +∞ (no
    # artificial per-food ceiling). The problem stays bounded because total
    # energy is pinned to calorie_target and costs are non-negative, so the
    # cheapest feasible diet is always finite.
    excluded = set(EXCLUDED_FOODS)
    bounds = []
    for i in range(n):
        nm_i = str(names[i])
        if nm_i in excluded or (enforce_hdb and group_of[i] is None):
            bounds.append((0.0, 0.0))   # food forbidden / unclassified for HDB
        else:
            bounds.append((0.0, np.inf))
    for nm in SCORE_NAMES:
        bounds.append((0.0, hei_bounds[nm]["max_score"]))

    # ---- equality: total energy (+ HDB group-energy targets) ----
    A_eq_rows = [np.concatenate([energy_p100 / 100.0, np.zeros(n_scores)])]
    b_eq_list = [calorie_target]
    if enforce_hdb:
        hdb_scale = calorie_target / float(data.HDB_KCAL_TOTAL)
        for grp in data.HDB_GROUPS:
            row = np.zeros(n_vars)
            for i in range(n):
                if group_of[i] == grp:
                    row[i] = energy_p100[i] / 100.0
            A_eq_rows.append(row)
            b_eq_list.append(data.HDB_TARGET_KCAL[grp] * hdb_scale)
    A_eq = np.vstack(A_eq_rows)
    b_eq = np.array(b_eq_list)

    # ---- inequality rows ----
    A_ub_rows: list[np.ndarray] = []
    b_ub_vals: list[float] = []
    score_idx = {nm: n + i for i, nm in enumerate(SCORE_NAMES)}

    # Adequacy: s_C - (max/upper) * density_C(x) <= 0
    for nm, comp_key in ADEQUACY_KEYS_FOR_NAME.items():
        b_C = hei_bounds[nm]
        upper = b_C["upper"]
        max_sc = b_C["max_score"]
        if upper <= 0:
            continue
        row = np.zeros(n_vars)
        row[score_idx[nm]] = 1.0
        row[:n] = -(max_sc / upper) * dc[comp_key]
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # Moderation: s_C * (upper-lower) + max_C * density(x) <= max_C * upper
    # AND density(x) <= upper (so s_C >= 0 stays feasible)
    moderation_specs = [
        ("Refined Grains", dc["rg"]),
        ("Sodium",         dc["Na"]),
        ("Added Sugars",   dc["AS"]),
        ("Sat Fat",        dc["SF"]),
    ]
    for nm, density_coef in moderation_specs:
        b_C = hei_bounds[nm]
        lower = b_C["lower"]; upper = b_C["upper"]; max_sc = b_C["max_score"]
        # Score relation
        row = np.zeros(n_vars)
        row[score_idx[nm]] = (upper - lower)
        row[:n] = max_sc * density_coef
        A_ub_rows.append(row)
        b_ub_vals.append(max_sc * upper)
        # Density cap (keeps s_C >= 0 feasible). If this binds tightly, the
        # solution is at the cap and s_C = 0; otherwise the score relation governs.
        row2 = np.zeros(n_vars)
        row2[:n] = density_coef
        A_ub_rows.append(row2)
        b_ub_vals.append(upper)

    # Fatty Acids ratio constraint + HEI floor are only added when enforcing HEI.
    if enforce_hei_floor:
        # Fatty Acids ratio constraint (non-linear; we fix alpha and add linear ineq):
        # PUFA + MUFA - (1.2 + 0.13*alpha) * SFA >= 0 in totals (g/day terms)
        fa_threshold = 1.2 + 0.13 * float(alpha_fa)
        fa_row = np.zeros(n_vars)
        fa_row[:n] = fa_threshold * sfa_p100 - pufa_p100 - mufa_p100
        A_ub_rows.append(fa_row)
        b_ub_vals.append(0.0)

        # HEI total floor: sum(s_C) >= target - alpha_fa
        floor_row = np.zeros(n_vars)
        for nm in SCORE_NAMES:
            floor_row[score_idx[nm]] = -1.0
        A_ub_rows.append(floor_row)
        b_ub_vals.append(-(target_hei - alpha_fa))

    # ---- DRI bounds (intake ∈ [RDA/AI, UL/CDRR]) for each nutrient ----
    # Skips Energy (already pinned by the equality constraint above).
    if dri is not None:
        for nutrient_name, (lo, up) in dri.items():
            if nutrient_name == "Energy":
                continue
            if nutrient_name not in catalog.columns:
                continue
            per100g = catalog[nutrient_name].to_numpy(dtype=float)
            coeffs = per100g / 100.0  # intake = coeffs · x
            if lo is not None and lo > 0:
                # intake >= lo  →  -intake <= -lo
                row = np.zeros(n_vars)
                row[:n] = -coeffs
                A_ub_rows.append(row)
                b_ub_vals.append(-float(lo))
            if up is not None and up > 0:
                # intake <= up
                row = np.zeros(n_vars)
                row[:n] = coeffs
                A_ub_rows.append(row)
                b_ub_vals.append(float(up))

    A_ub = np.vstack(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_ub_vals) if b_ub_vals else np.zeros((0,))

    # ---- solve ----
    if enforce_hdb_diversity:
        # MILP with binary selection vars so every HDB group uses at least its
        # required number of distinct food TYPES. Grams stay continuous.
        group_kcal = {grp: data.HDB_TARGET_KCAL[grp] * hdb_scale
                      for grp in data.HDB_GROUPS}
        return _solve_with_hdb_diversity(
            c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            n=n, n_vars=n_vars, group_of=group_of, type_of=type_of,
            hdb_equal_types=hdb_equal_types, energy_p100=energy_p100,
            group_kcal=group_kcal)

    if integer_grams:
        # MILP path: gram variables integer, score variables continuous.
        # Energy is relaxed to a small UPPER window so integer combinations
        # can hit at least 2330 kcal without overshooting.
        integrality = np.zeros(n_vars, dtype=int)
        integrality[:n] = 1
        lb_arr = np.array([b[0] for b in bounds])
        ub_arr = np.array([b[1] for b in bounds])
        bounds_obj = Bounds(lb=lb_arr, ub=ub_arr)
        constraints = [LinearConstraint(
            A_eq, lb=calorie_target, ub=calorie_target + ENERGY_INT_SLACK_KCAL,
        )]
        if A_ub.shape[0] > 0:
            constraints.append(LinearConstraint(A_ub, ub=b_ub))
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds_obj,
            options={"presolve": True, "time_limit": 30.0},
        )
        if res.x is None:
            return None, float("inf")
        return res.x, float(res.fun)

    # Continuous LP path (legacy / verification)
    res = linprog(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs-ds",
        options={"presolve": True,
                  "primal_feasibility_tolerance": 1e-10,
                  "dual_feasibility_tolerance": 1e-10},
    )
    if not res.success or res.x is None:
        return None, float("inf")
    return res.x, float(res.fun)


def _solve_with_hdb_diversity(*, c, A_ub, b_ub, A_eq, b_eq, bounds, n, n_vars,
                               group_of, type_of, hdb_equal_types=False,
                               energy_p100=None, group_kcal=None):
    """Augment the base LP with binary food-TYPE selection variables and solve as
    a MILP that enforces the HDB minimum distinct-types-per-group rule.

    A food type is its HDB Subgroup (e.g. every wheat food is one type); a food
    with no subgroup is its own type (data.hdb_type_of). Variables: the original
    `n_vars` (continuous grams + continuous HEI scores) plus one binary w_t per
    (HDB group, type). Returns (z, cost) where z is the original `n_vars`-length
    vector, or (None, inf) if infeasible.

    Equal-calorie rule (`hdb_equal_types=True`, used when HDB is the ONLY
    criterion): within each food group, every type the diet uses supplies the
    same calories. With one continuous share s_g per group and c_t = calories
    from type t's foods, the rule "c_t = s_g if type t is used, else 0" is linear:
        c_t <= M_g * w_t              (an unused type supplies nothing)
        c_t <= s_g                    (no used type supplies more than the share)
        c_t >= s_g - M_g * (1 - w_t)  (every used type supplies the full share)
    where M_g is the group's calorie target. Together with the group-energy
    equality this gives s_g = target_g / (number of types used). Foods inside one
    type may split that type's share freely.
    """
    # One binary per distinct (group, type) among HDB-eligible foods.
    types: list[tuple[str, str]] = []
    members: dict[tuple[str, str], list[int]] = {}
    for i in range(n):
        if group_of[i] is None:
            continue
        key = (group_of[i], type_of[i])
        if key not in members:
            members[key] = []
            types.append(key)
        members[key].append(i)
    n_bin = len(types)
    base = n_vars
    # Equal-calorie rule: one continuous calorie share s_g per food group, placed
    # after the binaries.
    share_groups = list(data.HDB_GROUPS) if hdb_equal_types else []
    n_share = len(share_groups)
    share_col = {g: base + n_bin + j for j, g in enumerate(share_groups)}
    n_extra = n_bin + n_share
    n_total = base + n_extra

    def _pad(M):
        if M.shape[0] == 0:
            return np.zeros((0, n_total))
        return np.hstack([M, np.zeros((M.shape[0], n_extra))])

    A_ub_p = _pad(A_ub)
    A_eq_p = _pad(A_eq)
    c_p = np.concatenate([c, np.zeros(n_extra)])

    # Linking + per-group count constraints (all expressed as <= rows).
    #
    # Gram variables stay free continuous weights (minimum 0) — they are NOT
    # clamped to the selection bits. The binary w_t only marks whether type t
    # "counts" toward its group's diversity. The rule counts any chosen type
    # (amount > 0), but a MILP cannot express "strictly > 0", so a counted type's
    # foods must together supply at least OPT_SELECTED_MIN_G:
    #     sum_{i in t} x_i >= MIN · w_t   ->   -sum x_i + MIN·w_t <= 0
    # Summing within the type means two wheat foods can never count twice: they
    # share one w_t, so they satisfy at most one unit of diversity.
    rows: list[np.ndarray] = []
    rb: list[float] = []
    for k, key in enumerate(types):
        r = np.zeros(n_total)
        for i in members[key]:
            r[i] = -1.0
        r[base + k] = OPT_SELECTED_MIN_G
        rows.append(r); rb.append(0.0)
    # Each group must select at least its minimum number of distinct types:
    #   sum_{t in g} w_t >= min_g   ->   -sum <= -min_g
    for grp in data.HDB_GROUPS:
        r = np.zeros(n_total)
        for k, (g, _t) in enumerate(types):
            if g == grp:
                r[base + k] = -1.0
        rows.append(r); rb.append(-float(data.HDB_MIN_ITEMS[grp]))

    # Equal calories across the types used within each group (HDB only).
    if hdb_equal_types:
        for k, key in enumerate(types):
            grp = key[0]
            big_m = float(group_kcal[grp])
            kcal = np.zeros(n_total)
            for i in members[key]:
                kcal[i] = energy_p100[i] / 100.0
            w, sg = base + k, share_col[grp]
            r = kcal.copy(); r[w] = -big_m        # c_t - M*w_t <= 0
            rows.append(r); rb.append(0.0)
            r = kcal.copy(); r[sg] = -1.0         # c_t - s_g <= 0
            rows.append(r); rb.append(0.0)
            r = -kcal; r[sg] = 1.0; r[w] = big_m  # -c_t + s_g + M*w_t <= M
            rows.append(r); rb.append(big_m)

    A_ub_all = np.vstack([A_ub_p] + rows) if rows else A_ub_p
    b_ub_all = np.concatenate([b_ub, np.array(rb)]) if rb else b_ub

    lb = np.array([b[0] for b in bounds] + [0.0] * n_bin + [0.0] * n_share)
    ub = np.array([b[1] for b in bounds] + [1.0] * n_bin
                  + [float(group_kcal[g]) for g in share_groups])
    integrality = np.zeros(n_total, dtype=int)
    integrality[base:base + n_bin] = 1  # only the selection indicators are integer

    constraints = [LinearConstraint(A_eq_p, lb=b_eq, ub=b_eq)]
    if A_ub_all.shape[0] > 0:
        constraints.append(LinearConstraint(A_ub_all, ub=b_ub_all))
    res = milp(
        c=c_p, constraints=constraints, integrality=integrality,
        bounds=Bounds(lb=lb, ub=ub),
        options={"presolve": True, "time_limit": 30.0},
    )
    # Accept only a PROVEN optimum. If the time limit stopped the search early,
    # HiGHS still hands back its best diet so far, but which diet that is depends
    # on timing — the same request could return different diets on different runs.
    if not res.success or res.x is None:
        return None, float("inf")
    return res.x[:n_vars], float(res.fun)


def optimize_diet_lp(target_hei: float, *, catalog, prices_df, hei_bounds,
                     calorie_target: float, fa_grid=FA_ALPHA_GRID,
                     dri: dict[str, tuple] | None = None,
                     enforce_hei_floor: bool = True,
                     enforce_hdb: bool = False,
                     enforce_hdb_diversity: bool = False,
                     hdb_equal_types: bool = False,
                     integer_grams: bool = True,
                     ) -> tuple[dict[str, float], float, float] | None:
    """Solve at every alpha_FA in fa_grid; return cheapest feasible integer-gram diet.

    With `integer_grams=True` (default) uses MILP and produces clean integer grams.
    With `enforce_hei_floor=False` skips the HEI total floor and the FA constraint
    (used for Mode 1 = DRI only).

    Returns (diet_dict, cost, achieved_hei) or None if every alpha is infeasible.
    """
    n = len(catalog)

    # If HEI is not enforced, the alpha grid is irrelevant — single solve.
    alpha_iter = list(fa_grid) if enforce_hei_floor else [0.0]

    best_z = None
    best_cost = float("inf")
    for alpha in alpha_iter:
        z, cost = _solve_lp_for_alpha(
            target_hei, alpha, catalog=catalog, prices_df=prices_df,
            hei_bounds=hei_bounds, calorie_target=calorie_target, dri=dri,
            enforce_hei_floor=enforce_hei_floor, enforce_hdb=enforce_hdb,
            enforce_hdb_diversity=enforce_hdb_diversity,
            hdb_equal_types=hdb_equal_types,
            integer_grams=integer_grams,
        )
        if z is not None and cost < best_cost:
            best_cost = cost
            best_z = z

    if best_z is None:
        return None
    x = best_z[:n].copy()

    if integer_grams:
        # MILP returns integer grams already; just round defensively to clean int.
        x = np.round(x).astype(float)
    else:
        # Continuous LP: rescale to calorie_target exactly.
        energy_p100 = catalog["Energy"].to_numpy(dtype=float)
        tot_kcal = float((x / 100.0) @ energy_p100)
        if tot_kcal > 0:
            x = x * (calorie_target / tot_kcal)

    # Build diet dict.
    threshold = DISPLAY_TRIM_G if not integer_grams else 0.99
    diet = {str(catalog.iloc[i]["name"]): float(x[i])
            for i in range(n) if x[i] >= threshold}
    score, _, _ = scoring.compute_hei(catalog, diet, hei_bounds)

    # For continuous LP, salvage tiny floor slack.
    if not integer_grams and enforce_hei_floor and score + 1e-6 < target_hei:
        diet, score = _close_floor_gap(diet, target_hei, catalog, prices_df,
                                         hei_bounds, calorie_target)
    return diet, best_cost, score


def _close_floor_gap(diet: dict[str, float], target: float, catalog, prices_df,
                     hei_bounds, calorie_target: float, max_iter: int = 30):
    """Nudge tiny extra grams of the cheapest contributing food into each below-max
    adequacy component until total HEI >= target. Used only to absorb LP solver
    floating-point slack (typical gap ~3e-4)."""
    pmap = dict(zip(prices_df["name"], prices_df["price_per_100g"]))
    cp = scoring.per100g_components(catalog)
    name_to_idx = {n: i for i, n in enumerate(catalog["name"])}

    # Component → best food (cheapest non-zero contributor) lookup
    adequacy_keys = {"Total Fruits": "tf", "Whole Fruits": "wf",
                     "Total Vegetables": "tv", "Greens & Beans": "gb",
                     "Whole Grains": "wg", "Dairy": "da",
                     "Total Protein": "tp", "Seafood & Plant Protein": "sp"}

    score, comps, _ = scoring.compute_hei(catalog, diet, hei_bounds)
    for _ in range(max_iter):
        if score + 1e-9 >= target:
            break
        # Find the component most below max
        worst_name = max(
            adequacy_keys.keys(),
            key=lambda nm: hei_bounds[nm]["max_score"] - comps.get(nm, 0)
        )
        comp_arr = cp[adequacy_keys[worst_name]]
        # Cheapest food that contributes
        candidates = [(i, pmap.get(catalog.iloc[i]["name"], 0.0))
                      for i in range(len(catalog)) if comp_arr[i] > 0]
        if not candidates:
            break
        candidates.sort(key=lambda kv: kv[1])
        cheapest_idx = candidates[0][0]
        food_name = str(catalog.iloc[cheapest_idx]["name"])
        diet[food_name] = diet.get(food_name, 0.0) + 0.5  # add 0.5 g
        score, comps, _ = scoring.compute_hei(catalog, diet, hei_bounds)
    return diet, score


# ============================================================================
#  Live continuous-LP optimizer (solved in-app, no diets.json)
# ============================================================================

# Guidelines the optimizer understands. "Thrifty Food Plan" is listed for the UI
# but contributes NO constraints yet (its rules have not been supplied), so it is
# effectively a no-op until defined.
#
# The Healthy Diet Basket is offered in two variants that solve SEPARATELY:
#   - "…(with diversity)"    → MILP: group-energy shares + minimum distinct items
#                              per group (binary selection vars; grams stay linear)
#   - "…(without diversity)" → pure continuous LP: group-energy shares ONLY
#                              (the min-items rule is not applied)
HDB_WITH_DIVERSITY = "Healthy Diet Basket (with diversity)"
HDB_WITHOUT_DIVERSITY = "Healthy Diet Basket (without diversity)"
OPT_GUIDELINES = [
    "DRI", "HEI 2020", "Thrifty Food Plan",
    HDB_WITH_DIVERSITY, HDB_WITHOUT_DIVERSITY,
]


def _solve_guideline_set(active, *, catalog, prices_df, hei_bounds, dri,
                         calorie_target, hdb_only=False):
    """Solve the cheapest continuous diet enforcing EVERY guideline in `active`.

    `active` is an ordered list of (name, hei_target) tuples. All listed
    guidelines' constraints are imposed simultaneously (min-cost objective).
    Returns (grams_dict, cost) or None if infeasible.
    """
    names = [a[0] for a in active]
    enforce_hei = "HEI 2020" in names
    enforce_dri = "DRI" in names
    hdb_with = HDB_WITH_DIVERSITY in names
    hdb_without = HDB_WITHOUT_DIVERSITY in names
    enforce_hdb = hdb_with or hdb_without
    # Food-type diversity applies whenever the user picks the "with diversity"
    # variant, on its own or alongside DRI and/or HEI.
    enforce_div = hdb_with
    # The equal-calorie rule (every used type in a group supplies the same
    # calories) is relaxed when HDB is combined with another criterion: stacked on
    # DRI bounds and an HEI floor it is usually infeasible, and it is the one HDB
    # rule that is a modelling choice rather than part of the guideline.
    enforce_equal = hdb_with and hdb_only
    hei_target = next((t for (nm, t) in active if nm == "HEI 2020"), 0.0)
    # "Thrifty Food Plan" adds nothing (undefined) — silently contributes no rules.

    out = optimize_diet_lp(
        target_hei=float(hei_target), catalog=catalog, prices_df=prices_df,
        hei_bounds=hei_bounds, calorie_target=calorie_target,
        dri=(dri if enforce_dri else None),
        enforce_hei_floor=enforce_hei, enforce_hdb=enforce_hdb,
        enforce_hdb_diversity=enforce_div,  # only the "with diversity" variant
        hdb_equal_types=enforce_equal,
        integer_grams=False,
    )
    if out is None:
        return None
    diet, cost, _ = out
    return diet, cost


# Fixed internal order of the criteria. The order the user picks them in must
# never affect the result, so every request is normalised to this order first.
CANONICAL_ORDER = ["DRI", "HEI 2020", HDB_WITH_DIVERSITY, HDB_WITHOUT_DIVERSITY,
                   "Thrifty Food Plan"]


def canonical_criteria(criteria) -> tuple:
    """Normalise a selection of (name, hei_target) pairs so order is irrelevant.

    Unknown names are ignored, duplicates collapse to their first occurrence, and
    the result is sorted into CANONICAL_ORDER. Any two selections of the same
    criteria (and HEI target) therefore produce an identical tuple — which is
    also what the app caches on.
    """
    chosen: dict[str, float] = {}
    for name, target in criteria:
        if name in OPT_GUIDELINES and name not in chosen:
            chosen[name] = float(target)
    return tuple(sorted(chosen.items(), key=lambda kv: CANONICAL_ORDER.index(kv[0])))


def optimize_criteria(criteria, *, catalog, prices_df, hei_bounds, dri,
                      calorie_target: float) -> dict:
    """Cheapest diet meeting a SET of criteria. The order they are given in does
    not matter.

    `criteria` is any iterable of (name, hei_target) pairs. It is normalised with
    `canonical_criteria`, then:
      1. Try to meet ALL selected criteria at once (minimum cost).
      2. If that is impossible, try every combination with one criterion fewer,
         then two fewer, and so on. Keep the combination that meets the MOST
         criteria; among those, the LOWEST cost. (Exact cost ties keep the first
         combination in CANONICAL_ORDER, so the result is still order-free.)

    Returns:
      {"grams": {...}, "cost": float, "applied": [names...], "dropped": [names...],
       "notes": [str...]}   (applied / dropped listed in CANONICAL_ORDER)
      or {"infeasible": True, "reason": str} if no selected criterion can be met.
    """
    crit = canonical_criteria(criteria)
    if not crit:
        return {"infeasible": True, "reason": "Select at least one criterion to optimize for."}

    notes = []
    names = [nm for nm, _ in crit]
    if "Thrifty Food Plan" in names:
        notes.append("Thrifty Food Plan has no rules defined yet, so it imposes no "
                     "constraints (acts as a placeholder).")
    if HDB_WITHOUT_DIVERSITY in names and HDB_WITH_DIVERSITY not in names:
        notes.append("Healthy Diet Basket (without diversity): only the six "
                     "group-energy shares are enforced (continuous LP). The minimum "
                     "distinct-items-per-group rule is NOT applied — expect groups to "
                     "collapse onto the cheapest single food.")

    # The equal-calorie-per-type rule switches on only when HDB is the single
    # criterion the user chose (food-type diversity now applies in every case).
    # Decided from the user's selection, so a fallback subset of a larger
    # selection never turns it on.
    hdb_only = names == [HDB_WITH_DIVERSITY]

    for size in range(len(crit), 0, -1):
        best = None   # (subset, diet, cost)
        for subset in itertools.combinations(crit, size):
            out = _solve_guideline_set(
                list(subset), catalog=catalog, prices_df=prices_df,
                hei_bounds=hei_bounds, dri=dri, calorie_target=calorie_target,
                hdb_only=hdb_only)
            if out is None:
                continue
            diet, cost = out
            if best is None or cost < best[2] - 1e-9:
                best = (subset, diet, cost)
        if best is not None:
            subset, diet, cost = best
            kept = {nm for nm, _ in subset}
            return {
                "grams": diet, "cost": cost,
                "applied": [nm for nm in names if nm in kept],
                "dropped": [nm for nm in names if nm not in kept],
                "notes": notes,
            }

    return {"infeasible": True,
            "reason": f"None of the selected criteria can be met at "
                      f"{calorie_target:.0f} kcal."}


# Backward-compatible name (it no longer depends on priority order).
optimize_lexicographic = optimize_criteria


def _build_result_record(diet, target, catalog, prices_df, hei_bounds, dri):
    """Score the diet, check DRI compliance, return a JSON-friendly record."""
    score, comps, debug = scoring.compute_hei(catalog, diet, hei_bounds)
    cost = scoring.total_cost(catalog, prices_df, diet)
    kcal = debug["total_energy_kcal"]

    # Verify DRI compliance from the actual gram values
    dri_violations: list[str] = []
    if dri is not None:
        totals = scoring.total_nutrients(catalog, diet)
        for nutrient, (lo, up) in dri.items():
            if nutrient == "Energy" or nutrient not in catalog.columns:
                continue
            intake = totals.get(nutrient, 0.0)
            if lo is not None and lo > 0 and intake + 1e-3 < lo:
                dri_violations.append(f"{nutrient} {intake:.2f} < {lo}")
            if up is not None and up > 0 and intake > up + 1e-3:
                dri_violations.append(f"{nutrient} {intake:.2f} > {up}")

    meets_hei = (target is None) or (score + 1e-3 >= target)
    return {
        "grams": {k: int(round(v)) for k, v in diet.items()},
        "score": round(score, 3),
        "energy_kcal": round(kcal, 1),
        "cost_usd": round(cost, 2),
        "components": {k: round(v, 3) for k, v in comps.items()},
        "n_foods": len(diet),
        "meets_hei": bool(meets_hei),
        "dri_violations": dri_violations,
    }


def _solve_dri_only(*, catalog, prices_df, hei_bounds, calorie_target, dri):
    """Mode 1: cheapest integer-gram diet that meets DRI bounds. No HEI floor."""
    print(f"\n{'='*60}\n  Mode 1 — DRI only (no HEI floor, integer grams)\n{'='*60}")
    out = optimize_diet_lp(
        target_hei=0.0, catalog=catalog, prices_df=prices_df,
        hei_bounds=hei_bounds, calorie_target=calorie_target,
        dri=dri, enforce_hei_floor=False, integer_grams=True,
    )
    if out is None:
        print("  INFEASIBLE — could not find an integer-gram diet meeting all DRI bounds.")
        return {"infeasible": True, "reason": "no integer-gram diet meets all DRI bounds"}
    diet, _, _ = out
    rec = _build_result_record(diet, target=None, catalog=catalog,
                                 prices_df=prices_df, hei_bounds=hei_bounds, dri=dri)
    print(f"  Cost=${rec['cost_usd']:.2f}, Cal={rec['energy_kcal']:.0f}, "
          f"foods={rec['n_foods']}, HEI={rec['score']:.2f}, "
          f"DRI violations={len(rec['dri_violations'])}")
    return rec


def _solve_per_target_series(label, *, catalog, prices_df, hei_bounds,
                               calorie_target, dri):
    """Modes 2 and 3: cheapest integer-gram diet at each HEI target."""
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    results: dict[str, dict] = {}
    for target in HEI_TARGETS:
        out = optimize_diet_lp(
            target_hei=float(target), catalog=catalog, prices_df=prices_df,
            hei_bounds=hei_bounds, calorie_target=calorie_target,
            dri=dri, enforce_hei_floor=True, integer_grams=True,
        )
        if out is None:
            print(f"  HEI {target}: INFEASIBLE (no integer-gram diet meets the constraints)")
            results[str(target)] = {"infeasible": True,
                                     "reason": "no integer-gram diet feasible at this HEI target"}
            continue
        diet, _, _ = out
        rec = _build_result_record(diet, target=float(target), catalog=catalog,
                                     prices_df=prices_df, hei_bounds=hei_bounds, dri=dri)
        results[str(target)] = rec
        meets = "YES" if rec["meets_hei"] else "NO"
        viol = f", DRI violations={len(rec['dri_violations'])}" if dri else ""
        print(f"  HEI {target}: HEI={rec['score']:.2f} ({meets}), "
              f"Cal={rec['energy_kcal']:.0f}, Cost=${rec['cost_usd']:.2f}, "
              f"foods={rec['n_foods']}{viol}")
    return results


def main():
    OUT_DIR.mkdir(exist_ok=True)
    catalog = data.load_catalog()
    prices_df = data.load_prices()
    hei_bounds = data.load_hei_bounds()
    dri = data.load_dri("female")
    calorie_target = float(dri["Energy"][0])

    print(f"MILP optimizer (integer grams, 3 modes)")
    print(f"Calorie target: {calorie_target} kcal "
          f"(actual integer-gram diets may land in [{calorie_target:.0f}, "
          f"{calorie_target + ENERGY_INT_SLACK_KCAL:.0f}])")
    print(f"Catalog: {len(catalog)} foods")
    print(f"FA grid: {len(FA_ALPHA_GRID)} values from {FA_ALPHA_GRID[0]} to {FA_ALPHA_GRID[-1]}")

    dri_only = _solve_dri_only(
        catalog=catalog, prices_df=prices_df, hei_bounds=hei_bounds,
        calorie_target=calorie_target, dri=dri,
    )
    hei_only = _solve_per_target_series(
        "Mode 2 — HEI only (no DRI bounds, integer grams)",
        catalog=catalog, prices_df=prices_df, hei_bounds=hei_bounds,
        calorie_target=calorie_target, dri=None,
    )
    with_dri = _solve_per_target_series(
        "Mode 3 — HEI + DRI bounds (integer grams)",
        catalog=catalog, prices_df=prices_df, hei_bounds=hei_bounds,
        calorie_target=calorie_target, dri=dri,
    )

    payload = {
        "dri_only": dri_only,
        "hei_only": hei_only,
        "with_dri": with_dri,
        "_meta": {
            "calorie_target_kcal": calorie_target,
            "energy_int_slack_kcal": ENERGY_INT_SLACK_KCAL,
            "integer_grams": True,
            "fa_alpha_grid_size": len(FA_ALPHA_GRID),
            "excluded_foods": EXCLUDED_FOODS,
        },
    }
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_FILE} (3 modes: 'dri_only', 'hei_only', 'with_dri')")


if __name__ == "__main__":
    main()

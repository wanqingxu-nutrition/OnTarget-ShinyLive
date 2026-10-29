"""Data loading. Every input comes from the app's own `data/` folder.

FOOD data (the 93 Stop & Shop foods) is assembled from exactly four sources:
  data/WXSep11_FoodGroupConvert+WM.xlsx, Sheet1   -> the ONLY source of food
                                                     categories: col B HEI
                                                     Category, col C HDB Category,
                                                     col D HDB Food Type
  data/New_Nutrient_Data.xlsx "3. Food Price"            -> unit cost ($/100 g)
  data/New_Nutrient_Data.xlsx "4. Nutrient Composition"  -> identifiers, energy
                                                            and 22 nutrients /100 g
  data/New_Nutrient_Data.xlsx "5. HEI -FPED"             -> FPED cup/oz-equivalents
                                                            and PUFA/MUFA/SFA /100 g
The same food can appear in all four; rows are joined on the food name, and the
loader refuses to run if any food is missing from any source.

GUIDELINE definitions (not food data) come from the same workbook:
  "1.NutrientRequirements"    -> DRI lower/upper bounds
  "2. HEI Scoring Reference"  -> HEI-2020 scoring standards

Columns are located by their HEADER TEXT, not by position, so inserting or
reordering columns in a sheet does not silently shift values.

HEI-2020 components are derived from raw FPED columns per the official methodology:
  Total Fruits     = F_TOTAL
  Whole Fruits     = F_TOTAL - F_JUICE
  Total Vegetables = V_TOTAL + V_LEGUMES        
  Greens & Beans   = V_DRKGR + V_LEGUMES        
  Whole Grains     = G_WHOLE
  Refined Grains   = G_REFINED
  Dairy            = D_TOTAL
  Total Protein    = PF_TOTAL + V_LEGUMES       
  Seafood & Plant  = PF_SEAFD_HI + PF_SEAFD_LOW + PF_SOY + PF_NUTSDS + PF_LEGUMES

Added Sugars unit conversion uses the standard USDA FPED constant 1 tsp eq = 4.2 g sugar.
"""

from __future__ import annotations
from pathlib import Path
import io
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SOURCE_XLSX = DATA_DIR / "New_Nutrient_Data.xlsx"
FOOD_GROUP_XLSX = DATA_DIR / "WXSep11_FoodGroupConvert+WM.xlsx"

SHEET_PRICE = "3. Food Price"
SHEET_NUTRIENTS = "4. Nutrient Composition"
SHEET_FPED = "5. HEI -FPED"
NAME_HEADER = "Item Description"          # food-name column in sheets 3/4/5
FG_NAME_HEADER = "Food Item"              # food-name column (col A) in the group file

ADDED_SUGAR_G_PER_TSP = 4.2  # USDA FPED standard conversion (documented in FPED methodology)

NUTRIENT_ORDER = [
    ("Protein", "g"),
    ("Fat", "g"),
    ("Carbohydrate", "g"),
    ("Fiber", "g"),
    ("Calcium", "mg"),
    ("Iron", "mg"),
    ("Magnesium", "mg"),
    ("Phosphorus", "mg"),
    ("Potassium", "mg"),
    ("Sodium", "mg"),
    ("Zinc", "mg"),
    ("Copper", "mg"),
    ("Selenium", "mcg"),
    ("Vit. C", "mg"),
    ("Thiamin", "mg"),
    ("Riboflavin", "mg"),
    ("Niacin", "mg"),
    ("Vit. B6", "mg"),
    ("Folate", "mcg"),
    ("Vit. B12", "mcg"),
    ("Vit. A", "mcg RAE"),
    ("Vit. E", "mg"),
]

# Energy + 22 nutrients, read from SHEET_NUTRIENTS.  internal name -> header text.
NUTRIENT_COMP_COLS = {"Energy": "Energy (kcal)"} | {
    name: f"{name} ({unit})" for name, unit in NUTRIENT_ORDER
}

# FPED equivalents + fatty acids, read from SHEET_FPED.  internal name -> header.
# (Keys are also the tokens an uploaded datasheet's headers are matched against.)
FPED_COLS = {
    **{k: f"{k} (cup eq)" for k in (
        "F_TOTAL", "F_CITMLB", "F_OTHER", "F_JUICE",
        "V_TOTAL", "V_DRKGR", "V_REDOR_TOTAL", "V_REDOR_TOMATO", "V_REDOR_OTHER",
        "V_STARCHY_TOTAL", "V_STARCHY_POTATO", "V_STARCHY_OTHER", "V_OTHER",
        "V_LEGUMES", "D_TOTAL", "D_MILK", "D_YOGURT", "D_CHEESE")},
    **{k: f"{k} (oz eq)" for k in (
        "G_TOTAL", "G_WHOLE", "G_REFINED",
        "PF_TOTAL", "PF_MPS_TOTAL", "PF_MEAT", "PF_CUREDMEAT", "PF_ORGAN",
        "PF_POULT", "PF_SEAFD_HI", "PF_SEAFD_LOW", "PF_EGGS", "PF_SOY",
        "PF_NUTSDS", "PF_LEGUMES")},
    "OILS_g": "OILS (grams)",
    "SOLID_FATS_g": "SOLID_FATS (grams)",
    "ADD_SUGARS_tsp": "ADD_SUGARS (tsp eq)",
    "A_DRINKS": "A_DRINKS (no. of drinks)",
    "PUFA": "PUFA (g)",
    "MUFA": "MUFA (g)",
    "SFA": "SFA (g)",
    "Sodium_dup": "Sodium (mg)",
}

# Numeric identifiers, read from SHEET_NUTRIENTS.  internal name -> header text.
ID_COLS = {
    "wweia_id": "WWEIA ID",
    "stop_shop_id": "Stop & Shop ID",
    "usda_fdc_id": "USDA FDC ID",
    "ndb_food_code": "NDB Number / Food Code",
    "fped_code": "FPED Code",
}

# Group labels, read from FOOD_GROUP_XLSX.  internal name -> header text.
# Column B "HEI Category" may list several categories separated by ";".
FOOD_GROUP_COLS = {
    "hei_category": "HEI Category",    # col B
    "hdb_group": "HDB Category",       # col C: 6 HDB groups or "Exclude"
    "hdb_subgroup": "HDB Food Type",   # col D: same type = one food for diversity
}

# DRI sheet layout: row 1 = nutrient header (cols 1..23 for Energy..Vit. E),
# row 2 = "Female", row 3 = lower bound, row 4 = upper bound,
# row 5 = "Male", row 6 = lower bound, row 7 = upper bound.
DRI_HEADER = {
    "Energy": 1, "Protein": 2, "Fat": 3, "Carbohydrate": 4, "Fiber": 5,
    "Calcium": 6, "Iron": 7, "Magnesium": 8, "Phosphorus": 9,
    "Potassium": 10, "Sodium": 11, "Zinc": 12, "Copper": 13, "Selenium": 14,
    "Vit. C": 15, "Thiamin": 16, "Riboflavin": 17, "Niacin": 18, "Vit. B6": 19,
    "Folate": 20, "Vit. B12": 21, "Vit. A": 22, "Vit. E": 23,
}

# HEI-2020 component definitions. Bound values are PARSED from the
# "2. HEI Scoring Reference" sheet; the entries below define COLUMN POSITION
# on that sheet (cols 5..17 in row 4) plus max-score / direction (rows 6..10).
HEI_COMPONENTS = [
    ("Total Fruits", 5, "adequacy"),
    ("Whole Fruits", 5, "adequacy"),
    ("Total Vegetables", 5, "adequacy"),
    ("Greens & Beans", 5, "adequacy"),
    ("Whole Grains", 10, "adequacy"),
    ("Dairy", 10, "adequacy"),
    ("Total Protein", 5, "adequacy"),
    ("Seafood & Plant Protein", 5, "adequacy"),
    ("Fatty Acids", 10, "adequacy_ratio"),
    ("Refined Grains", 10, "moderation"),
    ("Sodium", 10, "moderation"),
    ("Added Sugars", 10, "moderation_pct"),
    ("Sat Fat", 10, "moderation_pct"),
]


def _num(val):
    """Parse a numeric value, tolerating None / NaN / strings like '<= 1.8 oz equiv./1,000 kcal'."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return None
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        m = re.search(r"[-+]?\d*\.?\d+", s)
        if m:
            return float(m.group(0))
    return None


def _clean_header(h) -> str:
    """Collapse line breaks / repeated spaces: 'PUFA \\n(g)' -> 'PUFA (g)'."""
    return re.sub(r"\s+", " ", str(h)).strip()


def _read_food_sheet(sheet: str) -> tuple[list[str], pd.DataFrame]:
    """Read one of sheets 3/4/5. Row 3 (index 2) holds the headers; foods start
    on the next row. Returns (cleaned headers, food rows keyed by column index).
    Rows without a food name are dropped."""
    raw = pd.read_excel(SOURCE_XLSX, sheet_name=sheet, header=None)
    headers = [_clean_header(h) for h in raw.iloc[2]]
    rows = raw.iloc[3:].reset_index(drop=True)
    name_idx = _header_index(headers, NAME_HEADER, sheet)
    names = rows[name_idx].astype(str).str.strip()
    keep = rows[name_idx].notna() & (names != "") & (names != "nan")
    rows = rows[keep].reset_index(drop=True)
    rows[name_idx] = names[keep].reset_index(drop=True)
    return headers, rows


def _header_index(headers: list[str], label: str, where: str) -> int:
    """Position of the FIRST column titled `label` (sheet 4 repeats 'Sodium (mg)')."""
    try:
        return headers.index(label)
    except ValueError:
        raise ValueError(f"Column '{label}' not found in {where}.") from None


def _require_same_foods(base: set[str], other: set[str], where: str) -> None:
    """Every data source must list exactly the same foods, or joins go wrong."""
    missing, extra = sorted(base - other), sorted(other - base)
    if missing or extra:
        raise ValueError(
            f"{where} does not list the same foods as '{SHEET_NUTRIENTS}'. "
            f"Missing: {missing or 'none'}. Not in '{SHEET_NUTRIENTS}': {extra or 'none'}."
        )


def load_food_groups() -> pd.DataFrame:
    """Group labels per food from data/FoodGroupConvert.xlsx (first sheet)."""
    raw = pd.read_excel(FOOD_GROUP_XLSX, sheet_name="Sheet1")
    raw.columns = [_clean_header(c) for c in raw.columns]
    for label in [FG_NAME_HEADER, *FOOD_GROUP_COLS.values()]:
        if label not in raw.columns:
            raise ValueError(f"Column '{label}' not found in {FOOD_GROUP_XLSX.name}.")
    names = raw[FG_NAME_HEADER].astype(str).str.strip()
    keep = raw[FG_NAME_HEADER].notna() & (names != "") & (names != "nan")
    out = pd.DataFrame({"name": names[keep]})
    for col, label in FOOD_GROUP_COLS.items():
        vals = raw.loc[keep, label]
        out[col] = vals.where(vals.notna(), "").astype(str).str.strip()
    if out["name"].duplicated().any():
        dups = sorted(out.loc[out["name"].duplicated(), "name"])
        raise ValueError(f"{FOOD_GROUP_XLSX.name} lists these foods twice: {dups}")
    return out.reset_index(drop=True)


def load_catalog() -> pd.DataFrame:
    """Assemble the food catalog from its four sources (see module docstring).

    Returns one row per food with columns:
      - name, wweia_id, stop_shop_id, usda_fdc_id, ndb_food_code, fped_code
      - hdb_group, hdb_subgroup, hei_category, food_category   (FoodGroupConvert)
      - Energy, Protein, Fat, ..., Vit. E    (per 100 g,  4. Nutrient Composition)
      - F_TOTAL ... A_DRINKS, PUFA, MUFA, SFA, Sodium_dup   (per 100 g, 5. HEI -FPED)
      - added_sugar_g  (= ADD_SUGARS_tsp * 4.2,   USDA FPED constant)
    """
    # --- 4. Nutrient Composition: identifiers + energy + 22 nutrients -------
    h4, r4 = _read_food_sheet(SHEET_NUTRIENTS)
    out = pd.DataFrame({"name": r4[_header_index(h4, NAME_HEADER, SHEET_NUTRIENTS)]})
    for col, label in ID_COLS.items():
        out[col] = pd.to_numeric(r4[_header_index(h4, label, SHEET_NUTRIENTS)],
                                 errors="coerce")
    for nutrient, label in NUTRIENT_COMP_COLS.items():
        out[nutrient] = pd.to_numeric(r4[_header_index(h4, label, SHEET_NUTRIENTS)],
                                      errors="coerce").fillna(0.0)
    if out["name"].duplicated().any():
        raise ValueError(f"'{SHEET_NUTRIENTS}' lists a food twice: "
                         f"{sorted(out.loc[out['name'].duplicated(), 'name'])}")
    foods = set(out["name"])

    # --- 5. HEI -FPED: FPED equivalents + fatty acids -----------------------
    h5, r5 = _read_food_sheet(SHEET_FPED)
    fped = pd.DataFrame({"name": r5[_header_index(h5, NAME_HEADER, SHEET_FPED)]})
    for key, label in FPED_COLS.items():
        fped[key] = pd.to_numeric(r5[_header_index(h5, label, SHEET_FPED)],
                                  errors="coerce").fillna(0.0)
    _require_same_foods(foods, set(fped["name"]), f"'{SHEET_FPED}'")
    out = out.merge(fped, on="name", how="left", validate="one_to_one")

    # --- FoodGroupConvert.xlsx: group labels --------------------------------
    groups = load_food_groups()
    _require_same_foods(foods, set(groups["name"]), FOOD_GROUP_XLSX.name)
    out = out.merge(groups, on="name", how="left", validate="one_to_one")

    out["added_sugar_g"] = out["ADD_SUGARS_tsp"] * ADDED_SUGAR_G_PER_TSP
    # The group file has no separate "Food Category" column any more; the food's
    # category shown on the page is its HEI Category (col B).
    out["food_category"] = out["hei_category"]
    return out.sort_values("name").reset_index(drop=True)


def hei_categories(label: str) -> list[str]:
    """Split a col-B HEI Category cell: 'Total Fruits; Whole Fruits' -> 2 items."""
    return [c.strip() for c in str(label or "").split(";") if c.strip()]


def load_prices() -> pd.DataFrame:
    """Return DataFrame with columns: name, price_per_100g (from `3. Food Price`)."""
    h3, r3 = _read_food_sheet(SHEET_PRICE)
    out = pd.DataFrame({
        "name": r3[_header_index(h3, NAME_HEADER, SHEET_PRICE)],
        "price_per_100g": pd.to_numeric(
            r3[_header_index(h3, "Unit cost ($/100g)", SHEET_PRICE)],
            errors="coerce").fillna(0.0),
    })
    # A food absent here would be priced at $0 downstream (a free food the
    # optimizer would favour), so insist the price list matches the catalog.
    h4, r4 = _read_food_sheet(SHEET_NUTRIENTS)
    _require_same_foods(set(r4[_header_index(h4, NAME_HEADER, SHEET_NUTRIENTS)]),
                        set(out["name"]), f"'{SHEET_PRICE}'")
    return out.reset_index(drop=True)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name in `candidates` that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _dedupe_names(names: list[str]) -> list[str]:
    """Make a list of food names unique by suffixing repeats ' (2)', ' (3)', …
    (the app keys diets by food name, so duplicates would otherwise collide)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n} ({seen[n]})")
        else:
            seen[n] = 1
            out.append(n)
    return out


def load_catalog_from_filled(src) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt an uploaded *filled* datasheet (the pseudo-match format) into the
    internal (catalog, prices) pair the app expects.

    Recognized columns (matched by header; missing → 0 per the project rule):
      - name:        'Item Name' | 'name' | 'Food' | 'Item Description'
      - HDB group:   'HDB Food Group' | 'hdb_group'  (canonical 6 groups or 'Exclude')
      - category:    'Food Name' | 'Food Category' | 'food_category'
      - price:       'Price (USD/100g)' | 'price_per_100g' | 'Price (USD)'
      - nutrients:   any column shaped '<Nutrient> (<unit>) /100g' whose <Nutrient>
                     matches one of the 23 DRI nutrients (Energy..Vit. E).

    FPED columns are NOT present in nutrient datasheets, so every FPED field and
    added-sugar is set to 0 — HEI-2020 components will therefore be 0 for uploaded
    data (FPED cup/oz-equivalents are required to score HEI). DRI, calories, cost,
    and Healthy Diet Basket all work fully.
    """
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    raw = pd.read_excel(src)
    raw.columns = [str(c).strip() for c in raw.columns]

    # Map "<Nutrient> (unit) /100g" -> base nutrient name (first match wins).
    nutr_col: dict[str, str] = {}
    fped_col: dict[str, str] = {}
    for c in raw.columns:
        base = c.split(" (")[0].strip()
        if base in NUTRIENT_COMP_COLS and base not in nutr_col:
            nutr_col[base] = c
        # FPED equivalents share the SAME header convention, e.g.
        # "F_TOTAL (cup eq) /100g" -> base token "F_TOTAL" (one of FPED_COLS).
        if base in FPED_COLS and base not in fped_col:
            fped_col[base] = c

    name_col = _first_present(raw, ["Item Name", "name", "Food", "Item Description"])
    if name_col is None:
        raise ValueError("Uploaded sheet has no recognizable food-name column "
                         "(expected one of: 'Item Name', 'name', 'Food').")
    grp_col = _first_present(raw, ["HDB Food Group", "hdb_group"])
    cat_col = _first_present(raw, ["Food Name", "Food Category", "food_category"])
    price_col = _first_present(raw, ["Price (USD/100g)", "price_per_100g", "Price (USD)"])

    out = pd.DataFrame()
    out["name"] = raw[name_col].astype(str).str.strip()
    # keep only real rows BEFORE de-duplicating so blanks don't grab a suffix
    keep = out["name"].notna() & (out["name"] != "nan") & (out["name"] != "")
    out = out[keep].reset_index(drop=True)
    raw = raw[keep.values].reset_index(drop=True)
    out["name"] = _dedupe_names(out["name"].tolist())

    for idc in ("wweia_id", "stop_shop_id", "usda_fdc_id", "ndb_food_code", "fped_code"):
        out[idc] = np.nan
    out["hdb_group"] = (raw[grp_col].astype(str).str.strip() if grp_col else "")
    out["food_category"] = (raw[cat_col].astype(str).str.strip() if cat_col else out["name"])

    for nutrient in NUTRIENT_COMP_COLS:
        col = nutr_col.get(nutrient)
        out[nutrient] = (pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
                         if col else 0.0)

    # FPED columns (HEI inputs). Present -> read them; absent -> 0 (HEI then
    # can't be scored meaningfully, but DRI/HDB/cost still work fully).
    for fped in FPED_COLS:
        col = fped_col.get(fped)
        out[fped] = (pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
                     if col else 0.0)
    # Added sugars in grams derive from the tsp-equivalent column if supplied.
    out["added_sugar_g"] = out["ADD_SUGARS_tsp"] * ADDED_SUGAR_G_PER_TSP

    out["price_per_100g"] = (pd.to_numeric(raw[price_col], errors="coerce").fillna(0.0)
                             if price_col else 0.0)

    out = out.sort_values("name").reset_index(drop=True)
    prices = out[["name", "price_per_100g"]].copy()
    catalog = out.drop(columns=["price_per_100g"])
    return catalog, prices


def load_dri(sex: str = "female") -> dict[str, tuple[float | None, float | None]]:
    """Load Female or Male DRI (lower, upper) per nutrient from `1.NutrientRequirements`.

    Female: rows 3 (lower) / 4 (upper). Male: rows 6 / 7.
    """
    raw = pd.read_excel(SOURCE_XLSX, sheet_name="1.NutrientRequirements", header=None)
    if sex.lower() == "female":
        lo_row, up_row = raw.iloc[3], raw.iloc[4]
    elif sex.lower() == "male":
        lo_row, up_row = raw.iloc[6], raw.iloc[7]
    else:
        raise ValueError(f"sex must be 'female' or 'male', got {sex!r}")

    bounds: dict[str, tuple[float | None, float | None]] = {}
    for nutrient, col in DRI_HEADER.items():
        bounds[nutrient] = (_num(lo_row[col]), _num(up_row[col]))
    return bounds


def load_hei_bounds() -> dict[str, dict]:
    """Parse the textual HEI-2020 bounds from `2. HEI Scoring Reference`.

    Returns dict[component_name] = {lower, upper, max_score, direction}.
    The component_name keys match the first element of HEI_COMPONENTS tuples.
    Values for adequacy components: lower=0, upper=numeric (e.g. 0.8 cup eq).
    Values for moderation components: lower=numeric (best-score threshold),
                                       upper=numeric (zero-score threshold).
    """
    raw = pd.read_excel(SOURCE_XLSX, sheet_name="2. HEI Scoring Reference", header=None)
    lower_row = raw.iloc[6]   # bounds at lower-intake threshold
    upper_row = raw.iloc[7]   # bounds at upper-intake threshold
    # The sheet's row 9 = "Minimum Score" and row 10 = "Maximum Score" are labeled by
    # INTAKE-BOUND POSITION (score at lower-intake / score at upper-intake), not by
    # min/max of the score itself. For adequacy the higher value lives in row 10; for
    # moderation in row 9. Take the max to recover the actual max possible score.
    score_at_lo_row = raw.iloc[9]
    score_at_hi_row = raw.iloc[10]

    # Map sheet column index -> our component name. Sheet col 5..13 are adequacy,
    # 14..17 are moderation. Note last col (17) is 'Fatty Acids' but it's actually
    # Sat Fat per its bounds (lower=8% upper=16%). Header is mislabeled in the sheet.
    # We use the bounds from each column to populate our HEI_COMPONENTS in order.
    sheet_cols = {
        "Total Fruits": 5,
        "Whole Fruits": 6,
        "Total Vegetables": 7,
        "Greens & Beans": 8,
        "Whole Grains": 9,
        "Dairy": 10,
        "Total Protein": 11,
        "Seafood & Plant Protein": 12,
        "Fatty Acids": 13,
        "Refined Grains": 14,
        "Sodium": 15,
        "Added Sugars": 16,
        "Sat Fat": 17,
    }

    bounds: dict[str, dict] = {}
    for comp_name, _max_default, direction in HEI_COMPONENTS:
        col = sheet_cols[comp_name]
        lo = _num(lower_row[col])
        up = _num(upper_row[col])
        s_lo = _num(score_at_lo_row[col]) or 0.0
        s_hi = _num(score_at_hi_row[col]) or 0.0
        mx = max(s_lo, s_hi)
        bounds[comp_name] = {
            "lower": lo if lo is not None else 0.0,
            "upper": up if up is not None else 0.0,
            "max_score": mx if mx is not None else float(_max_default),
            "direction": direction,
        }
    return bounds


# ============================================================================
#  Healthy Diet Basket (HDB) guideline config
# ============================================================================
# Definition supplied by the user for a 30-year-old female (2330 kcal/day need).
# The six food-group energy targets below sum to exactly 2330 kcal. We keep the
# numbers as-is and normalize the IDEAL proportions over their own total (2330),
# which reproduces the user's stated percentages (50/5/7/13/13/13) and makes the
# ideal pie sum to 100%.

HDB_GROUPS = [
    "Starchy staples", "Vegetables", "Fruits",
    "Animal source foods", "Legumes, nuts and seeds", "Oils and fats",
]

HDB_TARGET_KCAL = {
    "Starchy staples": 1160,
    "Vegetables": 110,
    "Fruits": 160,
    "Animal source foods": 300,
    "Legumes, nuts and seeds": 300,
    "Oils and fats": 300,
}
HDB_KCAL_TOTAL = sum(HDB_TARGET_KCAL.values())  # 2330 = 1160+110+160+300+300+300

# Diversity rule: ANY chosen food (any amount above 0 g) counts as one of its
# group's distinct items — there is no minimum weight. (This was 5 g before.)
# Scoring counts every food with grams > 0; the optimizer, which cannot express
# "strictly above 0", uses optimize.OPT_SELECTED_MIN_G instead.
HDB_MIN_ITEM_GRAMS = 0.0

# Minimum number of distinct food items required per group (diet diversity).
HDB_MIN_ITEMS = {
    "Starchy staples": 2,
    "Vegetables": 3,
    "Fruits": 2,
    "Animal source foods": 2,
    "Legumes, nuts and seeds": 1,
    "Oils and fats": 1,
}

# Map a food's HDB group label to one of the 6 canonical groups. The built-in
# catalog takes its labels from FoodGroupConvert.xlsx ("HDB Category"), which
# already uses the canonical names; a few legacy spellings are still accepted for
# uploaded datasheets. Any other label — notably "Exclude" — means the food is
# outside the six groups.
_HDB_RAW_TO_GROUP = {
    "Starchy staples": "Starchy staples",
    "Vegetables": "Vegetables",
    "Fruit": "Fruits",
    "Fruits": "Fruits",
    "Animal source foods": "Animal source foods",
    "Beans, nuts & seeds": "Legumes, nuts and seeds",
    "Legumes, nuts and seeds": "Legumes, nuts and seeds",
    "Legumes nuts and seeds": "Legumes, nuts and seeds",
    "Oils and fats": "Oils and fats",
}


def hdb_group_of(raw_group: str, food_name: str) -> str | None:
    """Return the canonical HDB group for a food, or None if it is excluded.

    The group comes only from the data label; nothing is inferred from the food's
    name. (The old rule that re-routed "Beverage" items — juices to Fruits, plant
    milks to Legumes — is gone: FoodGroupConvert.xlsx now assigns every food
    explicitly, e.g. the two juices are "Exclude" and Oat Milk is a Starchy staple.)
    `food_name` is kept for call-site compatibility.
    """
    return _HDB_RAW_TO_GROUP.get((raw_group or "").strip())


def hdb_type_of(food_name: str, raw_subgroup: str) -> str:
    """The food TYPE used for HDB diversity (FoodGroupConvert "HDB Subgroup").

    Foods that share a subgroup are the same type: two wheat foods, or two dairy
    foods, together count as ONE toward a group's diversity. A food with no
    subgroup is its own type. Matching ignores case and surrounding spaces.
    The type only matters for HDB diversity; everything else still treats each
    food separately.
    """
    sub = (raw_subgroup or "").strip().lower()
    if sub and sub != "nan":
        return f"subgroup:{sub}"
    return f"food:{food_name}"


def hdb_type_label(type_key: str) -> str:
    """Human-readable label for a type key: the subgroup name, or 'own type'."""
    return type_key.split(":", 1)[1] if type_key.startswith("subgroup:") else "—"


def hdb_ideal_shares() -> dict[str, float]:
    """Ideal energy share per group, normalized to 1.0 over the 2330-kcal total."""
    return {grp: HDB_TARGET_KCAL[grp] / HDB_KCAL_TOTAL for grp in HDB_GROUPS}

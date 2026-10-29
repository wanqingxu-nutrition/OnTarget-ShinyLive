# OnTarget — Diet Optimizer

**Nutritional Quality and Cost Calculator for Hypothetical Diets in Boston**

Open the calculator: **https://wanqingxu-nutrition.github.io/OnTarget-ShinyLive/**

Build a daily diet from 93 foods sold at Stop & Shop in Boston, or let the
calculator find the lowest-cost diet that meets the health criteria you choose.
It reports cost per day, calories, and how the diet scores against three
standards:

- **DRI** — Dietary Reference Intakes, 22 nutrients
- **HEI-2020** — Healthy Eating Index, 13 components
- **HDB** — Healthy Diet Basket, 6 food groups

Built by **Wanqing Xu** for teaching and research, under the supervision of
**William A. Masters**, Friedman School of Nutrition Science and Policy, Tufts
University. Contact: wanqing.xu@tufts.edu or william.masters@tufts.edu

## How it runs

The whole calculator runs **inside your browser** — Python compiled to
WebAssembly via [Shinylive](https://shiny.posit.co/py/docs/shinylive.html).
There is no server and nothing you enter leaves your computer. The first visit
takes 30–90 seconds while the page starts Python; after that it is fast.

## Repository layout

```
app/
  app.py            Shiny user interface and server logic
  data.py           workbook loading, food catalog, guideline definitions
  scoring.py        HEI / DRI / HDB scoring
  optimize.py       the linear and mixed-integer cost-minimisation model
  viz.py            charts
  requirements.txt  extra packages installed into the browser runtime
  data/
    New_Nutrient_Data.xlsx            prices, nutrient composition, FPED
                                      equivalents, DRI bounds, HEI standards
    WXSep11_FoodGroupConvert+WM.xlsx  food-group labels (HEI and HDB categories)
```

Pushing to `main` rebuilds the site and republishes it
(`.github/workflows/deploy.yml`).

## Data

Food prices were collected at Stop & Shop in Boston in December 2025. Nutrient
composition is matched to USDA FoodData Central, with WWEIA, FDC and FPED
identifiers retained in the workbook. The workbook contains only the five sheets
the calculator reads.

## Running or rebuilding it yourself

As an ordinary Shiny app (needs `shiny`, `plotly`, `openpyxl`, `pandas`,
`scipy`):

```bash
shiny run --launch-browser app/app.py
```

As the browser-only build:

```bash
shinylive export app site
python -m http.server 8008 --directory site
```

Then open http://127.0.0.1:8008. A real HTTP server is required — opening
`site/index.html` from the file system does not work, because Shinylive
registers a service worker.

## Embedding it in a web page

```html
<iframe src="https://wanqingxu-nutrition.github.io/OnTarget-ShinyLive/"
        title="OnTarget — Diet Optimizer"
        style="width:100%;height:1100px;border:1px solid #ddd;border-radius:6px;"
        loading="lazy"></iframe>
```

"""
Generate a simulated inventory dataset for an aerospace composite-parts
manufacturing site.

The data is SYNTHETIC. It is built to behave like real production-consumable
data (mixed smooth / erratic / intermittent demand, variable supplier lead
times, a legacy "rule of thumb" Kanban setting) so the analysis methods can be
demonstrated end-to-end. No company data is used.

Outputs (in ../data):
    parts_master.csv   - one row per part number (120 parts)
    weekly_demand.csv  - 104 weeks of consumption per part
    receipts_log.csv   - historical replenishment orders with actual lead times
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_PARTS = 120
N_WEEKS = 104
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- categories
CATEGORIES = {
    # name: (share, unit cost range AUD, mean weekly demand range, lead time wks range)
    "Carbon fibre prepreg":     (0.12, (180, 900), (8, 40), (6, 12)),
    "Resin & adhesives":        (0.12, (60, 400), (10, 60), (4, 9)),
    "Fasteners":                (0.22, (0.5, 12), (200, 2500), (3, 8)),
    "Honeycomb core":           (0.08, (120, 600), (4, 20), (6, 11)),
    "Vacuum bagging consumables": (0.18, (2, 35), (80, 900), (2, 5)),
    "Sealants & paint":         (0.10, (25, 160), (15, 90), (3, 7)),
    "Tooling & cutting":        (0.10, (15, 220), (5, 60), (2, 6)),
    "PPE & shop consumables":   (0.08, (1, 20), (50, 600), (1, 4)),
}
SUPPLIERS = [f"SUP-{i:03d}" for i in range(1, 41)]

cats = rng.choice(list(CATEGORIES), size=N_PARTS,
                  p=[v[0] for v in CATEGORIES.values()])

parts = []
for i, cat in enumerate(cats, start=1):
    _, cost_r, dem_r, lt_r = CATEGORIES[cat]
    unit_cost = round(float(np.exp(rng.uniform(np.log(cost_r[0]), np.log(cost_r[1])))), 2)
    mean_dem = float(np.exp(rng.uniform(np.log(dem_r[0]), np.log(dem_r[1]))))
    # demand pattern mix: smooth / erratic / intermittent
    pattern = rng.choice(["smooth", "erratic", "intermittent"], p=[0.5, 0.3, 0.2])
    lt_mean = float(rng.uniform(*lt_r))
    lt_sd = lt_mean * float(rng.uniform(0.08, 0.35))
    container = int(max(1, round(mean_dem * rng.uniform(0.5, 1.5), -1 if mean_dem > 50 else 0)))
    parts.append(dict(
        part_number=f"PN-{10000 + i}",
        category=cat,
        supplier_id=rng.choice(SUPPLIERS),
        unit_cost_aud=unit_cost,
        demand_pattern=pattern,
        _mean_demand=mean_dem,
        lead_time_mean_wks=round(lt_mean, 2),
        lead_time_sd_wks=round(lt_sd, 2),
        kanban_container_qty=container,
    ))
parts = pd.DataFrame(parts)

# ---------------------------------------------------------------- demand
# Build rate ramps up ~15% over two years (production rate increase),
# with a mild quarterly cycle.
weeks = np.arange(1, N_WEEKS + 1)
rate = 1 + 0.15 * weeks / N_WEEKS + 0.05 * np.sin(2 * np.pi * weeks / 13)

rows = []
for _, p in parts.iterrows():
    mu = p._mean_demand * rate
    if p.demand_pattern == "smooth":
        d = rng.normal(mu, 0.18 * mu)
    elif p.demand_pattern == "erratic":
        d = rng.gamma(shape=1.6, scale=mu / 1.6)
    else:  # intermittent: demand occurs ~30% of weeks, in lumps
        occurs = rng.random(N_WEEKS) < 0.3
        d = occurs * rng.gamma(2.0, (mu / 0.3) / 2.0)
    d = np.clip(np.round(d), 0, None).astype(int)
    rows.append(pd.DataFrame({"week": weeks, "part_number": p.part_number, "qty": d}))
demand = pd.concat(rows, ignore_index=True)

# ---------------------------------------------------------------- legacy Kanban
# The site's legacy rule, set part-by-part by different buyers when each part
# was introduced and rarely reviewed since:
#   reorder point = (quoted lead time + a buffer of 1-6 weeks) x first-year
#                   average weekly demand
# The buffer reflects buyer judgement, not data, so it ignores demand
# variability, lead-time variability and the production rate increase.
yr1 = demand[demand.week <= 52].groupby("part_number").qty.mean()
quoted_lt = parts.lead_time_mean_wks.round()
buffer_wks = rng.choice([1, 2, 3, 4, 6], size=N_PARTS, p=[0.2, 0.25, 0.2, 0.2, 0.15])
parts["legacy_reorder_point"] = ((quoted_lt + buffer_wks) * parts.part_number.map(yr1)).round().astype(int)

# ---------------------------------------------------------------- receipts
rec = []
for _, p in parts.iterrows():
    n_orders = int(rng.integers(8, 26))
    order_weeks = np.sort(rng.choice(np.arange(1, N_WEEKS - 12), n_orders, replace=False))
    for w in order_weeks:
        lt = max(1.0, rng.normal(p.lead_time_mean_wks, p.lead_time_sd_wks))
        rec.append(dict(part_number=p.part_number, supplier_id=p.supplier_id,
                        order_week=int(w), actual_lead_time_wks=round(lt, 1)))
receipts = pd.DataFrame(rec)

parts.drop(columns="_mean_demand").to_csv(OUT / "parts_master.csv", index=False)
demand.to_csv(OUT / "weekly_demand.csv", index=False)
receipts.to_csv(OUT / "receipts_log.csv", index=False)
print(f"parts={len(parts)}  demand_rows={len(demand)}  receipts={len(receipts)}")

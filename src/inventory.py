"""Reusable inventory-analytics functions used by the notebook."""
import numpy as np
import pandas as pd
from scipy.stats import norm


# ------------------------------------------------------------ classification
def abc_class(values: pd.Series, a=0.80, b=0.95) -> pd.Series:
    """ABC by cumulative share of annual consumption value (80/15/5)."""
    s = values.sort_values(ascending=False)
    cum = s.cumsum() / s.sum()
    cls = np.where(cum <= a, "A", np.where(cum <= b, "B", "C"))
    # the item that crosses the threshold belongs to the higher class
    out = pd.Series(cls, index=s.index)
    return out.reindex(values.index)


def xyz_class(cv: pd.Series, x=0.5, y=1.0) -> pd.Series:
    """XYZ by coefficient of variation of weekly demand."""
    return pd.Series(np.where(cv <= x, "X", np.where(cv <= y, "Y", "Z")), index=cv.index)


def sbc_class(adi: float, cv2: float) -> str:
    """Syntetos-Boylan-Croston demand classification."""
    if adi <= 1.32:
        return "Smooth" if cv2 <= 0.49 else "Erratic"
    return "Intermittent" if cv2 <= 0.49 else "Lumpy"


# ------------------------------------------------------------ forecasting
def f_moving_average(y, n):
    return float(np.mean(y[-n:]))


def f_ses(y, alpha):
    level = y[0]
    for v in y[1:]:
        level = alpha * v + (1 - alpha) * level
    return float(level)


def f_croston(y, alpha=0.1):
    """Croston (SBA-corrected) for intermittent demand."""
    nz = np.flatnonzero(y)
    if len(nz) < 2:
        return float(np.mean(y))
    z, p = y[nz[0]], max(nz[0] + 1, 1)
    q = 1
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z = alpha * y[t] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
    return float((1 - alpha / 2) * z / p)


METHODS = {
    "MA-4": lambda y: f_moving_average(y, 4),
    "MA-13": lambda y: f_moving_average(y, 13),
    "MA-26": lambda y: f_moving_average(y, 26),
    "SES a=0.1": lambda y: f_ses(y, 0.1),
    "SES a=0.3": lambda y: f_ses(y, 0.3),
    "Croston-SBA": lambda y: f_croston(y, 0.1),
}


def rolling_origin_errors(y, start, method):
    """One-step-ahead errors from `start` to end of y (rolling origin)."""
    f = METHODS[method]
    preds = np.array([f(y[:t]) for t in range(start, len(y))])
    return y[start:] - preds


# ------------------------------------------------------------ policy
def safety_stock(z, d, sd_d, lt, sd_lt):
    """SS = z * sqrt(LT*sigma_d^2 + d^2*sigma_LT^2)  (demand & lead-time uncertainty)."""
    return z * np.sqrt(lt * sd_d ** 2 + (d ** 2) * sd_lt ** 2)


def z_for(service_level):
    return norm.ppf(service_level)


# ------------------------------------------------------------ simulation
def simulate_kanban(demand, rop, container, lt_mean, lt_sd, rng,
                    measure_from=0, start_on_hand=None):
    """
    Weekly multi-card Kanban simulation with stochastic lead times.

    Each week: receive arrivals -> satisfy demand (unmet demand is backordered,
    i.e. the production line waits) -> if inventory position <= ROP, release
    Kanban cards (one container each) until position > ROP.

    Returns dict of KPIs measured from index `measure_from`.
    """
    T = len(demand)
    on_hand = rop + container if start_on_hand is None else start_on_hand
    backorder = 0
    pipeline = np.zeros(T + 60)
    oh_trace, served, dem_tot, so_weeks, orders = [], 0, 0, 0, 0
    for t in range(T):
        on_hand += pipeline[t]
        need = demand[t] + backorder
        ship = min(on_hand, need)
        on_hand -= ship
        backorder = need - ship
        if t >= measure_from:
            # fill rate: share of this week's demand served on time from stock
            served += min(demand[t], max(0, ship - (need - demand[t])))
            dem_tot += demand[t]
            so_weeks += int(backorder > 0)
            oh_trace.append(on_hand)
        position = on_hand + pipeline[t + 1:].sum() - backorder
        while position <= rop:
            lt = max(1, int(round(rng.normal(lt_mean, lt_sd))))
            pipeline[t + lt] += container
            position += container
            orders += int(t >= measure_from)
    return dict(
        fill_rate=served / dem_tot if dem_tot else 1.0,
        stockout_weeks=so_weeks,
        avg_on_hand=float(np.mean(oh_trace)),
        orders=orders,
        trace=np.array(oh_trace),
    )


def z_for_fill_rate(fill_rate, container_qty, sigma_lt_demand):
    """
    Safety factor z that achieves a target *fill rate* (beta service level).

    Uses the standard normal loss function G(z) = phi(z) - z*(1-Phi(z)):
        expected units short per replenishment cycle = sigma_L * G(z)
        fill rate = 1 - sigma_L * G(z) / Q
    so we solve G(z) = (1 - fill_rate) * Q / sigma_L for z.
    Parts with a large container relative to their demand uncertainty need
    little safety stock; volatile, long lead-time parts need more.
    z is floored at 0 (never plan to run below the expected lead-time demand).
    """
    from scipy.optimize import brentq
    target = (1 - fill_rate) * container_qty / max(sigma_lt_demand, 1e-9)
    G = lambda z: norm.pdf(z) - z * (1 - norm.cdf(z))
    if target >= G(0):
        return 0.0
    return brentq(lambda z: G(z) - target, 0, 6)

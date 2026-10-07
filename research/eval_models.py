"""Compare gold forecasting models on real XAU/USD 5-minute data (Twelve Data).

Every forecast uses only data before its origin. Closed-market bars (weekends, daily break) are removed first,
because Twelve Data fills them with near-constant prices. Horizons are counted in trading hours.

Models: naive, daily pattern (current app method), shrunk pattern, seasonal naive, drift, ETS (flat / trend /
damped), AR(p) on returns, ARIMA(1,1,1), ensemble. Each model's settings and look-back window (<= 6 months)
are chosen on a validation period before the test window; test days are never used for selection.

Usage: .venv/bin/python eval_models.py        -> writes results.md and compare.png
"""
import math, os, warnings
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
LOOKBACKS = {"1w": 7, "2w": 14, "1m": 30, "3m": 91, "6m": 182}       # calendar days, max 6 months
H = 24                                                                 # horizon in trading hours
MIN_SLOTS = 72                                                         # a day needs >= 6 h of trading bars


# ---------------------------------------------------------------- data
def market_open(ts_utc):
    t = ts_utc.astimezone(ET)
    w, h = t.weekday(), t.hour                                         # Mon=0 .. Sun=6
    if w == 5: return False
    if w == 6: return h >= 18
    if w == 4: return h < 17
    return h != 17


def load():
    df = pd.read_csv(os.path.join(HERE, "data", "gold_5min.csv"), parse_dates=["datetime"])
    df["utc"] = df.datetime - pd.Timedelta(hours=7)
    df["open_mkt"] = [market_open(t.replace(tzinfo=UTC).to_pydatetime()) for t in df.utc]
    removed = (~df.open_mkt).sum()
    df = df[df.open_mkt].reset_index(drop=True)
    df["date"] = df.datetime.dt.date
    df["slot"] = df.datetime.dt.hour * 12 + df.datetime.dt.minute // 5
    return df, removed


# ---------------------------------------------------------------- helpers
def theil_sen(ys):
    s = [(ys[j] - ys[i]) / (j - i) for i in range(len(ys)) for j in range(i + 1, len(ys))]
    return float(np.median(s)) if s else 0.0


class Data:
    def __init__(self, df):
        self.df = df
        self.c = df.close.to_numpy()
        self.t = df.datetime.to_numpy()
        # hourly series: close of the last 5-min bar in each trading hour
        hr = df.groupby(df.datetime.dt.floor("h")).agg(close=("close", "last"), idx=("close", lambda s: s.index[-1]))
        self.hours = hr.index.to_pydatetime()
        self.y = hr.close.to_numpy()
        self.y_idx = hr.idx.to_numpy()                                 # index of that bar in the 5-min frame
        self.log_y = np.log(self.y)
        # per-day 5-min close matrix for the pattern models
        days = {}
        for d, g in df.groupby("date"):
            arr = np.full(288, np.nan); arr[g.slot.to_numpy()] = g.close.to_numpy()
            if np.isfinite(arr).sum() >= MIN_SLOTS:
                days[d] = arr
        self.days = days
        self.day_list = sorted(days)
        self.med = {d: np.nanmedian(a) for d, a in days.items()}
        self._ref = {}

    def ref_stats(self, before, L):
        """Median relative price per 5-min slot over complete trading days in [before - L, before)."""
        key = (before, L)
        if key not in self._ref:
            ds = [d for d in self.day_list if before - timedelta(days=L) <= d < before]
            if len(ds) < 3:
                self._ref[key] = None
            else:
                rel = np.vstack([(self.days[d] / self.med[d] - 1) * 100 for d in ds])
                with np.errstate(all="ignore"):
                    m = np.nanmedian(rel, axis=0)
                # fill empty slots by carrying neighbours
                s = pd.Series(m).ffill().bfill().to_numpy()
                self._ref[key] = (s, ds)
        return self._ref[key]


# ---------------------------------------------------------------- models: each returns forecasts for h = 1..H
def f_naive(D, j, cfg):
    return np.full(H, D.y[j])


def f_pattern(D, j, cfg):
    """Current app method: anchor + damped Theil-Sen trend + typical intraday shape."""
    t0 = D.hours[j]; d0 = t0.date()
    rs = D.ref_stats(d0, LOOKBACKS[cfg["L"]])
    if rs is None: return None
    rel, ds = rs
    i5 = D.y_idx[j]; s_now = D.df.slot[i5]
    anchor = D.y[j] / (1 + rel[s_now] / 100)
    prior = [d for d in D.day_list if d < d0][-10:]
    slope = theil_sen([D.med[d] for d in prior]) * 0.5
    out, dates = [], []
    for h in range(1, H + 1):
        if j + h >= len(D.hours): out.append(np.nan); continue
        tt = D.hours[j + h] + timedelta(minutes=55)                      # end of target hour
        dd = tt.date()
        if dd not in dates: dates.append(dd)
        k = dates.index(dd) + (0 if dates[0] == d0 else 1)
        s = tt.hour * 12 + tt.minute // 5
        out.append((anchor + slope * k) * (1 + rel[s] / 100))
    return np.array(out)


def f_shrunk(D, j, cfg):
    """Last price moved by w x (typical shape at target - typical shape now)."""
    rs = D.ref_stats(D.hours[j].date(), LOOKBACKS[cfg["L"]])
    if rs is None: return None
    rel, _ = rs
    s_now = D.df.slot[D.y_idx[j]]
    out = []
    for h in range(1, H + 1):
        if j + h >= len(D.hours): out.append(np.nan); continue
        tt = D.hours[j + h] + timedelta(minutes=55)
        s = tt.hour * 12 + tt.minute // 5
        out.append(D.y[j] * (1 + cfg["w"] * (rel[s] - rel[s_now]) / 100))
    return np.array(out)


def f_seasonal(D, j, cfg):
    """Last price x yesterday's move over the same trading hours."""
    if j < H: return None
    base = D.y[j - H]
    return np.array([D.y[j] * D.y[j - H + h] / base for h in range(1, H + 1)])


def f_drift(D, j, cfg):
    n = cfg["N"] * 23
    if j < n: return None
    mu = (D.log_y[j] - D.log_y[j - n]) / n
    return D.y[j] * np.exp(mu * np.arange(1, H + 1))


# ETS and ARIMA/AR: parameters fitted once per test day on the look-back window, then run forward hour by hour.
def ets_fit(y, kind):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    if kind == "ses":
        m = ExponentialSmoothing(y, trend=None).fit(optimized=True)
        return dict(a=m.params["smoothing_level"], b=0.0, phi=0.0)
    m = ExponentialSmoothing(y, trend="add", damped_trend=(kind == "damped")).fit(optimized=True)
    return dict(a=m.params["smoothing_level"], b=m.params["smoothing_trend"],
                phi=m.params.get("damping_trend", 1.0) if kind == "damped" else 1.0)


def phi_sum(phi, h):
    return h if phi == 1 else (0.0 if phi == 0 else phi * (1 - phi ** h) / (1 - phi))


class DayFitted:
    """Caches per (model, config, test-day) fitted parameters + filtered states for every hour of that day."""
    def __init__(self, D):
        self.D, self.cache = D, {}

    def hours_window(self, d, L):
        start = datetime.combine(d, datetime.min.time()) - timedelta(days=L)
        end = datetime.combine(d, datetime.min.time())
        lo = np.searchsorted(self.D.hours, start); hi = np.searchsorted(self.D.hours, end)
        return lo, hi

    def ets_states(self, d, L, kind):
        key = ("ets", d, L, kind)
        if key in self.cache: return self.cache[key]
        lo, hi = self.hours_window(d, L)
        if hi - lo < 50: self.cache[key] = None; return None
        p = ets_fit(self.D.y[lo:hi], kind)
        a, b, phi = p["a"], p["b"], p["phi"]
        y = self.D.y; Lv, T = y[lo], (0.0 if kind == "ses" else y[lo + 1] - y[lo])
        states = {}
        end = min(len(y), np.searchsorted(self.D.hours, datetime.combine(d + timedelta(days=1), datetime.min.time())))
        for i in range(lo + 1, end):
            Lp = Lv; Lv = a * y[i] + (1 - a) * (Lp + phi * T); T = b * (Lv - Lp) + (1 - b) * phi * T
            if i >= hi: states[i] = (Lv, T)
        self.cache[key] = (p, states)
        return self.cache[key]

    def arima_states(self, d, L, order):
        key = ("arima", d, L, order)
        if key in self.cache: return self.cache[key]
        lo, hi = self.hours_window(d, L)
        if hi - lo < 80: self.cache[key] = None; return None
        r = np.diff(self.D.log_y[lo:hi])
        p, q = order
        if q == 0:                                                     # AR(p) by least squares
            X = np.column_stack([np.ones(len(r) - p)] + [r[p - k - 1:len(r) - k - 1] for k in range(p)])
            beta = np.linalg.lstsq(X, r[p:], rcond=None)[0]
            par = dict(c=beta[0], phi=list(beta[1:]), theta=0.0)
        else:                                                          # ARIMA(1,1,1) via statsmodels
            from statsmodels.tsa.arima.model import ARIMA
            m = ARIMA(r, order=(1, 0, 1), trend="c").fit()
            mu = m.params[0]; ar = m.params[1]; ma = m.params[2]
            par = dict(c=mu * (1 - ar), phi=[ar], theta=ma)
        self.cache[key] = (par, lo)
        return self.cache[key]


def f_ets(D, j, cfg, F):
    st = F.ets_states(D.hours[j].date(), LOOKBACKS[cfg["L"]], cfg["kind"])
    if st is None or j not in st[1]: return None
    p, states = st; Lv, T = states[j]
    return np.array([Lv + phi_sum(p["phi"], h) * T for h in range(1, H + 1)])


def f_arima(D, j, cfg, F):
    st = F.arima_states(D.hours[j].date(), LOOKBACKS[cfg["L"]], cfg["order"])
    if st is None: return None
    par, lo = st
    r = np.diff(D.log_y[max(lo, j - 400):j + 1])
    p = len(par["phi"])
    if len(r) < p + 2: return None
    # residuals for the MA term (ARIMA only), filtered over the recent history
    eps = 0.0
    if par["theta"]:
        for k in range(p, len(r)):
            pred = par["c"] + sum(par["phi"][m] * r[k - m - 1] for m in range(p)) + par["theta"] * eps
            eps = r[k] - pred
    hist = list(r[-p:]); out = []; level = D.log_y[j]
    for h in range(1, H + 1):
        nxt = par["c"] + sum(par["phi"][m] * hist[-m - 1] for m in range(p)) + (par["theta"] * eps if h == 1 else 0.0)
        hist.append(nxt); level += nxt; out.append(math.exp(level))
    return np.array(out)


# ---------------------------------------------------------------- evaluation
def origins_between(D, start, end):
    return [j for j, t in enumerate(D.hours) if start <= t < end and j + 1 < len(D.hours)]


def score(D, model, cfg, js, F):
    errs = {h: [] for h in range(1, H + 1)}; dirs = {4: [], 24: []}
    for j in js:
        fc = model(D, j, cfg, F) if model in (f_ets, f_arima) else model(D, j, cfg)
        if fc is None: continue
        for h in range(1, H + 1):
            if j + h >= len(D.y) or not np.isfinite(fc[h - 1]): continue
            e = fc[h - 1] - D.y[j + h]; errs[h].append(e)
            if h in dirs and fc[h - 1] != D.y[j] and D.y[j + h] != D.y[j]:
                dirs[h].append((fc[h - 1] > D.y[j]) == (D.y[j + h] > D.y[j]))
    allv = np.concatenate([np.array(v) for v in errs.values() if v]) if any(errs.values()) else np.array([])
    if not len(allv): return None
    by_h = {h: float(np.mean(np.abs(errs[h]))) for h in (1, 4, 12, 24) if errs[h]}
    return dict(mae=float(np.mean(np.abs(allv))), rmse=float(np.sqrt(np.mean(allv ** 2))), by_h=by_h, n=len(js),
                dir4=(np.mean(dirs[4]) if dirs[4] else None), dir24=(np.mean(dirs[24]) if dirs[24] else None))


def candidates():
    for L in LOOKBACKS: yield "Daily pattern (current)", f_pattern, dict(L=L)
    for L in LOOKBACKS:
        for w in (0.25, 0.5, 0.75, 1.0): yield "Shrunk pattern", f_shrunk, dict(L=L, w=w)
    yield "Seasonal naive", f_seasonal, {}
    for N in (5, 10, 20): yield "Drift", f_drift, dict(N=N)
    for L in LOOKBACKS:
        for kind in ("ses", "linear", "damped"): yield "ETS", f_ets, dict(L=L, kind=kind)
    for L in LOOKBACKS:
        for order in ((1, 0), (2, 0), (3, 0)): yield "AR on returns", f_arima, dict(L=L, order=order)
    for L in ("1m", "3m", "6m"): yield "ARIMA(1,1,1)", f_arima, dict(L=L, order=(1, 1))


def cfg_label(cfg):
    if not cfg: return "—"
    parts = []
    if "L" in cfg: parts.append(f"look-back {cfg['L']}")
    if "w" in cfg: parts.append(f"w={cfg['w']}")
    if "kind" in cfg: parts.append({"ses": "flat", "linear": "trend", "damped": "damped"}[cfg["kind"]])
    if "order" in cfg and cfg["order"][1] == 0: parts.append(f"AR({cfg['order'][0]})")
    if "N" in cfg: parts.append(f"{cfg['N']} days")
    return ", ".join(parts)


def run_window(D, F, val, test, title):
    """Select each model's best config on `val` origins (by MAE), then score it on `test` origins."""
    best = {}
    for name, fn, cfg in candidates():
        s = score(D, fn, cfg, val, F)
        if s and (name not in best or s["mae"] < best[name][2]["mae"]): best[name] = (fn, cfg, s)
    rows, fcs = [], {}
    base = score(D, f_naive, {}, test, F)
    rows.append(("No change (naive)", "—", base))
    for name, (fn, cfg, _) in best.items():
        rows.append((name, cfg_label(cfg), score(D, fn, cfg, test, F)))
    # Ensemble: median of naive, shrunk, ETS, AR (each with its validated config)
    parts = [(f_naive, {})] + [(best[n][0], best[n][1]) for n in ("Shrunk pattern", "ETS", "AR on returns") if n in best]
    def f_ens(D_, j, cfg_):
        fs = [(fn(D_, j, c, F) if fn in (f_ets, f_arima) else fn(D_, j, c)) for fn, c in parts]
        fs = [f for f in fs if f is not None]
        return np.nanmedian(np.vstack(fs), axis=0) if fs else None
    rows.append(("Ensemble (median)", "naive + shrunk + ETS + AR", score(D, f_ens, {}, test, F)))
    rows = [r for r in rows if r[2]]
    for r in rows: r[2]["skill"] = 1 - r[2]["mae"] / base["mae"]
    rows.sort(key=lambda r: r[2]["mae"])
    best_fns = {n: (best[n][0], best[n][1]) for n in best}; best_fns["Ensemble (median)"] = (f_ens, {})
    return title, rows, best_fns


def table(title, rows, price):
    out = [f"### {title}", "",
           "| Rank | Model | Settings (chosen on validation) | MAE $ | MAE % | RMSE $ | MAE 1h / 4h / 12h / 24h $ | Direction 4h / 24h | vs no change |",
           "|---|---|---|---|---|---|---|---|---|"]
    for i, (name, cfg, s) in enumerate(rows, 1):
        bh = " / ".join(f"{s['by_h'].get(h, float('nan')):.1f}" for h in (1, 4, 12, 24))
        d4 = f"{s['dir4']*100:.0f}%" if s["dir4"] is not None else "—"
        d24 = f"{s['dir24']*100:.0f}%" if s["dir24"] is not None else "—"
        out.append(f"| {i} | {name} | {cfg} | {s['mae']:.2f} | {s['mae']/price*100:.2f}% | {s['rmse']:.2f} | {bh} | {d4} / {d24} | {s['skill']*100:+.1f}% |")
    return "\n".join(out)


def main():
    df, removed = load()
    D = Data(df); F = DayFitted(D)
    last = D.hours[-1]
    td = sorted({h.date() for h in D.hours})                              # trading dates
    # (a) the user's period: origins from 5 Oct 00:00; validation = 10 trading days before
    a_start = datetime(2026, 10, 5)
    a_val_start = datetime.combine(td[td.index(date(2026, 10, 5)) - 10], datetime.min.time())
    A = run_window(D, F, origins_between(D, a_val_start, a_start), origins_between(D, a_start, last), "5–7 Oct 2026 (every hourly forecast from 5 Oct 00:00)")
    # (b) robustness: last 20 trading days; validation = the 10 trading days before them
    b_start = datetime.combine(td[-21], datetime.min.time())
    b_val_start = datetime.combine(td[-31], datetime.min.time())
    B = run_window(D, F, origins_between(D, b_val_start, b_start), origins_between(D, b_start, last), "Last 20 trading days (robustness check)")
    price = float(np.median(D.y[-500:]))
    md = ["# Gold forecast model comparison (real Twelve Data XAU/USD, 5-minute bars)", "",
          f"Data: {df.datetime.iloc[0]} → {df.datetime.iloc[-1]} Thai time, {len(df):,} trading bars "
          f"({removed:,} closed-market bars removed: weekends and the daily break).",
          f"Forecasts: every trading hour, 1–{H} trading hours ahead, using only earlier data. "
          "Look-back windows tried: 1w, 2w, 1m, 3m, 6m (max 6 months). Settings chosen on the 10 trading days before each test window.", "",
          table(A[0], A[1], price), "", table(B[0], B[1], price), ""]
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(md))
    print("\n".join(md))
    plot(D, F, A, a_start, last)


def plot(D, F, A, start, last):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = A[1]; fns = A[2]
    pick = [r[0] for r in rows if r[0] not in ("No change (naive)",)][:2]
    if "Daily pattern (current)" not in pick: pick.append("Daily pattern (current)")
    js = [j for j, t in enumerate(D.hours) if start <= t <= last]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot([D.hours[j] for j in js], D.y[js], color="#222", lw=2, label="Actual (hourly close)")
    # "prediction log" style: the value each model predicted 24 trading hours earlier
    colors = ["#2f9e44", "#1c7ed6", "#9c36b5", "#e8590c"]
    for name, col in zip(["No change (naive)"] + pick, colors):
        fn, cfg = (f_naive, {}) if name == "No change (naive)" else fns[name]
        xs, ys = [], []
        for j in js:
            o = j - H
            if o < 0: continue
            fc = fn(D, o, cfg, F) if fn in (f_ets, f_arima) else fn(D, o, cfg)
            if fc is None or not np.isfinite(fc[H - 1]): continue
            xs.append(D.hours[j]); ys.append(fc[H - 1])
        ax.plot(xs, ys, color=col, lw=1.4, ls="--" if name == "Daily pattern (current)" else "-",
                label=f"{name}: predicted 24 trading h earlier")
    ax.set_title("5–7 Oct 2026: actual gold price vs the value each model predicted 24 trading hours earlier")
    ax.set_ylabel("XAU/USD"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="best")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "compare.png"), dpi=130)


if __name__ == "__main__":
    main()

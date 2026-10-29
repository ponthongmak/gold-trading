"""Gold (XAU/USD) forecast experiment: the website's models vs new methods, on real Twelve Data 5-minute bars.

Three questions, each scored only on data the method had not seen:
  1. Price path, 1-24 trading hours ahead (hourly closes, MAE)  -> site models + site Auto  vs  gradient boosting etc.
  2. Uncertainty band (80% / 95%)                               -> site error-quantile band  vs  GARCH x intraday-seasonal band
  3. Buy / sell hour of the day                                  -> site median-pattern windows  vs  recency-weighted rank windows

Every selection (model, setting, calibration) is redone at the start of each test day using earlier data only.
Writes results_new_methods.json (all chart data) next to this file.

Usage: python update_data.py (TD_KEY set) -> data/gold_5min_full.csv, then
       python experiment_new_methods.py   (needs requirements-new.txt: scikit-learn, arch)
"""
import json, math, os, sys, time, warnings
from datetime import datetime, timedelta, date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import eval_models as E                                   # the website's research code (same model definitions)

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
H = 24
VAL_DAYS = 15                                             # site: Auto chooses on the 15 trading days before
NEW_START = date(2026, 10, 8)                             # first day after the site's own research data ended
LONG_START = date(2026, 7, 1)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ data
def load():
    df = pd.read_csv(os.path.join(HERE, "data", "gold_5min_full.csv"), parse_dates=["datetime"])
    df["utc"] = df.datetime - pd.Timedelta(hours=7)
    df = df[[E.market_open(t.replace(tzinfo=E.UTC).to_pydatetime()) for t in df.utc]].reset_index(drop=True)
    df["date"] = df.datetime.dt.date
    df["slot"] = df.datetime.dt.hour * 12 + df.datetime.dt.minute // 5
    return df


df = load()
D = E.Data(df)
F = E.DayFitted(D)
n = len(D.y)
hours = D.hours
hdate = np.array([h.date() for h in hours])
cnt = pd.Series(hdate).value_counts()
TD = sorted(d for d, c in cnt.items() if c >= 6)          # trading days (>= 6 trading hours)
first_idx = {d: int(np.searchsorted(hdate, d)) for d in TD}
log(f"{len(df):,} bars, {n:,} hourly closes, {len(TD)} trading days, last {hours[-1]}")

day0 = lambda d0: next(d for d in TD if d >= d0)
long_k = TD.index(day0(LONG_START))
WARM = 21                                                  # extra days so band calibration has 20 earlier days
o_lo = first_idx[TD[long_k - WARM - VAL_DAYS - 1]]
ORIG = np.arange(o_lo, n)                                  # forecast origins (hourly index)
nO = len(ORIG)
A = np.full((nO, H), np.nan)                               # actual hourly close h hours later
TGT = np.full((nO, H), n + 999)                            # index of that target
for i, j in enumerate(ORIG):
    for h in range(1, H + 1):
        if j + h < n: A[i, h - 1] = D.y[j + h]; TGT[i, h - 1] = j + h
odate = hdate[ORIG]


# ------------------------------------------------------------------ 1. site candidates (same code as the website's research)
def site_candidates():
    out = [("No change · —", "No change", E.f_naive, {})]
    for name, fn, cfg in E.candidates():
        if fn is E.f_ets and cfg["kind"] == "linear": continue   # trend ETS: errors 3-8x naive, never selectable
        fam = {"Daily pattern (current)": "Daily pattern"}.get(name, name)
        out.append((f"{fam} · {E.cfg_label(cfg)}", fam, fn, cfg))
    return out


CACHE = os.path.join(HERE, "cache_site_fc.npz")
SITE = site_candidates()
if os.path.exists(CACHE) and np.load(CACHE)["origins"].tolist() == ORIG.tolist():
    z = np.load(CACHE); FC = {lab: z["fc"][k] for k, (lab, *_r) in enumerate(SITE)}
    log("site forecasts loaded from cache")
else:
    FC = {}
    for lab, fam, fn, cfg in SITE:
        M = np.full((nO, H), np.nan)
        for i, j in enumerate(ORIG):
            f = fn(D, j, cfg, F) if fn in (E.f_ets, E.f_arima) else fn(D, j, cfg)
            if f is not None: M[i] = f
        FC[lab] = M
        log("site", lab)
    np.savez_compressed(CACHE, origins=ORIG, fc=np.stack([FC[l] for l, *_r in SITE]))
FAMILY = {lab: fam for lab, fam, *_r in SITE}


# ------------------------------------------------------------------ 2. new point-forecast methods
ly = D.log_y
r1 = np.r_[np.nan, np.diff(ly)]
S_ly, S_r = pd.Series(ly), pd.Series(r1)
vol24 = S_r.rolling(24).std().to_numpy()
vol120 = S_r.rolling(120).std().to_numpy()
z24 = ((S_ly - S_ly.rolling(24).mean()) / vol24).to_numpy()
pos120 = ((S_ly - S_ly.rolling(120).min()) / (S_ly.rolling(120).max() - S_ly.rolling(120).min())).to_numpy()
lag = lambda k: np.r_[np.full(k, np.nan), ly[k:] - ly[:-k]]
LAGS = (1, 2, 4, 8, 24, 72, 120)
RL = {k: lag(k) for k in LAGS}
hod = np.array([h.hour for h in hours]); dow = np.array([h.weekday() for h in hours])
slot_now = D.df.slot.to_numpy()[D.y_idx]


def future_hours(last, k):
    """Next k trading hours after `last` (Thai, naive), skipping weekends and the daily break."""
    out, t = [], last
    while len(out) < k:
        t = t + timedelta(hours=1)
        if E.market_open((t - timedelta(hours=7)).replace(tzinfo=E.UTC)): out.append(t)
    return out


EXT = list(hours) + future_hours(hours[-1], 48)            # calendar of target hours, incl. the future
hod_ext = np.array([h.hour for h in EXT])


def pattern_rel(j):
    rs = D.ref_stats(hdate[j], 30)
    return None if rs is None else rs[0]


def gbm_rows(js, with_y):
    X, Y, J, Hh = [], [], [], []
    for j in js:
        if j < 125 or not np.isfinite(vol120[j]): continue
        rel = pattern_rel(j)
        v = vol120[j]
        base = [math.log(v),vol24[j] / v, z24[j], pos120[j], hod[j], dow[j]] + [RL[k][j] / (v * math.sqrt(k)) for k in LAGS]
        for h in range(1, H + 1):
            if with_y and j + h >= n: continue
            if j + h >= len(EXT): continue
            pat = 0.0 if rel is None else (rel[hod_ext[j + h] * 12 + 11] - rel[slot_now[j]]) / 100 / (v * math.sqrt(h))
            X.append(base + [h, hod_ext[j + h], pat]); J.append(j); Hh.append(h)
            if with_y: Y.append((ly[j + h] - ly[j]) / (v * math.sqrt(h)))
    return np.array(X), np.array(Y), np.array(J), np.array(Hh)


def gbm_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.04, max_iter=250, max_leaf_nodes=15,
                                         min_samples_leaf=300, l2_regularization=1.0, random_state=0)


def gbm_fit(cut):
    """Fit on origins from the 120 trading days before index `cut`, using only targets observed before `cut`."""
    k = int(np.searchsorted(np.array([first_idx[d] for d in TD]), cut))
    lo = first_idx[TD[max(0, k - 120)]]
    X, Y, J, Hh = gbm_rows(range(lo, cut - 1), True)
    keep = J + Hh < cut
    m = gbm_model(); m.fit(X[keep], Y[keep])
    return m


GB = {1.0: np.full((nO, H), np.nan), 0.5: np.full((nO, H), np.nan)}
eval_days = [d for d in TD if first_idx[d] >= o_lo]
refits = eval_days[::5]
for r, d in enumerate(refits):
    cut = first_idx[d]
    nxt = first_idx[refits[r + 1]] if r + 1 < len(refits) else n
    m = gbm_fit(cut)
    X, _, J, Hh = gbm_rows(range(cut, nxt), False)
    p = m.predict(X)
    for x, j, h in zip(p, J, Hh):
        i = j - o_lo
        for k in GB: GB[k][i, h - 1] = D.y[j] * math.exp(k * x * vol120[j] * math.sqrt(h))
log("GBM done,", len(refits), "refits")
FC["Gradient boosting (new)"] = GB[1.0]; FAMILY["Gradient boosting (new)"] = "Gradient boosting (new)"
FC["Gradient boosting ×0.5 (new)"] = GB[0.5]; FAMILY["Gradient boosting ×0.5 (new)"] = "Gradient boosting (new)"

# Smoothed last price: EMA of the last 5-minute closes (filters tick noise in the latest print)
c5 = D.c
for m_ in (3, 6):
    a = 2 / (m_ + 1); M = np.full((nO, H), np.nan)
    for i, j in enumerate(ORIG):
        idx = D.y_idx[j]; seg = c5[max(0, idx - 4 * m_):idx + 1]; e = seg[0]
        for v in seg[1:]: e = a * v + (1 - a) * e
        M[i] = e
    lab = f"Smoothed last price · EMA {m_}×5min (new)"; FC[lab] = M; FAMILY[lab] = "Smoothed last price (new)"
log("smoothed done")

NEW_LABS = [l for l in FC if l.endswith("(new)")]
SITE_LABS = [l for l, *_r in SITE]


# ------------------------------------------------------------------ rolling daily selection (site Auto rule)
ABS = {lab: np.abs(FC[lab] - A) for lab in FC}


def val_stats(lab, d):
    """Validation MAE (overall + per day) over the 15 trading days before d, using only targets known by d."""
    k = TD.index(d); vd = TD[k - VAL_DAYS:k]; cut = first_idx[d]
    rows = np.isin(odate, vd)
    e = np.where(TGT[rows] < cut, ABS[lab][rows], np.nan)
    dd = odate[rows]
    per = {x: np.nanmean(e[dd == x]) for x in vd if np.isfinite(e[dd == x]).any()}
    return np.nanmean(e), per


NAIVE = "No change · —"


def auto_pick(labs, d, rule=True):
    """rule=True: the website's Auto (lowest validation MAE that is >= 1% better than no change with t >= 2, else no change).
    rule=False: simply the lowest validation MAE among `labs`."""
    st = {l: val_stats(l, d) for l in set(labs) | {NAIVE}}
    nv, nper = st[NAIVE]
    ranked = sorted((l for l in labs if np.isfinite(st[l][0])), key=lambda l: st[l][0])
    if not rule: return ranked[0] if ranked else NAIVE
    for l in ranked:
        if l == NAIVE: return l
        vm, per = st[l]; skill = 1 - vm / nv
        diff = np.array([nper[x] - per.get(x, nper[x]) for x in nper])
        sd = diff.std(ddof=1) if len(diff) > 1 else 0
        t = diff.mean() / (sd / math.sqrt(len(diff))) if sd > 0 else 0
        if skill >= 0.01 and t >= 2: return l
    return NAIVE


test_days = [d for d in TD if d >= TD[long_k - WARM]]      # selection runs from here; scores start at LONG_START
pick_site, pick_new, pick_fam = {}, {}, {}
fams = sorted(set(FAMILY.values()))
for d in test_days:
    pick_site[d] = auto_pick(SITE_LABS, d)
    pick_new[d] = auto_pick(SITE_LABS + NEW_LABS, d)
    for fam in fams:
        pick_fam[(fam, d)] = auto_pick([l for l in FC if FAMILY.get(l) == fam], d, rule=False)
log("rolling selection done")


def assemble(picker):
    M = np.full((nO, H), np.nan)
    for d in test_days:
        lab = picker(d); rows = odate == d
        M[rows] = FC[lab][rows]
    return M


ROLL = {"Site Auto (as on the website)": assemble(lambda d: pick_site[d]),
        "New Auto (site rule, site + new models)": assemble(lambda d: pick_new[d])}
for fam in fams:
    ROLL[fam] = assemble(lambda d, fam=fam: pick_fam[(fam, d)])


def score(M, start):
    rows = odate >= start
    e = M[rows] - A[rows]; ok = np.isfinite(e)
    nv = np.abs(FC["No change · —"][rows] - A[rows])
    ae = np.abs(e)
    mae = np.nanmean(ae); nmae = np.nanmean(np.where(ok, nv, np.nan))
    dd = odate[rows]; days = sorted(set(dd))
    diffs = np.array([np.nanmean(np.where(ok[dd == x], nv[dd == x], np.nan)) - np.nanmean(ae[dd == x]) for x in days])
    diffs = diffs[np.isfinite(diffs)]; days = diffs
    t = diffs.mean() / (diffs.std(ddof=1) / math.sqrt(len(diffs))) if diffs.std() > 1e-12 else 0.0
    dirs = {}
    for h in (4, 24):
        y0 = D.y[ORIG[rows]]; f = M[rows, h - 1]; a = A[rows, h - 1]
        g = np.isfinite(f) & np.isfinite(a) & (np.abs(f - y0) > 1e-9) & (a != y0)
        dirs[h] = float(np.mean((f[g] > y0[g]) == (a[g] > y0[g]))) if g.sum() > 20 else None
    return dict(mae=float(mae), rmse=float(np.sqrt(np.nanmean(e ** 2))), skill=float(1 - mae / nmae),
                by_h=[float(np.nanmean(ae[:, h])) for h in range(H)], better_days=int((diffs > 0).sum()), days=len(days),
                t=float(t), dir4=dirs[4], dir24=dirs[24], n=int(ok.sum()))


out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "last_bar": str(df.datetime.iloc[-1]),
       "bars": len(df), "windows": {}}
for key, start in (("new", day0(NEW_START)), ("long", TD[long_k])):
    rows = []
    for name, M in ROLL.items():
        s = score(M, start); s["name"] = name
        s["kind"] = "new" if "(new)" in name or name.startswith("New") else "site"
        rows.append(s)
    rows.sort(key=lambda r: r["mae"])
    out["windows"][key] = {"start": str(start), "end": str(TD[-1]), "rows": rows,
                           "picks_site": {str(d): pick_site[d] for d in test_days if d >= start},
                           "picks_new": {str(d): pick_new[d] for d in test_days if d >= start}}
    log(key, [(r["name"][:28], round(r["mae"], 2), f"{r['skill'] * 100:+.2f}%") for r in rows])


# ------------------------------------------------------------------ 3. bands: site (error quantiles, last 5 days) vs GARCH x intraday seasonality
from arch import arch_model

CENTER_SITE = ROLL["Site Auto (as on the website)"]
CENTER_NEW = ROLL["New Auto (site rule, site + new models)"]
SIG = np.full((nO, H), np.nan)                              # GARCH sd of log price change h hours ahead
garch_params = {}


def garch_day(d, upto=None):
    k = TD.index(d) if d in TD else len(TD); cut = first_idx[d] if d in first_idx else n
    lo = first_idx[TD[max(0, k - 120)]]
    r = r1[lo + 1:cut]; hh = hod[lo + 1:cut]
    lo60 = first_idx[TD[max(0, k - 60)]]
    r60, h60 = r1[lo60 + 1:cut], hod[lo60 + 1:cut]
    base = np.nanmean(r60 ** 2)
    seas = np.array([np.nanmean(r60[h60 == x] ** 2) / base if (h60 == x).sum() > 10 else 1.0 for x in range(24)])
    seas = 0.8 * seas + 0.2                                   # mild shrink toward 1 (24 bins, ~60 obs each)
    u = r / np.sqrt(seas[hh]) * 100
    res = arch_model(u, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    w, a, b = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
    s2 = np.var(u)
    for x in u: s2 = w + a * x * x + b * s2                  # filtered variance for the first hour of day d
    return dict(w=w, a=a, b=b, seas=seas, s2=s2, nu=res.params["nu"])


def sig_path(p, s2_next, hods):
    V = p["w"] / max(1e-6, 1 - p["a"] - p["b"]); ab = p["a"] + p["b"]
    var = np.cumsum([p["seas"][x] * (V + ab ** i * (s2_next - V)) for i, x in enumerate(hods)])
    return np.sqrt(var) / 100


for d in [x for x in TD if first_idx[x] >= o_lo]:
    p = garch_day(d); garch_params[d] = p
    s2 = p["s2"]                                            # variance of the return into hour first_idx[d]
    for j in range(first_idx[d], min(n, first_idx[d] + int(cnt[d]))):
        s2 = p["w"] + p["a"] * (r1[j] / math.sqrt(p["seas"][hod[j]]) * 100) ** 2 + p["b"] * s2   # -> variance of return j+1
        SIG[j - o_lo] = sig_path(p, s2, hod_ext[j + 1:j + 1 + H])
log("GARCH done")


def band_eval(start):
    """Coverage / width / Winkler interval score for site vs new bands, 80% and 95%."""
    res = {"site": {}, "new": {}}
    rows_test = odate >= start
    # site: half-width per horizon = quantile of |error| of the chosen model over the previous 5 trading days
    hw_site = {q: np.full((nO, H), np.nan) for q in (0.8, 0.95)}
    hw_new = {q: np.full((nO, H), np.nan) for q in (0.8, 0.95)}
    for d in test_days:
        k = TD.index(d); cut = first_idx[d]; rows = odate == d
        prev5 = np.isin(odate, TD[k - 5:k]); known = TGT < cut
        es = np.where(prev5[:, None] & known, np.abs(CENTER_SITE - A), np.nan)
        prev20 = np.isin(odate, TD[k - 20:k])
        zz = np.where(prev20[:, None] & known, np.abs(np.log(A) - np.log(CENTER_NEW)) / SIG, np.nan)
        for q in (0.8, 0.95):
            hw_site[q][rows] = np.nanquantile(es, q, axis=0)
            hw_new[q][rows] = np.nanquantile(zz, q)          # one calibrated multiplier, scaled by each origin's sigma
    for q in (0.8, 0.95):
        alpha = 1 - q
        for key, C, lo_, hi_ in (("site", CENTER_SITE, CENTER_SITE - hw_site[q], CENTER_SITE + hw_site[q]),
                                 ("new", CENTER_NEW, CENTER_NEW * np.exp(-hw_new[q] * SIG), CENTER_NEW * np.exp(hw_new[q] * SIG))):
            a = A[rows_test]; L, U = lo_[rows_test], hi_[rows_test]
            g = np.isfinite(a) & np.isfinite(L) & np.isfinite(U)
            a, L, U = a[g], L[g], U[g]
            wid = U - L
            wink = wid + 2 / alpha * (np.maximum(L - a, 0) + np.maximum(a - U, 0))
            res[key][str(q)] = dict(coverage=float(np.mean((a >= L) & (a <= U))), width=float(np.mean(wid)), winkler=float(np.mean(wink)))
            A2, L2, U2 = A[rows_test], lo_[rows_test], hi_[rows_test]
            cov_h, wid_h = [], []
            for h in range(H):
                g = np.isfinite(A2[:, h]) & np.isfinite(L2[:, h]) & np.isfinite(U2[:, h])
                cov_h.append(float(np.mean((A2[g, h] >= L2[g, h]) & (A2[g, h] <= U2[g, h]))))
                wid_h.append(float(np.mean(U2[g, h] - L2[g, h])))
            res[key][str(q)].update(cov_h=cov_h, wid_h=wid_h)
            # coverage split by how volatile the hour was (calm vs busy), from the realised |24h move|
            mv = np.abs(A2[:, H - 1] - D.y[ORIG[rows_test]]); g = np.isfinite(mv) & np.isfinite(L2[:, H - 1]) & np.isfinite(U2[:, H - 1])
            hit = (A2[g, H - 1] >= L2[g, H - 1]) & (A2[g, H - 1] <= U2[g, H - 1])
            vb = vol24[ORIG[rows_test]][g]; qs = np.nanquantile(vb, [1 / 3, 2 / 3])
            res[key][str(q)]["cov_by_vol"] = [float(np.mean(hit[vb <= qs[0]])), float(np.mean(hit[(vb > qs[0]) & (vb <= qs[1])])), float(np.mean(hit[vb > qs[1]]))]
            res[key][str(q)]["wid_by_vol"] = [float(np.mean((U2[g, H - 1] - L2[g, H - 1])[m])) for m in (vb <= qs[0], (vb > qs[0]) & (vb <= qs[1]), vb > qs[1])]
    return res, hw_site, hw_new


for key, start in (("new", day0(NEW_START)), ("long", TD[long_k])):
    br, HWS, HWN = band_eval(start)
    out["windows"][key]["bands"] = br
    log("bands", key, json.dumps(br))

# chart series for the new-data window: actual, and what each method said 24 trading hours earlier (+ 80% band)
st = first_idx[day0(NEW_START)]
ser = {"t": [], "actual": [], "site": [], "new": [], "pattern": [], "gbm": [], "site_lo": [], "site_hi": [], "new_lo": [], "new_hi": []}
for j in range(st, n):
    o = j - H; i = o - o_lo
    if i < 0: continue
    ser["t"].append(hours[j].strftime("%Y-%m-%d %H:00")); ser["actual"].append(round(float(D.y[j]), 2))
    put = lambda k, v: ser[k].append(None if not np.isfinite(v) else round(float(v), 2))
    put("site", CENTER_SITE[i, H - 1]); put("new", CENTER_NEW[i, H - 1])
    put("pattern", ROLL["Daily pattern"][i, H - 1]); put("gbm", FC["Gradient boosting (new)"][i, H - 1])
    put("site_lo", CENTER_SITE[i, H - 1] - HWS[0.8][i, H - 1]); put("site_hi", CENTER_SITE[i, H - 1] + HWS[0.8][i, H - 1])
    put("new_lo", CENTER_NEW[i, H - 1] * math.exp(-HWN[0.8][i, H - 1] * SIG[i, H - 1]))
    put("new_hi", CENTER_NEW[i, H - 1] * math.exp(HWN[0.8][i, H - 1] * SIG[i, H - 1]))
out["series24"] = ser


# ------------------------------------------------------------------ 4. buy / sell hour: site median pattern vs recency-weighted rank
DAYS = D.days; DL = D.day_list; MED = D.med; WIN = 12


def site_windows(ref):
    rel = np.vstack([(DAYS[d] / MED[d] - 1) * 100 for d in ref])
    cntv = np.isfinite(rel).sum(0)
    with np.errstate(all="ignore"): med = np.nanmedian(rel, axis=0)
    med[cntv < 2] = np.nan
    cands = [(s, float(np.median(med[s:s + WIN]))) for s in range(288 - WIN + 1) if np.isfinite(med[s:s + WIN]).all()]
    return pick(cands, lowest_is_buy=True)


def pick(cands, lowest_is_buy):
    if not cands: return None
    asc = sorted(cands, key=lambda x: x[1]); desc = asc[::-1]
    ov = lambda a, b: abs(a - b) < WIN
    buy = asc[0][0]; sell = next((s for s, _ in desc if not ov(s, buy)), None)
    return (buy, sell) if sell is not None else None


def ref_days(d, ref, weekday):
    prior = [x for x in DL if x < d]
    if not prior: return []
    frm = prior[-1] - timedelta(days=ref - 1)
    r = [x for x in prior if x >= frm]
    if weekday:
        same = [x for x in r if x.weekday() == d.weekday()]
        if len(same) >= 3: return same
    return r


def window_ranks(c):
    """Rank (0 = cheapest, 1 = dearest) of every full 1-hour window's mean price within one day."""
    ok = np.isfinite(c); cs = np.r_[0, np.cumsum(np.where(ok, c, 0))]; cn = np.r_[0, np.cumsum(ok)]
    win = (cs[WIN:] - cs[:-WIN]) / WIN; win[(cn[WIN:] - cn[:-WIN]) < WIN] = np.nan
    g = np.isfinite(win); rk = np.full_like(win, np.nan)
    if g.sum() >= 24: rk[g] = np.argsort(np.argsort(win[g])) / (g.sum() - 1)
    return rk


RANKS = {}
for x in DL:
    c = DAYS[x]; g = np.isfinite(c); s = np.arange(288)
    ct = c - np.polyval(np.polyfit(s[g], c[g], 1), s) if g.sum() > 20 else c
    RANKS[(x, False)], RANKS[(x, True)] = window_ranks(c), window_ranks(ct)


def rank_windows(d, hl, detrend, L=91):
    """Recency-weighted average rank of each 1-hour window over the last L calendar days (half-life hl trading days)."""
    prior = [x for x in DL if d - timedelta(days=L) <= x < d]
    if len(prior) < 5: return None
    acc = np.zeros(288 - WIN + 1); wsum = np.zeros(288 - WIN + 1); wtot = 0
    for age, x in enumerate(reversed(prior)):
        w = 0.5 ** (age / hl); rk = RANKS[(x, detrend)]; g = np.isfinite(rk)
        if not g.any(): continue
        acc[g] += w * rk[g]; wsum[g] += w; wtot += w
    ok = wsum >= 0.5 * wtot
    cands = [(s, acc[s] / wsum[s]) for s in range(len(acc)) if ok[s]]
    return pick(cands, True)


def outcome(d, wins):
    if not wins: return None
    c = DAYS[d]; m = MED[d]
    val = lambda s: np.nanmean(c[s:s + WIN]) if np.isfinite(c[s:s + WIN]).any() else np.nan
    b, s = val(wins[0]), val(wins[1])
    if not (np.isfinite(b) and np.isfinite(s)): return None
    return dict(buy=(b / m - 1) * 100, sell=(s / m - 1) * 100, spread=(s / b - 1) * 100, usd=float(s - b),
                bw=int(wins[0]), sw=int(wins[1]))


tdays = [d for d in DL if date(2026, 4, 1) <= d < hdate[-1]]   # complete days only (today is still running)
SITE_SET = [(r, w) for r in (7, 14, 30, 91, 182) for w in (False, True)]
NEW_SET = [(hl, dt) for hl in (5, 10, 20, 40) for dt in (False, True)]
RES = {}
for d in tdays:
    for sset in SITE_SET:
        rd = ref_days(d, *sset)
        RES[("site", sset, d)] = outcome(d, site_windows(rd)) if len(rd) >= 3 else None
    for nset in NEW_SET:
        RES[("new", nset, d)] = outcome(d, rank_windows(d, *nset))
log("timing grid done")


def binom_tail(k, nn):
    from math import comb
    return sum(comb(nn, i) for i in range(k, nn + 1)) / 2 ** nn


def site_verdict_score(vals):
    vals = [v for v in vals if v]
    if len(vals) < 10: return (0, -1e9)
    nn = len(vals); kb = sum(v["buy"] < 0 for v in vals); ks = sum(v["sell"] > 0 for v in vals)
    spread = float(np.median([v["spread"] for v in vals])); worst = max(binom_tail(kb, nn), binom_tail(ks, nn))
    lvl = 3 if worst < 0.05 and spread > 0.05 else 2 if worst < 0.20 and spread > 0.02 else 1
    return (lvl, min(kb, ks) / nn * 100 + spread)


def rolling(kind, sets, scorer, d, look=20):
    prev = [x for x in tdays if x < d][-look:]
    return max(sets, key=lambda s: scorer([RES[(kind, s, x)] for x in prev]))


mean_spread = lambda vals: (1, float(np.mean([v["spread"] for v in vals if v])) if any(vals) else -1e9)
timing = {}
for key, start in (("new", day0(NEW_START)), ("long", TD[long_k])):
    ds = [d for d in tdays if d >= start]
    methods = {
        "Site default (30-day median pattern)": [RES[("site", (30, False), d)] for d in ds],
        "Site auto-scan (best setting, last 20 days)": [RES[("site", rolling("site", SITE_SET, site_verdict_score, d), d)] for d in ds],
        "New: weighted rank (auto, last 20 days)": [RES[("new", rolling("new", NEW_SET, mean_spread, d), d)] for d in ds],
        "New: weighted rank (half-life 10 days)": [RES[("new", (10, False), d)] for d in ds],
    }
    tab = {}
    for name, vals in methods.items():
        v = [x for x in vals if x]
        sp = np.array([x["spread"] for x in v])
        tab[name] = dict(days=len(v), buy=float(np.mean([x["buy"] for x in v])), sell=float(np.mean([x["sell"] for x in v])),
                         spread=float(sp.mean()), spread_med=float(np.median(sp)), usd=float(np.mean([x["usd"] for x in v])),
                         buy_hit=float(np.mean([x["buy"] < 0 for x in v])), sell_hit=float(np.mean([x["sell"] > 0 for x in v])),
                         win=float(np.mean(sp > 0)), t=float(sp.mean() / (sp.std(ddof=1) / math.sqrt(len(sp)))) if len(sp) > 2 else 0.0,
                         cum=[round(float(x), 2) for x in np.cumsum([x["usd"] for x in v])],
                         dates=[str(d) for d, x in zip(ds, vals) if x])
    timing[key] = tab
    log("timing", key, {k: (round(v["spread"], 3), round(v["usd"], 1)) for k, v in tab.items()})
out["timing"] = timing


# ------------------------------------------------------------------ 5. live forecast from the latest bar
j = n - 1
today = hdate[j]
fut = EXT[n:n + H]
live = {"origin": str(df.datetime.iloc[-1]), "price": float(df.close.iloc[-1]), "t": [h.strftime("%Y-%m-%d %H:00") for h in fut]}
# new model: gradient boosting refit on everything known now
m = gbm_fit(n)
X, _, J, Hh = gbm_rows([j], False)
p = m.predict(X)
live["gbm"] = [round(float(df.close.iloc[-1] * math.exp(x * vol120[j] * math.sqrt(h))), 2) for x, h in zip(p, Hh)]
# GARCH band from today's fit, calibrated on the last 20 trading days
pg = garch_day(today) if today in first_idx else garch_day(TD[-1])
s2 = pg["s2"]
for jj in range(first_idx[today], n):
    s2 = pg["w"] + pg["a"] * (r1[jj] / math.sqrt(pg["seas"][hod[jj]]) * 100) ** 2 + pg["b"] * s2
sg = sig_path(pg, s2, hod_ext[n:n + H])
k = TD.index(today); prev20 = np.isin(odate, TD[k - 20:k])
zz = np.where(prev20[:, None], np.abs(np.log(A) - np.log(CENTER_NEW)) / SIG, np.nan)
q80, q95 = np.nanquantile(zz, 0.8), np.nanquantile(zz, 0.95)
P0 = float(df.close.iloc[-1])
live["new_pick"] = pick_new.get(today, NAIVE); live["site_pick"] = pick_site.get(today, NAIVE)


def live_center(lab):
    if lab == NAIVE: return np.full(H, P0)
    if lab.startswith("Gradient boosting"):
        k = 0.5 if "×0.5" in lab else 1.0
        return np.array([P0 * math.exp(k * x * vol120[j] * math.sqrt(h)) for x, h in zip(p, Hh)])
    log("WARNING: no live path for", lab, "- using no change"); return np.full(H, P0)


ctr_new = live_center(live["new_pick"])
live["new"] = [round(float(x), 2) for x in ctr_new]
live["new_lo80"] = [round(float(c * math.exp(-q80 * s)), 2) for c, s in zip(ctr_new, sg)]
live["new_hi80"] = [round(float(c * math.exp(q80 * s)), 2) for c, s in zip(ctr_new, sg)]
live["new_lo95"] = [round(float(c * math.exp(-q95 * s)), 2) for c, s in zip(ctr_new, sg)]
live["new_hi95"] = [round(float(c * math.exp(q95 * s)), 2) for c, s in zip(ctr_new, sg)]
prev5 = np.isin(odate, TD[k - 5:k])
es = np.where(prev5[:, None], np.abs(CENTER_SITE - A), np.nan)
e80, e95 = np.nanquantile(es, 0.8, axis=0), np.nanquantile(es, 0.95, axis=0)
ctr_site = live_center(live["site_pick"])
live["site"] = [round(float(x), 2) for x in ctr_site]
live["site_lo80"] = [round(float(c - e), 2) for c, e in zip(ctr_site, e80)]
live["site_hi80"] = [round(float(c + e), 2) for c, e in zip(ctr_site, e80)]
live["recent_t"] = [h.strftime("%Y-%m-%d %H:00") for h in hours[n - 72:n]]
live["recent"] = [round(float(x), 2) for x in D.y[n - 72:n]]
# buy / sell hours for today and tomorrow (Thai time)
hhmm = lambda s: f"{s * 5 // 60 % 24:02d}:{s * 5 % 60:02d}"
nxt_day = next(h.date() for h in fut if h.date() > today)
best_new = rolling("new", NEW_SET, mean_spread, today)
wins = {}
for dd in (today, nxt_day):
    ws = site_windows(ref_days(dd, 30, False)); wn = rank_windows(dd, *best_new)
    wins[str(dd)] = {"site": [hhmm(ws[0]), hhmm(ws[0] + WIN), hhmm(ws[1]), hhmm(ws[1] + WIN)] if ws else None,
                     "new": [hhmm(wn[0]), hhmm(wn[0] + WIN), hhmm(wn[1]), hhmm(wn[1] + WIN)] if wn else None}
live["windows"] = wins; live["new_setting"] = {"half_life": best_new[0], "detrend": best_new[1]}
out["live"] = live
log("live", live["origin"], live["price"], live["new_pick"], live["site_pick"], wins)

json.dump(out, open(os.path.join(HERE, "results_new_methods.json"), "w"), default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o))
log("wrote results.json")

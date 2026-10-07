"""Detailed view of the model experiment (uses the same data, models and rules as eval_models.py).

Outputs (in research/):
  details.md            every model x setting: error on the validation days and on the test days
  fig_horizon.png       error vs how far ahead (1-24 trading hours), best setting of each model
  fig_lookback.png      how the look-back window (1 week ... 6 months) changes each model's error
  fig_daily.png         day-by-day: better or worse than "no change"?
  fig_paths.png         5-7 Oct: actual price vs each model's forecast made 24 trading hours earlier
"""
import os
from datetime import datetime
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eval_models as E

HERE = os.path.dirname(__file__)
FAMS = ["No change", "Daily pattern (current)", "Shrunk pattern", "Seasonal naive", "Drift", "ETS", "AR on returns", "ARIMA(1,1,1)"]
COL = dict(zip(FAMS, ["#222222", "#e8590c", "#f59f00", "#868e96", "#9c36b5", "#1c7ed6", "#2f9e44", "#0ca678"]))


def run(D, F, fn, cfg, j):
    return fn(D, j, cfg, F) if fn in (E.f_ets, E.f_arima) else fn(D, j, cfg)


def detail(D, F, fn, cfg, js):
    """Absolute errors by horizon and by origin day (None if the model cannot cover every origin)."""
    by_h = {h: [] for h in range(1, E.H + 1)}; by_day = {}
    for j in js:
        fc = run(D, F, fn, cfg, j)
        if fc is None: return None
        d = D.hours[j].date()
        for h in range(1, E.H + 1):
            if j + h >= len(D.y) or not np.isfinite(fc[h - 1]): continue
            e = abs(fc[h - 1] - D.y[j + h]); by_h[h].append(e); by_day.setdefault(d, []).append(e)
    allv = np.concatenate([np.array(v) for v in by_h.values() if v])
    return dict(mae=allv.mean(), by_h={h: np.mean(v) for h, v in by_h.items() if v}, by_day={d: np.mean(v) for d, v in by_day.items()})


def main():
    df, _ = E.load(); D = E.Data(df); F = E.DayFitted(D)
    last = D.hours[-1]; td = sorted({h.date() for h in D.hours})
    val = E.origins_between(D, datetime.combine(td[-31], datetime.min.time()), datetime.combine(td[-21], datetime.min.time()))
    test = E.origins_between(D, datetime.combine(td[-21], datetime.min.time()), last)

    cands = [("No change", E.f_naive, {})] + list(E.candidates())
    rows = []
    for name, fn, cfg in cands:
        v = detail(D, F, fn, cfg, val); t = detail(D, F, fn, cfg, test)
        if v and t: rows.append(dict(name=name, fn=fn, cfg=cfg, val=v, test=t))
        print(f"{name:26s} {E.cfg_label(cfg):28s} val {v['mae'] if v else float('nan'):6.2f}  test {t['mae'] if t else float('nan'):6.2f}", flush=True)
    naive = rows[0]
    for r in rows:
        r["val_skill"] = 1 - r["val"]["mae"] / naive["val"]["mae"]
        r["test_skill"] = 1 - r["test"]["mae"] / naive["test"]["mae"]
        dd = np.array([naive["test"]["by_day"][d] - r["test"]["by_day"][d] for d in naive["test"]["by_day"]])
        r["better_days"] = int((dd > 0).sum()); r["n_days"] = len(dd)
        r["t"] = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if dd.std(ddof=1) > 0 else 0.0
    best = {}
    for r in rows:                                                     # best setting per model, chosen on VALIDATION days
        if r["name"] not in best or r["val"]["mae"] < best[r["name"]]["val"]["mae"]: best[r["name"]] = r
    fams = [f for f in FAMS if f in best]

    # ---- details.md
    v0, v1, t0 = D.hours[val[0]].date(), D.hours[val[-1]].date(), D.hours[test[0]].date()
    md = ["# Model experiment: full details", "",
          f"Validation days (used to choose each model's settings): {v0} → {v1} ({len(val)} hourly forecasts).",
          f"Test days (never used for choosing): {t0} → {last.date()} ({len(test)} hourly forecasts, each 1–24 trading hours ahead).",
          "Error = mean absolute difference between forecast and actual hourly close, in USD. Skill = how much smaller than 'no change' (positive = better).",
          "'Better days' = on how many test days the model's average error was below 'no change'. t = paired t-statistic of those daily differences (t ≥ 2 ≈ consistently better).", "",
          "## Best setting of each model (chosen on validation, scored on test)", "",
          "| Model | Setting | Validation error | Test error | Test skill | Better days | t |", "|---|---|---|---|---|---|---|"]
    for f in sorted(fams, key=lambda f: best[f]["test"]["mae"]):
        r = best[f]
        md.append(f"| {f} | {E.cfg_label(r['cfg'])} | ${r['val']['mae']:.2f} | ${r['test']['mae']:.2f} | {r['test_skill']*100:+.1f}% | {r['better_days']}/{r['n_days']} | {r['t']:.1f} |")
    md += ["", "## Every model and setting tried", "",
           "| Model | Setting | Validation error | Validation skill | Test error | Test skill | Better days | t |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["test"]["mae"]):
        md.append(f"| {r['name']} | {E.cfg_label(r['cfg'])} | ${r['val']['mae']:.2f} | {r['val_skill']*100:+.1f}% | ${r['test']['mae']:.2f} | {r['test_skill']*100:+.1f}% | {r['better_days']}/{r['n_days']} | {r['t']:.1f} |")
    e = best["ETS"]
    md += ["", f"Overfitting example: ETS ({E.cfg_label(e['cfg'])}) looked best on the validation days ({e['val_skill']*100:+.1f}% vs no change) "
           f"but was {e['test_skill']*100:+.1f}% on the test days. Small wins on a few days often do not repeat."]
    open(os.path.join(HERE, "details.md"), "w").write("\n".join(md))

    # ---- fig_horizon
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for f in fams:
        bh = best[f]["test"]["by_h"]
        ax.plot(list(bh), list(bh.values()), color=COL[f], lw=2.6 if f == "No change" else 1.5,
                ls="--" if f == "Daily pattern (current)" else "-", label=f"{f} ({E.cfg_label(best[f]['cfg'])})")
    ax.set_xlabel("Trading hours ahead"); ax.set_ylabel("Average error, USD")
    ax.set_title("Error grows with how far ahead you forecast — and every model tracks 'No change'\n(last 20 trading days, best setting of each model)")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_horizon.png"), dpi=130)

    # ---- fig_lookback (test skill per family x look-back; best other setting within that look-back)
    Ls = list(E.LOOKBACKS); lf = ["Daily pattern (current)", "Shrunk pattern", "ETS", "AR on returns", "ARIMA(1,1,1)"]
    M = np.full((len(lf), len(Ls)), np.nan)
    for a, f in enumerate(lf):
        for b, L in enumerate(Ls):
            rs = [r for r in rows if r["name"] == f and r["cfg"].get("L") == L]
            if rs: M[a, b] = min(rs, key=lambda r: r["val"]["mae"])["test_skill"] * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lim = np.nanmax(np.abs(M)); im = ax.imshow(M, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for a in range(len(lf)):
        for b in range(len(Ls)):
            if np.isfinite(M[a, b]): ax.text(b, a, f"{M[a,b]:+.1f}%", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(Ls)), ["1 week", "2 weeks", "1 month", "3 months", "6 months"]); ax.set_yticks(range(len(lf)), lf)
    ax.set_title("Look-back window: test skill vs 'No change' (green = better)"); fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_lookback.png"), dpi=130)

    # ---- fig_daily
    days = sorted(naive["test"]["by_day"]); x = np.arange(len(days))
    fig, ax = plt.subplots(figsize=(11, 5))
    others = [f for f in fams if f not in ("No change", "Seasonal naive")]
    w = 0.8 / len(others)
    for i, f in enumerate(others):
        bd = best[f]["test"]["by_day"]
        ax.bar(x - 0.4 + w * (i + .5), [(naive["test"]["by_day"][d] - bd[d]) for d in days], w, color=COL[f], label=f)
    ax.axhline(0, color="#222", lw=1)
    ax.set_xticks(x, [d.strftime("%d/%m") for d in days], rotation=45)
    ax.set_ylabel("USD less error than 'No change' (above 0 = better)")
    ax.set_title("Day by day: no model is better than 'No change' consistently")
    ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8, ncol=3); fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_daily.png"), dpi=130)

    # ---- fig_paths (5-7 Oct, forecast made 24 trading hours earlier)
    js = [j for j, t in enumerate(D.hours) if datetime(2026, 10, 5) <= t <= last]
    show = [f for f in fams if f != "No change"]
    fig, axs = plt.subplots(len(show) // 2 + len(show) % 2, 2, figsize=(13, 2.6 * (len(show) // 2 + len(show) % 2)), sharex=True, sharey=True)
    for ax, f in zip(axs.flat, show):
        ax.plot([D.hours[j] for j in js], D.y[js], color="#222", lw=1.8, label="Actual")
        for nm, style in (("No change", ":"), (f, "-")):
            r = best[nm]; xs, ys = [], []
            for j in js:
                if j - E.H < 0: continue
                fc = run(D, F, r["fn"], r["cfg"], j - E.H)
                if fc is not None and np.isfinite(fc[E.H - 1]): xs.append(D.hours[j]); ys.append(fc[E.H - 1])
            ax.plot(xs, ys, color=COL[nm] if nm != "No change" else "#999", ls=style, lw=1.5, label=f"{nm} (24 h earlier)")
        ax.set_title(f"{f} · test skill {best[f]['test_skill']*100:+.1f}%", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=7)
    for ax in axs.flat[len(show):]: ax.axis("off")
    fig.suptitle("5–7 Oct 2026: actual price vs each model's forecast made 24 trading hours earlier", fontsize=12)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_paths.png"), dpi=120)
    print("\n".join(md[:20]))


if __name__ == "__main__":
    main()

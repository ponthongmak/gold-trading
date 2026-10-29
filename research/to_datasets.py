"""Split results.json into the row files the dashboard reads (one JSON array per dataset)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "datasets"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results_new_methods.json")))
WIN = {"new": "oct", "long": "jul_oct"}
r4 = lambda v: None if v is None else round(v, 4)


def save(name, rows):
    json.dump(rows, open(os.path.join(OUT, name + ".json"), "w"), ensure_ascii=False, separators=(",", ":"))
    print(name, len(rows), os.path.getsize(os.path.join(OUT, name + ".json")))


models, bands, band_h = [], [], []
for k, w in R["windows"].items():
    for r in w["rows"]:
        models.append(dict(window=WIN[k], model=r["name"], kind=r["kind"], mae=r4(r["mae"]), skill=r4(r["skill"]),
                           better_days=r["better_days"], days=r["days"], t=r4(r["t"]), dir24=r4(r["dir24"]),
                           mae_h1=r4(r["by_h"][0]), mae_h4=r4(r["by_h"][3]), mae_h12=r4(r["by_h"][11]), mae_h24=r4(r["by_h"][23])))
    for m, lv in w["bands"].items():
        for q, v in lv.items():
            bands.append(dict(window=WIN[k], method=m, level=float(q), coverage=r4(v["coverage"]), width=r4(v["width"]), winkler=r4(v["winkler"])))
            for h, (c, wd) in enumerate(zip(v["cov_h"], v["wid_h"]), 1):
                band_h.append(dict(window=WIN[k], method=m, level=float(q), h=h, coverage=r4(c), width=r4(wd)))
save("models", models); save("bands", bands); save("band_h", band_h)

s = R["series24"]
save("backtest", [dict(t=t, actual=s["actual"][i], site=s["site"][i], pattern=s["pattern"][i], gbm=s["gbm"][i],
                       new=s["new"][i], site_lo=s["site_lo"][i], site_hi=s["site_hi"][i], new_lo=s["new_lo"][i], new_hi=s["new_hi"][i])
                  for i, t in enumerate(s["t"])])

L = R["live"]
live = [dict(t=t, phase="recent", actual=v) for t, v in zip(L["recent_t"], L["recent"])]
live.append(dict(t=L["origin"][:16], phase="now", actual=round(L["price"], 2), site=round(L["price"], 2), new=round(L["price"], 2),
                 gbm=round(L["price"], 2), site_lo80=round(L["price"], 2), site_hi80=round(L["price"], 2),
                 new_lo80=round(L["price"], 2), new_hi80=round(L["price"], 2), new_lo95=round(L["price"], 2), new_hi95=round(L["price"], 2)))
for i, t in enumerate(L["t"]):
    live.append(dict(t=t, phase="forecast", site=L["site"][i], site_lo80=L["site_lo80"][i], site_hi80=L["site_hi80"][i],
                     new=L["new"][i], new_lo80=L["new_lo80"][i], new_hi80=L["new_hi80"][i], new_lo95=L["new_lo95"][i],
                     new_hi95=L["new_hi95"][i], gbm=L["gbm"][i]))
save("live", live)

hours = []
for d, v in L["windows"].items():
    for m in ("site", "new"):
        if v[m]: hours.append(dict(date=d, method=m, buy_from=v[m][0], buy_to=v[m][1], sell_from=v[m][2], sell_to=v[m][3]))
save("hours", hours)

NAME = {"Site default (30-day median pattern)": "site_default", "Site auto-scan (best setting, last 20 days)": "site_scan",
        "New: weighted rank (auto, last 20 days)": "new_auto", "New: weighted rank (half-life 10 days)": "new_hl10"}
timing, cum = [], []
for k, tab in R["timing"].items():
    for name, v in tab.items():
        timing.append(dict(window=WIN[k], method=NAME[name], days=v["days"], buy=r4(v["buy"] / 100), sell=r4(v["sell"] / 100),
                           spread=r4(v["spread"] / 100), usd=round(v["usd"], 2), buy_hit=r4(v["buy_hit"]), sell_hit=r4(v["sell_hit"]),
                           win=r4(v["win"]), t=r4(v["t"])))
        if k == "long":
            cum += [dict(method=NAME[name], date=d, cum_usd=c) for d, c in zip(v["dates"], v["cum"])]
save("timing", timing); save("timing_cum", cum)
print("live origin", L["origin"], "price", L["price"])

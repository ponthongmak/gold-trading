"""Append new XAU/USD 5-min bars (Thai time) from Twelve Data to the existing history.

Reads data/gold_5min.csv, fetches everything after its last bar, writes data/gold_5min_full.csv.
Key comes from TD_KEY (never printed).
"""
import csv, json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime, timedelta

KEY = os.environ.get("TD_KEY") or sys.exit("Set TD_KEY")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gold_5min.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gold_5min_full.csv")

bars = {}
with open(SRC) as f:
    for r in csv.DictReader(f):
        bars[r["datetime"]] = (r["open"], r["high"], r["low"], r["close"])
last_old = max(bars)
print("existing:", len(bars), "bars, last", last_old)

end, calls, new = None, 0, 0
while calls < 6:
    q = {"symbol": "XAU/USD", "interval": "5min", "timezone": "Asia/Bangkok", "outputsize": 5000, "apikey": KEY,
         "start_date": last_old}
    if end: q["end_date"] = end
    if calls: time.sleep(8)
    with urllib.request.urlopen("https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(q), timeout=60) as r:
        j = json.load(r)
    calls += 1
    if j.get("status") == "error":
        if "no data" in j.get("message", "").lower(): break
        sys.exit(f"Twelve Data error {j.get('code')}: {j.get('message')}")
    vals = j.get("values", [])
    if not vals: break
    for v in vals:
        if v["datetime"] not in bars: new += 1
        bars[v["datetime"]] = (v["open"], v["high"], v["low"], v["close"])
    oldest = min(v["datetime"] for v in vals)
    print(f"call {calls}: {len(vals)} bars, {oldest} -> {max(v['datetime'] for v in vals)}", flush=True)
    if oldest <= last_old or len(vals) < 5000: break
    end = (datetime.strptime(oldest, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["datetime", "open", "high", "low", "close"])
    for dt in sorted(bars): w.writerow([dt, *bars[dt]])
print(f"saved {len(bars)} bars ({min(bars)} -> {max(bars)}), {new} new, {calls} API calls")

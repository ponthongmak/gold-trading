"""Download XAU/USD 5-minute OHLC (Thai time) from Twelve Data into data/gold_5min.csv.

The API key is read from the TD_KEY environment variable and is never written or printed.
Pages backwards 5000 bars at a time (8 s apart: free plan allows 8 calls/min) until ~7 months are covered.
"""
import csv, json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime, timedelta

KEY = os.environ.get("TD_KEY")
if not KEY:
    sys.exit("Set TD_KEY in the environment.")
OUT = os.path.join(os.path.dirname(__file__), "data", "gold_5min.csv")
DAYS_BACK = 215                                   # 6-month training window + test period + margin


def fetch(end_date=None):
    q = {"symbol": "XAU/USD", "interval": "5min", "timezone": "Asia/Bangkok", "outputsize": 5000, "apikey": KEY}
    if end_date:
        q["end_date"] = end_date
    with urllib.request.urlopen("https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(q), timeout=60) as r:
        j = json.load(r)
    if j.get("status") == "error":
        if "no data" in j.get("message", "").lower():
            return []
        raise RuntimeError(f"Twelve Data error {j.get('code')}: {j.get('message')}")
    return j.get("values", [])


bars, end, calls = {}, None, 0
target = datetime.now() + timedelta(hours=7) - timedelta(days=DAYS_BACK)  # Thai time
while True:
    if calls:
        time.sleep(8)
    vals = fetch(end)
    calls += 1
    if not vals:
        break
    for v in vals:
        bars[v["datetime"]] = (v["open"], v["high"], v["low"], v["close"])
    oldest = min(v["datetime"] for v in vals)
    print(f"call {calls}: {len(vals)} bars, oldest {oldest}", flush=True)
    if datetime.strptime(oldest, "%Y-%m-%d %H:%M:%S") <= target or calls >= 14:
        break
    end = (datetime.strptime(oldest, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "open", "high", "low", "close"])
    for dt in sorted(bars):
        w.writerow([dt, *bars[dt]])
print(f"saved {len(bars)} bars ({min(bars)} → {max(bars)}) to {OUT} using {calls} API calls")

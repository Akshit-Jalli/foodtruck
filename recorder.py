"""Archive every day of the live game, for calibrating the customer model.

    python recorder.py            # long-polls and appends to days.jsonl

One long-poll connection, same as the bot. Each completed day is written once:
the forecast that preceded it, the weather and events that landed, and every
spot's customers, hungry count and the full line-up of trucks with their
price, quality, meals and sales. That is everything needed to fit the
per-spot price-limit and quality-minimum priors in bot.py.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("NIGHTSHIFT_URL", "https://nightshift.joinbuilders.org").rstrip("/")
KEY = os.environ.get("NIGHTSHIFT_KEY", "")
OUT = os.environ.get("NIGHTSHIFT_DAYS", "days.jsonl")


def get(path, timeout=35):
    req = urllib.request.Request(URL + path, headers={"Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {"error": str(e)}


def seen_days():
    days = set()
    try:
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    days.add((r.get("mode"), r.get("market"), r.get("day")))
                except ValueError:
                    pass
    except OSError:
        pass
    return days


def main():
    sys.stdout.reconfigure(line_buffering=True)
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first.")
    done = seen_days()
    print("recorder started, %d days already archived -> %s" % (len(done), OUT))
    version, backoff, last_forecast, hangups = 0, 1, None, 0
    while True:
        try:
            status, state = get("/api/v1/state?since=%d" % version)
        except (urllib.error.URLError, OSError) as e:
            s = str(e)
            if "10054" in s or "forcibly closed" in s or "Connection reset" in s:
                hangups += 1
                time.sleep(min(0.25 * hangups, 2.0))
                continue
            print("network error (%s), retry in %ds" % (e, backoff))
            time.sleep(backoff)
            backoff = min(backoff * 2, 15)
            continue
        backoff, hangups = 1, 0
        if status == 429:
            time.sleep(3)
            continue
        if status != 200:
            print("error %d: %s" % (status, state.get("error")))
            if status == 401:
                sys.exit(1)
            time.sleep(5)
            continue
        version = state["version"]

        last = state.get("last_day") or {}
        key = (state.get("mode"), state.get("market"), last.get("day"))
        if last.get("day") and key not in done:
            rec = {
                "mode": state.get("mode"),
                "market": state.get("market"),
                "day": last.get("day"),
                "weather": last.get("weather"),
                "events": last.get("events"),
                "spots": last.get("spots"),
                "you": last.get("you"),
                "your_rep": (state.get("you") or {}).get("reputation"),
                "cash": (state.get("you") or {}).get("cash"),
                "trucks": state.get("trucks"),
                # the forecast that was on the table the night before this day
                "forecast_before": last_forecast,
                "recorded_at": state.get("server_time"),
            }
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            done.add(key)
            nspots = len(rec["spots"] or {})
            total = sum(s.get("customers", 0) for s in (rec["spots"] or {}).values())
            hungry = sum(s.get("hungry", 0) for s in (rec["spots"] or {}).values())
            print("archived day %s (%s): %d spots, %d customers, %d hungry" % (
                last.get("day"), rec["weather"], nspots, total, hungry))

        if state.get("phase") == "night" and state.get("forecast"):
            last_forecast = state["forecast"]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nrecorder stopped")

"""Food Truck Wars starter bot. Python 3.8+, standard library only.

    export NIGHTSHIFT_KEY=fk_your_key_here
    python3 bot.py              # play the live game (leave it running)
    python3 bot.py --practice   # play a private practice market and print your profit

It works, but it plays badly. Rewrite decide() below.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("NIGHTSHIFT_URL", "https://nightshift.joinbuilders.org").rstrip("/")
KEY = os.environ.get("NIGHTSHIFT_KEY", "")


def decide(state):
    """Return tomorrow's order. THIS IS THE PART TO REWRITE.

    Useful fields in state (see the API reference for all of them):
      state["order_for_day"]          the day this order is for
      state["rules"]                  budget ($500), cost per meal by quality, quality values, spots, price limits
      state["forecast"]               tomorrow: weather odds, events, crowd labels, expected_customers per spot
      state["you"]["cash"]            your money so far
      state["you"]["reputation"]      0-100 at each spot (starts at 50)
      state["last_day"]               the latest day: every spot's customers, hungry count, and each
                                      truck's price, quality, meals and sales; your line is last_day["you"]
      state["trucks"]                 every truck: name, company, cash, average reputation
    """
    rules = state["rules"]
    spot = random.choice(rules["spots"])        # anywhere at all
    quality = "standard"
    price = 10.0                                 # the same price every day
    meals = int(rules["budget"] // rules["costs"][quality])   # as much as the budget allows (133)
    return {"spot": spot, "price": price, "quality": quality, "meals": meals}


# ---------------------------------------------------------------- plumbing

def call(method, path, body=None, timeout=35):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {"error": str(e)}


def play(base):
    """Long-poll the state and send one order per night. Works for live, final and practice."""
    version, ordered_for, backoff = 0, None, 1
    while True:
        try:
            status, state = call("GET", base + "/state?since=%d" % version)
        except (urllib.error.URLError, OSError) as e:
            print("network error (%s), retrying in %ds" % (e, backoff))
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        backoff = 1
        if status == 429:
            time.sleep(2)
            continue
        if status != 200:
            print("error %d: %s" % (status, state.get("error")))
            if status in (401, 404):
                sys.exit(1)
            time.sleep(3)
            continue
        version = state["version"]
        if state["phase"] == "over" and state["mode"] == "practice":
            report(state)
            return state
        key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
        if state["phase"] == "night" and state.get("orders_open") and key != ordered_for:
            report(state)
            order = decide(state)
            status, res = call("POST", base + "/orders", order)
            if status == 200:
                ordered_for = key
                print("  ordered for day %s: %s" % (res.get("for_day"), order))
                if "state" in res:          # practice plays the day right away
                    version = 0
            else:
                print("  order rejected (%d): %s" % (status, res.get("error")))
                time.sleep(1)


def report(state):
    last = state.get("last_day") or {}
    mine = last.get("you") or {}
    where = state["mode"] + (" market %s" % state["market"] if state["mode"] == "final" else "")
    if mine.get("spot"):
        print("[%s] day %s: %s at %s, sold %s/%s, profit $%.2f, cash $%.2f" % (
            where, last.get("day"), mine.get("quality"), mine.get("spot"), mine.get("sold"),
            mine.get("meals"), mine.get("profit", 0), state["you"]["cash"]))
    elif last.get("day"):
        print("[%s] day %s: sat out, cash $%.2f" % (where, last.get("day"), state["you"]["cash"]))
    else:
        print("[%s] night before day %s, cash $%.2f" % (where, state.get("order_for_day"), state["you"]["cash"]))


def main():
    sys.stdout.reconfigure(line_buffering=True)   # show each line right away, even when piped
    ap = argparse.ArgumentParser(description="Food Truck Wars starter bot")
    ap.add_argument("--practice", action="store_true", help="play a private practice market")
    ap.add_argument("--seed", help="practice seed, for repeatable markets")
    ap.add_argument("--days", type=int, default=30, help="practice market length")
    args = ap.parse_args()
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first (you get it when you register).")
    if args.practice:
        body = {"days": args.days}
        if args.seed:
            body["seed"] = args.seed
        status, res = call("POST", "/api/v1/practice", body)
        if status not in (200, 201):
            sys.exit("could not start practice (%d): %s" % (status, res.get("error")))
        print("practice market %s started" % res["id"])
        final = play(res["base"])
        you = final["you"]
        print("practice over: cash $%.2f" % you["cash"])
        for t in sorted(final["trucks"], key=lambda t: -t["cash"]):
            print("  %-24s $%10.2f" % (t["truck_name"], t["cash"]))
    else:
        print("playing the live game at %s (Ctrl+C to stop)" % URL)
        play("/api/v1")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")

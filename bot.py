"""Food Truck Wars bot - Builders Night Shift.

A first-party client for https://nightshift.joinbuilders.org

    export NIGHTSHIFT_KEY=fk_...
    python bot.py --practice              # one 30-day practice market
    python bot.py --bench 8               # 8 seeded practice markets, averaged
    python bot.py                         # play the live game (leave running)

Strategy lives in decide(). Everything below it is plumbing.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

URL = os.environ.get("NIGHTSHIFT_URL", "https://nightshift.joinbuilders.org").rstrip("/")
KEY = os.environ.get("NIGHTSHIFT_KEY", "")
LOG_PATH = os.environ.get("NIGHTSHIFT_LOG", "run_log.jsonl")

QUALITIES = ("cheap", "standard", "premium")
QRANK = {"cheap": 0, "standard": 1, "premium": 2}
QVALUE = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
QCOST = {"cheap": 2.50, "standard": 3.75, "premium": 5.50}
QREP = {"cheap": -3.0, "standard": 2.0, "premium": 5.0}

# Per-spot customer priors, read off the docs ("Campus spots").
# limit_mean/sd: the customer's private price ceiling, normally distributed.
# qmin: P(minimum quality is cheap / standard / premium).
SPOTS = {
    "highst":   {"limit_mean": 12.0, "limit_sd": 3.0, "qmin": (0.50, 0.35, 0.15)},
    "oval":     {"limit_mean": 13.0, "limit_sd": 2.3, "qmin": (0.30, 0.50, 0.20)},
    "union":    {"limit_mean": 13.0, "limit_sd": 2.2, "qmin": (0.22, 0.53, 0.25)},
    "northres": {"limit_mean": 11.0, "limit_sd": 2.1, "qmin": (0.45, 0.45, 0.10)},
    "thompson": {"limit_mean": 12.0, "limit_sd": 2.3, "qmin": (0.45, 0.40, 0.15)},
    "stadium":  {"limit_mean": 13.0, "limit_sd": 2.6, "qmin": (0.25, 0.50, 0.25)},
}

# Tunables. Swept with --sweep.
P = {
    "rep_bonus": 5.0,        # $ per point of reputation gained, at full horizon
    "rep_horizon": 12.0,     # days over which reputation keeps paying
    "crowd_mults": ((0.70, 0.15), (0.85, 0.20), (1.00, 0.30), (1.15, 0.20), (1.35, 0.15)),
    "intruder_p": 0.45,      # chance an unpredicted rival shows up at my spot
    "risk_lambda": 0.55,     # how hard to penalise the bad tail
    "risk_tail": 0.30,       # what counts as the bad tail
    "price_step": 0.25,
    "shortlist": 18,
    "limit_cells": 18,
}


def repfactor(rep):
    return 0.7 + 0.6 * rep / 100.0


def norm_ppf(p):
    """Inverse normal CDF (Acklam's approximation, plenty accurate here)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def cells_for(spot, n_cells):
    """Discretise the customer population into (weight, min_rank, price_limit) cells."""
    cfg = SPOTS[spot]
    out = []
    for rank, pq in enumerate(cfg["qmin"]):
        if pq <= 0:
            continue
        for i in range(n_cells):
            q = (i + 0.5) / n_cells
            limit = cfg["limit_mean"] + cfg["limit_sd"] * norm_ppf(q)
            out.append((pq / n_cells, rank, max(1.0, limit)))
    return out


# ------------------------------------------------------------------ market model

class Truck:
    __slots__ = ("value", "price", "qrank", "cap")

    def __init__(self, value, price, qrank, cap):
        self.value = value
        self.price = price
        self.qrank = qrank
        self.cap = cap


def beats(a, b):
    """Customer prefers a over b: higher value, ties to the cheaper truck."""
    if a.value > b.value + 1e-9:
        return True
    if a.value < b.value - 1e-9:
        return False
    return a.price < b.price


def allocate(cells, trucks, n_customers):
    """Assign customers to trucks, respecting sell-outs. Returns sales per truck."""
    sales = [0.0] * len(trucks)
    cap = [t.cap for t in trucks]
    weights = [w * n_customers for (w, _, _) in cells]

    for _ in range(5):
        demand = defaultdict(float)
        choice = {}
        for ci, (_, rank, limit) in enumerate(cells):
            w = weights[ci]
            if w <= 1e-9:
                continue
            best = -1
            for ti, t in enumerate(trucks):
                if cap[ti] <= 1e-9 or t.qrank < rank or t.price > limit or t.value < 0:
                    continue
                if best < 0 or beats(t, trucks[best]):
                    best = ti
            if best >= 0:
                demand[best] += w
                choice[ci] = best
        if not demand:
            break
        frac = {}
        for ti, d in demand.items():
            frac[ti] = 1.0 if d <= cap[ti] else cap[ti] / d
        for ti, d in demand.items():
            take = d * frac[ti]
            sales[ti] += take
            cap[ti] -= take
        spill = False
        for ci, ti in choice.items():
            left = weights[ci] * (1.0 - frac[ti])
            if left > 1e-9:
                spill = True
            weights[ci] = left
        for ci in range(len(weights)):
            if ci not in choice:
                weights[ci] = 0.0
        if not spill:
            break
    return sales


# ------------------------------------------------------------------ rivals

def predict_rivals(state):
    """Assume every truck repeats yesterday's parking spot, price and quality."""
    me = state["you"]["id"]
    rep_avg = {t["id"]: t.get("reputation_avg", 50.0) for t in state.get("trucks", [])}
    last = state.get("last_day") or {}
    out = []
    for spot, info in (last.get("spots") or {}).items():
        for t in info.get("trucks", []):
            if t["id"] == me:
                continue
            q = t.get("quality", "standard")
            rep = rep_avg.get(t["id"], 50.0)
            out.append({
                "id": t["id"], "spot": spot, "price": float(t.get("price", 11.0)),
                "quality": q, "rep": rep,
                "meals": float(t.get("meals") or budget_max(q)),
            })
    if not out:
        # Day 1: nothing observed. Assume the field spreads over the busy spots.
        others = [t for t in state.get("trucks", []) if t["id"] != me]
        ring = ["highst", "oval", "union", "northres", "thompson", "stadium"]
        for i, t in enumerate(others):
            out.append({"id": t["id"], "spot": ring[i % len(ring)], "price": 11.0,
                        "quality": "standard", "rep": 50.0, "meals": 133.0})
    return out


def strongest_rival(rivals):
    """The most dangerous truck in the field: what it costs us if it moves in."""
    best = None
    for r in rivals:
        v = QVALUE[r["quality"]] * repfactor(r["rep"]) - r["price"]
        if best is None or v > best[0]:
            best = (v, r)
    if best is None:
        return Truck(QVALUE["standard"] - 10.5, 10.5, 1, 133.0)
    v, r = best
    return Truck(v, r["price"], QRANK[r["quality"]], r["meals"])


def rival_trucks_at(rivals, spot, intruder=None):
    ts = []
    for r in rivals:
        if r["spot"] != spot:
            continue
        v = QVALUE[r["quality"]] * repfactor(r["rep"]) - r["price"]
        ts.append(Truck(v, r["price"], QRANK[r["quality"]], r["meals"]))
    if intruder is not None:
        ts.append(intruder)
    return ts


def tail_mean(outcomes, frac):
    """Average profit over the worst `frac` of the probability mass (a CVaR)."""
    xs = sorted(outcomes)
    acc, tot = 0.0, 0.0
    for p, w in xs:
        take = min(w, frac - tot)
        if take <= 1e-12:
            break
        acc += p * take
        tot += take
    return acc / tot if tot > 0 else xs[0][0]


def budget_max(quality, budget=500.0):
    return int(budget // QCOST[quality])


# ------------------------------------------------------------------ the decision

def decide(state):
    """Pick tomorrow's spot, price, quality and meal count."""
    rules = state["rules"]
    budget = float(rules.get("budget", 500))
    spots = [s for s in rules.get("spots", list(SPOTS)) if s in SPOTS]
    pmin = float(rules.get("price_min", 1))
    pmax = float(rules.get("price_max", 30))
    fc = state.get("forecast") or {}
    expected = fc.get("expected_customers") or {}
    myrep = (state["you"].get("reputation") or {})
    rivals = predict_rivals(state)
    intruder = strongest_rival(rivals)

    days_left = max(1, int(state.get("days", 30)) - int(state.get("order_for_day", 1)) + 1)
    rep_weight = P["rep_bonus"] * min(days_left, P["rep_horizon"]) / P["rep_horizon"]

    scen = []
    for mult, w in P["crowd_mults"]:
        scen.append((mult, w * (1 - P["intruder_p"]), False))
        scen.append((mult, w * P["intruder_p"], True))

    # ---- stage 1: fast scan, ignoring capacity feedback, to shortlist candidates.
    shortlist = []
    cellcache = {}
    for spot in spots:
        n_base = float(expected.get(spot, 150))
        if n_base <= 0:
            continue
        cells = cellcache.setdefault(spot, cells_for(spot, P["limit_cells"]))
        rep = float(myrep.get(spot, 50.0))
        rf = repfactor(rep)
        # Best rival value per cell, per scenario-shape (only `extra` matters here).
        rbest = {}
        for extra in (False, True):
            rts = rival_trucks_at(rivals, spot, intruder if extra else None)
            arr = []
            for (_, rank, limit) in cells:
                best = None
                for t in rts:
                    if t.qrank < rank or t.price > limit or t.value < 0:
                        continue
                    if best is None or beats(t, best):
                        best = t
                arr.append(best)
            rbest[extra] = arr
        for quality in QUALITIES:
            qv = QVALUE[quality] * rf
            qr = QRANK[quality]
            cost = QCOST[quality]
            cap = budget // cost
            price = max(pmin, 4.0)
            while price <= min(pmax, qv + 0.01):
                myv = qv - price
                if myv < 0:
                    price += P["price_step"]
                    continue
                mine = Truck(myv, price, qr, cap)
                exp_sales = 0.0
                for mult, w, extra in scen:
                    arr = rbest[extra]
                    n = n_base * mult
                    d = 0.0
                    for ci, (cw, rank, limit) in enumerate(cells):
                        if qr < rank or price > limit:
                            continue
                        b = arr[ci]
                        if b is None or beats(mine, b):
                            d += cw * n
                    exp_sales += w * min(d, cap)
                profit = exp_sales * price - cap * cost
                shortlist.append((profit, spot, quality, price))
                price += P["price_step"]
    if not shortlist:
        return {"spot": "union", "price": 11.0, "quality": "standard", "meals": 133}
    shortlist.sort(key=lambda x: -x[0])
    shortlist = shortlist[:P["shortlist"]]

    # ---- stage 2: full simulation with sell-outs, then pick the meal count.
    best = None
    for _, spot, quality, price in shortlist:
        n_base = float(expected.get(spot, 150))
        cells = cellcache[spot]
        rep = float(myrep.get(spot, 50.0))
        myv = QVALUE[quality] * repfactor(rep) - price
        cost = QCOST[quality]
        hardcap = int(budget // cost)

        demands = []
        for mult, w, extra in scen:
            rts = rival_trucks_at(rivals, spot, intruder if extra else None)
            mine = Truck(myv, price, QRANK[quality], 1e9)
            sales = allocate(cells, [mine] + rts, n_base * mult)
            demands.append((sales[0], w))

        # Newsvendor over the demand scenarios, penalising the bad tail: a day
        # that sells nothing still pays for every meal, so hedge against it.
        for meals in meal_grid(demands, hardcap):
            outcomes = [(min(d, meals) * price - meals * cost, w) for d, w in demands]
            mean = sum(p * w for p, w in outcomes)
            downside = mean - tail_mean(outcomes, P["risk_tail"])
            rep_gain = rep_weight * QREP[quality] if meals > 0 else 0.0
            score = mean + rep_gain - P["risk_lambda"] * downside
            if best is None or score > best[0]:
                best = (score, mean, spot, quality, price, meals)

    _, profit, spot, quality, price, meals = best
    if profit < 0 and meals > 0:
        # Every option loses money; sitting out costs nothing.
        return {"spot": spot, "price": round(price, 2), "quality": quality, "meals": 0}
    return {"spot": spot, "price": round(price, 2), "quality": quality, "meals": int(meals)}


def meal_grid(demands, hardcap):
    """Candidate meal counts: around each demand scenario, plus the budget cap."""
    pts = {hardcap, 0}
    for d, _ in demands:
        for f in (0.9, 1.0, 1.1):
            pts.add(max(0, min(hardcap, int(round(d * f)))))
    return sorted(p for p in pts if p >= 0)


# ------------------------------------------------------------------ plumbing

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


def is_hangup(e):
    """A long-poll the server closed on us. Routine, not a fault."""
    s = str(e)
    return ("10054" in s or "forcibly closed" in s or "Connection reset" in s
            or "RemoteDisconnected" in s or "IncompleteRead" in s
            or isinstance(getattr(e, "reason", None), ConnectionResetError))


def log(rec):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def play(base, quiet=False):
    version, ordered_for, backoff, hangups = 0, None, 1, 0
    while True:
        try:
            status, state = call("GET", base + "/state?since=%d" % version)
        except (urllib.error.URLError, OSError) as e:
            # Reconnect immediately on a hang-up: in the final each night is
            # only open for 2 seconds, so backing off would cost us the day.
            if is_hangup(e):
                hangups += 1
                time.sleep(min(0.1 * hangups, 1.0))
                continue
            print("network error (%s), retrying in %ds" % (e, backoff))
            time.sleep(backoff)
            backoff = min(backoff * 2, 5)
            continue
        backoff, hangups = 1, 0
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
            return state
        key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
        if state["phase"] == "night" and state.get("orders_open") and key != ordered_for:
            if not quiet:
                report(state)
            t0 = time.time()
            order = decide(state)
            took = time.time() - t0
            status, res = call("POST", base + "/orders", order)
            if status == 200:
                ordered_for = key
                if not quiet:
                    print("  -> %s %s $%.2f x%d  (%.2fs)" % (
                        order["spot"], order["quality"], order["price"], order["meals"], took))
                log({"mode": state.get("mode"), "market": state.get("market"),
                     "day": state.get("order_for_day"), "order": order,
                     "cash": state["you"]["cash"], "decide_s": round(took, 3),
                     "last": (state.get("last_day") or {}).get("you")})
                if "state" in res:
                    version = 0
            else:
                print("  order rejected (%d): %s" % (status, res.get("error")))
                time.sleep(1)


def report(state):
    last = state.get("last_day") or {}
    mine = last.get("you") or {}
    where = state["mode"] + (" m%s" % state["market"] if state["mode"] == "final" else "")
    if mine.get("spot"):
        print("[%s] day %s: %s %s $%s sold %s/%s profit $%.2f | cash $%.2f" % (
            where, last.get("day"), mine.get("spot"), mine.get("quality"), mine.get("price"),
            mine.get("sold"), mine.get("meals"), mine.get("profit", 0), state["you"]["cash"]))
    elif last.get("day"):
        print("[%s] day %s: sat out | cash $%.2f" % (where, last.get("day"), state["you"]["cash"]))


def run_practice(seed=None, days=30, rivals=3, quiet=False):
    body = {"days": days, "rivals": rivals}
    if seed:
        body["seed"] = seed
    status, res = call("POST", "/api/v1/practice", body)
    if status not in (200, 201):
        print("could not start practice (%d): %s" % (status, res.get("error")))
        return None
    final = play(res["base"], quiet=quiet)
    return final


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="Food Truck Wars bot")
    ap.add_argument("--practice", action="store_true")
    ap.add_argument("--bench", type=int, metavar="N", help="run N seeded practice markets")
    ap.add_argument("--seed")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    args = ap.parse_args()
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first.")

    if args.bench:
        results = []
        for i in range(args.bench):
            seed = "bench-%d" % i
            final = run_practice(seed, args.days, args.rivals, quiet=True)
            if not final:
                continue
            cash = final["you"]["cash"]
            field = sorted(final["trucks"], key=lambda t: -t["cash"])
            rank = 1 + [t["id"] for t in field].index(final["you"]["id"])
            results.append((cash, rank))
            print("  %-10s cash $%9.2f  rank %d/%d  (best rival $%.2f)" % (
                seed, cash, rank, len(field),
                max(t["cash"] for t in field if t["id"] != final["you"]["id"])))
        if results:
            cashes = [c for c, _ in results]
            print("\naverage profit $%.2f   median $%.2f   wins %d/%d" % (
                statistics.mean(cashes), statistics.median(cashes),
                sum(1 for _, r in results if r == 1), len(results)))
    elif args.practice:
        final = run_practice(args.seed, args.days, args.rivals)
        if final:
            print("\npractice over: cash $%.2f" % final["you"]["cash"])
            for t in sorted(final["trucks"], key=lambda t: -t["cash"]):
                mark = " <-- you" if t["id"] == final["you"]["id"] else ""
                print("  %-24s $%10.2f%s" % (t["truck_name"], t["cash"], mark))
    else:
        print("playing live at %s (Ctrl+C to stop)" % URL)
        play("/api/v1")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")

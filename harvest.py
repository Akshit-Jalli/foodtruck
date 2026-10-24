"""Designed experiments against practice markets, to identify the demand curve.

    export NIGHTSHIFT_KEY=fk_...
    python harvest.py                      # 6 markets x 30 days -> probe_days.jsonl
    python harvest.py --markets 4 --days 30

Why this exists
---------------
Observational data from a *good* bot is useless for calibration: it always
parks near the same price, so the price-limit distribution is never probed.
This script throws that away and runs a factorial design instead. Our truck
deliberately sweeps (spot x quality x price) over a grid that runs from well
below any plausible price limit ($3) to the very top of what the value rule
allows (~$15.75 premium), at every spot, preferring spots the house bots
leave empty -- a spot where we are the only truck is the cleanest experiment
there is, because everyone who walks away reveals that their limit is under
our price or their minimum quality is over ours.

Meal counts are deliberately generous (1.8x the forecast crowd, capped by the
$500 budget) so that our own observations are almost never censored by a
sell-out. Practice cash is irrelevant; identification is not.

Output: one JSON line per spot per day in probe_days.jsonl, carrying the
crowd, the hungry count, the weather/events, the preceding forecast, and
every truck present with price/quality/meals/sold and its *reconstructed*
per-spot reputation (start 50; +quality delta on any day it sold >=1 meal
there, otherwise 3% of the gap back toward 50). The server happens to expose
reputation too, so each market also cross-checks the reconstruction.

SAFETY: this script only ever POSTs to a base path handed back by
POST /api/v1/practice. It refuses to touch the live endpoint.
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
OUT = os.environ.get("NIGHTSHIFT_PROBE", "probe_days.jsonl")

QUALITIES = ("cheap", "standard", "premium")
QVALUE = {"cheap": 8.0, "standard": 12.0, "premium": 16.0}
QCOST = {"cheap": 2.50, "standard": 3.75, "premium": 5.50}
QREP = {"cheap": -3.0, "standard": 2.0, "premium": 5.0}
DRIFT = 0.03

# Price ladders, per quality. A customer never buys at value < 0, and
# value = QVALUE * repfactor - price, so prices at or above QVALUE (at the
# baseline reputation of 50, where repfactor == 1) can never sell anything.
# Each ladder therefore runs from far below any plausible limit up to just
# under that mechanical ceiling.
LADDER = {
    "cheap":    (3.0, 4.0, 5.0, 6.0, 7.0, 7.75),
    "standard": (4.0, 5.5, 7.0, 8.5, 9.5, 10.5, 11.25, 11.75),
    "premium":  (5.0, 6.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.75),
}


def repfactor(rep):
    return 0.7 + 0.6 * rep / 100.0


# ------------------------------------------------------------------ transport

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
    s = str(e)
    return ("10054" in s or "forcibly closed" in s or "Connection reset" in s
            or "RemoteDisconnected" in s or "IncompleteRead" in s)


class TooManyThrottles(Exception):
    pass


class Session:
    """A throttled request pump. The live bot's traffic matters more than ours."""

    def __init__(self, delay=0.2, max_429=12):
        self.delay = delay
        self.max_429 = max_429
        self.throttles = 0
        self.requests = 0

    def __call__(self, method, path, body=None):
        backoff = 1.0
        hangups = 0
        for _ in range(40):
            time.sleep(self.delay)
            try:
                status, res = call(method, path, body)
            except (urllib.error.URLError, OSError) as e:
                if is_hangup(e) and hangups < 8:
                    hangups += 1
                    time.sleep(min(0.25 * hangups, 2.0))
                    continue
                print("  network error (%s), retry in %.0fs" % (e, backoff))
                time.sleep(backoff)
                backoff = min(backoff * 2, 15)
                continue
            self.requests += 1
            if status == 429:
                self.throttles += 1
                if self.throttles > self.max_429:
                    raise TooManyThrottles(
                        "%d HTTP 429s; backing off entirely so the live bot keeps priority"
                        % self.throttles)
                wait = min(2.0 * self.throttles, 20.0)
                print("  429 (#%d), sleeping %.0fs" % (self.throttles, wait))
                time.sleep(wait)
                continue
            return status, res
        raise TooManyThrottles("gave up on %s %s after 40 attempts" % (method, path))


# ------------------------------------------------------------------ reputation

class RepBook:
    """Deterministic reconstruction of every truck's per-spot reputation.

    Start at 50, then each day, per (truck, spot):
      * parked there and sold >=1 meal  -> move by the quality delta
      * parked there and sold nothing   -> unchanged
      * did not park there              -> close 3% of the gap back toward 50
    clamped to [0, 100] and rounded to 2dp.

    That middle case is a correction to the documented rule, which reads as
    though a no-sale day drifts like an absence. It does not: replaying 960
    truck-spot-days of harvested data, "parked but sold nothing holds steady"
    reproduces the server exactly (max error 0.01, i.e. display rounding),
    while drifting on a no-sale day misses on 175 of them by up to 12 points.
    """

    def __init__(self, spots):
        self.spots = list(spots)
        self.rep = {}

    def _row(self, truck_id):
        return self.rep.setdefault(truck_id, {s: 50.0 for s in self.spots})

    def of(self, truck_id, spot):
        return self._row(truck_id).get(spot, 50.0)

    def snapshot(self, truck_id):
        return dict(self._row(truck_id))

    def apply_day(self, day_spots):
        """day_spots: last_day["spots"]. Call once per resolved day, after recording."""
        parked = {}
        for spot, info in (day_spots or {}).items():
            for t in (info.get("trucks") or []):
                self._row(t["id"])
                parked[(t["id"], spot)] = (
                    QREP.get(t.get("quality", "standard"), 0.0)
                    if float(t.get("sold") or 0) >= 1 else 0.0)
        for tid, row in self.rep.items():
            for spot in self.spots:
                cur = row.get(spot, 50.0)
                if (tid, spot) in parked:
                    cur += parked[(tid, spot)]
                else:
                    cur += (50.0 - cur) * DRIFT
                row[spot] = round(max(0.0, min(100.0, cur)), 2)

    def check(self, server_trucks):
        """Largest absolute disagreement with the server's own reputation table."""
        worst, n = 0.0, 0
        for t in server_trucks or []:
            for spot, val in (t.get("reputation") or {}).items():
                if t["id"] in self.rep and spot in self.rep[t["id"]]:
                    worst = max(worst, abs(self.rep[t["id"]][spot] - float(val)))
                    n += 1
        return worst, n


# ------------------------------------------------------------------ the design

def design_points(spots):
    pts = []
    for spot in spots:
        for q in QUALITIES:
            for p in LADDER[q]:
                pts.append({"spot": spot, "quality": q, "price": p})
    return pts


class Design:
    """A shuffled factorial queue, biased toward spots the house bots vacate."""

    def __init__(self, spots, rng, lookahead=10):
        self.spots = list(spots)
        self.rng = rng
        self.lookahead = lookahead
        self.queue = []
        self.passes = 0
        self.cells = len(design_points(spots))
        self.refill()

    def refill(self):
        pts = design_points(self.spots)
        self.rng.shuffle(pts)
        self.queue.extend(pts)
        self.passes += 1

    def take(self, empty_spots, myrep):
        """Pick the next experiment. Prefers an uncontested spot, and skips a
        cell whose price is mechanically unsellable at our current reputation."""
        if len(self.queue) < self.lookahead:
            self.refill()
        window = self.queue[:self.lookahead]

        def sellable(pt):
            rf = repfactor(float(myrep.get(pt["spot"], 50.0)))
            return pt["price"] < QVALUE[pt["quality"]] * rf - 0.05

        for pick_empty in (True, False):
            for i, pt in enumerate(window):
                if pick_empty and pt["spot"] not in empty_spots:
                    continue
                if not sellable(pt):
                    continue
                return self.queue.pop(i)
        # Nothing in the window is sellable right now: take the head anyway and
        # let it record a genuine zero.
        return self.queue.pop(0)


def budget_max(quality, budget=500.0):
    return int(budget // QCOST[quality])


def meals_for(quality, expected, budget=500.0):
    """Enough meals that a sell-out is unlikely: a censored row teaches less."""
    want = int(round(1.8 * max(10.0, float(expected))))
    return max(20, min(budget_max(quality, budget), want))


# ------------------------------------------------------------------ one market

def run_market(sess, seed, days, rivals, out_f, verbose=True):
    status, res = sess("POST", "/api/v1/practice",
                       {"days": days, "rivals": rivals, "seed": seed})
    if status not in (200, 201):
        print("  could not start practice (%d): %s" % (status, res.get("error")))
        return 0, None
    base = res["base"]
    if not base.startswith("/api/v1/practice/"):
        raise SystemExit("refusing to drive a non-practice base path: %r" % base)
    pid = res.get("id") or base.rsplit("/", 1)[-1]

    rng = random.Random(seed)
    book = None
    design = None
    forecasts = {}
    recorded = set()
    rows = 0
    version = 0
    ordered_for = None
    placements = []

    while True:
        status, state = sess("GET", base + "/state?since=%d" % version)
        if status != 200:
            print("  state error %d: %s" % (status, state.get("error")))
            return rows, None
        version = state.get("version", version)
        spots = list((state.get("rules") or {}).get("spots") or [])
        if book is None:
            book = RepBook(spots)
            design = Design(spots, rng)
        me = state["you"]["id"]

        last = state.get("last_day") or {}
        day = last.get("day")
        if day and day not in recorded:
            recorded.add(day)
            rows += write_day(out_f, state, last, book, forecasts.get(day), pid, seed, me)
            book.apply_day(last.get("spots"))

        if state.get("phase") == "over":
            worst, n = book.check(state.get("trucks"))
            if verbose:
                print("  %-12s %2d days, %4d rows, cash $%.0f, rep check max err %.3f over %d pairs"
                      % (seed, len(recorded), rows, state["you"]["cash"], worst, n))
            return rows, {"seed": seed, "practice_id": pid, "days": len(recorded),
                          "rows": rows, "rep_max_err": worst, "rep_pairs": n,
                          "placements": placements}

        key = state.get("order_for_day")
        if state.get("phase") == "night" and state.get("orders_open") and key != ordered_for:
            fc = state.get("forecast") or {}
            forecasts[key] = fc
            expected = fc.get("expected_customers") or {}
            busy = set()
            for spot, info in ((last.get("spots")) or {}).items():
                if any(t["id"] != me for t in (info.get("trucks") or [])):
                    busy.add(spot)
            empty = [s for s in spots if s not in busy]
            myrep = state["you"].get("reputation") or {}
            pt = design.take(empty, myrep)
            order = {"spot": pt["spot"], "price": round(pt["price"], 2),
                     "quality": pt["quality"],
                     "meals": meals_for(pt["quality"], expected.get(pt["spot"], 80))}
            status, resp = sess("POST", base + "/orders", order)
            if status != 200:
                print("  order rejected (%d): %s" % (status, resp.get("error")))
                # Retry smaller next loop rather than spinning on the same cell.
                ordered_for = key
                continue
            ordered_for = key
            placements.append(dict(order, day=key, alone=order["spot"] in empty))
            version = 0


def write_day(out_f, state, last, book, forecast, pid, seed, me):
    """One JSON line per spot for a resolved day."""
    server_rep = {t["id"]: (t.get("reputation") or {}) for t in (state.get("trucks") or [])}
    expected = (forecast or {}).get("expected_customers") or {}
    events = last.get("events") or []
    my_rep_all = book.snapshot(me)
    n = 0
    for spot, info in (last.get("spots") or {}).items():
        trucks = []
        for t in (info.get("trucks") or []):
            trucks.append({
                "id": t["id"],
                "is_me": t["id"] == me,
                "price": float(t.get("price") or 0),
                "quality": t.get("quality", "standard"),
                "meals": float(t.get("meals") or 0),
                "sold": float(t.get("sold") or 0),
                # Reconstructed from history; this is the value that was in
                # force *during* this day, before today's update.
                "rep": round(book.of(t["id"], spot), 4),
            })
        rec = {
            "source": "probe",
            "seed": seed,
            "practice_id": pid,
            "day": last.get("day"),
            "spot": spot,
            "customers": float(info.get("customers") or 0),
            "hungry": float(info.get("hungry") or 0),
            "weather": last.get("weather"),
            "events": events,
            "event_here": [e for e in events if e.get("spot") == spot],
            "trucks": trucks,
            "my_rep": round(book.of(me, spot), 4),
            "my_rep_all": {k: round(v, 4) for k, v in my_rep_all.items()},
            "expected_customers": expected.get(spot),
            # post-day server reputation, kept only to audit the reconstruction
            "server_rep_after": {t["id"]: server_rep.get(t["id"], {}).get(spot)
                                 for t in (info.get("trucks") or [])},
        }
        out_f.write(json.dumps(rec) + "\n")
        n += 1
    return n


# ------------------------------------------------------------------ repair

def repair(path):
    """Recompute every reputation column in place, offline, from the recorded
    history. Needed after a change to the reputation rule; costs no requests.
    Prints how well the reconstruction matches the server's own table."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("nothing to repair in %s" % path)
        return

    by_market = {}
    for i, r in enumerate(rows):
        by_market.setdefault((r.get("seed"), r.get("practice_id")), []).append(i)

    worst, checked, mismatches = 0.0, 0, 0
    for key, idxs in by_market.items():
        spots = sorted({rows[i]["spot"] for i in idxs})
        book = RepBook(spots)
        me = None
        for i in idxs:
            for t in rows[i]["trucks"]:
                if t.get("is_me"):
                    me = t["id"]
        by_day = {}
        for i in idxs:
            by_day.setdefault(rows[i]["day"], []).append(i)
        for day in sorted(by_day):
            day_spots = {rows[i]["spot"]: {"trucks": rows[i]["trucks"]} for i in by_day[day]}
            snap = book.snapshot(me) if me else {}
            for i in by_day[day]:
                r = rows[i]
                for t in r["trucks"]:
                    t["rep"] = round(book.of(t["id"], r["spot"]), 4)
                if me:
                    r["my_rep"] = round(book.of(me, r["spot"]), 4)
                    r["my_rep_all"] = {k: round(v, 4) for k, v in snap.items()}
            book.apply_day(day_spots)
            # audit against the server's post-day table
            for i in by_day[day]:
                r = rows[i]
                for t in r["trucks"]:
                    srv = (r.get("server_rep_after") or {}).get(t["id"])
                    if srv is None:
                        continue
                    e = abs(book.of(t["id"], r["spot"]) - float(srv))
                    worst = max(worst, e)
                    checked += 1
                    if e > 0.02:
                        mismatches += 1

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    print("repaired %d rows across %d markets" % (len(rows), len(by_market)))
    print("reputation audit: %d/%d truck-spot-days disagree with the server, max abs error %.4f"
          % (mismatches, checked, worst))


# ------------------------------------------------------------------ entry point

def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="designed-experiment harvester")
    ap.add_argument("--repair", action="store_true",
                    help="recompute reputation columns in an existing file, no requests")
    ap.add_argument("--markets", type=int, default=6)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between requests")
    ap.add_argument("--prefix", default="harvest")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.repair:
        repair(args.out)
        return
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first.")

    sess = Session(delay=args.delay)
    total = 0
    summaries = []
    t0 = time.time()
    print("harvesting %d x %d days -> %s" % (args.markets, args.days, args.out))
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(args.markets):
            seed = "%s-%d" % (args.prefix, i)
            try:
                rows, summary = run_market(sess, seed, args.days, args.rivals, f)
            except TooManyThrottles as e:
                print("STOPPING: %s" % e)
                break
            total += rows
            if summary:
                summaries.append(summary)
            f.flush()

    alone = sum(1 for s in summaries for p in s["placements"] if p["alone"])
    placed = sum(len(s["placements"]) for s in summaries)
    worst = max([s["rep_max_err"] for s in summaries], default=0.0)
    print("\n%d spot-day rows from %d markets in %.0fs (%d requests, %d throttles)"
          % (total, len(summaries), time.time() - t0, sess.requests, sess.throttles))
    print("%d/%d placements were at an uncontested spot" % (alone, placed))
    print("reputation reconstruction vs server: max abs error %.4f" % worst)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")

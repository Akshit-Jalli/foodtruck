"""Food Truck Wars bot - Builders Night Shift.

A first-party client for https://nightshift.joinbuilders.org

    export NIGHTSHIFT_KEY=fk_...
    python bot.py --practice              # one 30-day practice market
    python bot.py --bench 8               # 8 seeded practice markets, averaged
    python bot.py                         # play the live game (leave running)

The decision engine is decide(); everything below `plumbing` is transport.

Model, in short. Each customer has a private price limit and a private
minimum quality, and buys from whichever truck they can afford that offers
the best `value = quality_value * reputation_factor - price`. So our sales
are not a demand curve we control: they are what is left after every rival's
offer is scored against ours, customer by customer, until somebody sells out.
decide() therefore (1) predicts where the rivals park and what they charge,
(2) reconstructs their per-spot reputation from public history, (3) simulates
the resulting scramble for every spot/quality/price we could pick, and
(4) chooses the meal count by newsvendor over the spread of outcomes, with a
penalty on the bad tail and credit for the reputation a day's trading builds.
"""
import argparse
import ast
import json
import math
import os
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
REP_SLOPE = 0.006          # d(reputation_factor)/d(reputation point)
REP_DRIFT = 0.03           # fraction of the gap to 50 closed per day away

# Per-spot customer priors, from the docs and refined by fit.py against
# observed play. limit_mean/sd describe the private price ceiling; qmin is
# P(minimum quality is cheap / standard / premium).
SPOTS = {
    "highst":   {"limit_mean": 12.0, "limit_sd": 3.0, "qmin": (0.50, 0.35, 0.15)},
    "oval":     {"limit_mean": 12.4, "limit_sd": 2.3, "qmin": (0.30, 0.50, 0.20)},
    "union":    {"limit_mean": 12.8, "limit_sd": 2.2, "qmin": (0.22, 0.47, 0.31)},
    "northres": {"limit_mean": 11.0, "limit_sd": 2.1, "qmin": (0.45, 0.45, 0.10)},
    "thompson": {"limit_mean": 12.0, "limit_sd": 2.3, "qmin": (0.45, 0.40, 0.15)},
    "stadium":  {"limit_mean": 13.0, "limit_sd": 2.6, "qmin": (0.25, 0.50, 0.25)},
}

# fit.py's maximum-likelihood priors, from 854 designed observations, cut
# out-of-sample demand error from 4.99 to 4.41 meals RMSE (better at 5 of 6
# spots) - and then lost $238/market over 8 paired seeds, 2 wins to 6. Better
# average demand prediction is not better decisions: profit turns on exactly
# where the price-limit curve falls away, not on the average fit. Not adopted.
FITTED_SPOTS = {
    "highst":   {"limit_mean": 12.12, "limit_sd": 3.16, "qmin": (0.50, 0.32, 0.18)},
    "oval":     {"limit_mean": 12.65, "limit_sd": 2.52, "qmin": (0.33, 0.48, 0.19)},
    "union":    {"limit_mean": 12.80, "limit_sd": 1.98, "qmin": (0.28, 0.41, 0.31)},
    "northres": {"limit_mean": 11.25, "limit_sd": 2.41, "qmin": (0.47, 0.42, 0.11)},
    "thompson": {"limit_mean": 12.00, "limit_sd": 2.83, "qmin": (0.41, 0.44, 0.15)},
    "stadium":  {"limit_mean": 13.50, "limit_sd": 2.63, "qmin": (0.34, 0.40, 0.27)},
}

# Both sets above, plus anything a calibration run registers, addressed by
# name so one harness can compare them. decide() re-reads P["priors"] every
# night, so `bench.py --ablate priors="'rmse'"` switches priors without an
# edit. Registered as copies: use_priors() rebinds SPOTS and must not alias.
# calibrate.py's decision-focused result: priors tuned to maximise realised
# profit through the optimiser rather than to minimise demand RMSE. Found by
# compass search over a 3-scalar family (blend toward the RMSE fit, a global
# shift on limit_mean, a global scale on limit_sd) against the cal-* seeds
# only, regularised toward "orig". theta = (alpha 0.125, delta +0.750, kappa 1.000):
# almost all of the movement is the $0.75 global shift in limit_mean.
# Train-seed cash $10595.35 against "orig"'s $10025.19 -- but that is the best of 24
# noisy evaluations on 8 shared seeds, so treat it as a candidate, not a
# result, until the hold-* gate says otherwise.
DFL_SPOTS = {
        "highst":    {"limit_mean": 12.77, "limit_sd": 3.02, "qmin": (0.5000, 0.3462, 0.1537)},
        "oval":      {"limit_mean": 13.18, "limit_sd": 2.33, "qmin": (0.3038, 0.4975, 0.1988)},
        "union":     {"limit_mean": 13.55, "limit_sd": 2.17, "qmin": (0.2275, 0.4625, 0.3100)},
        "northres":  {"limit_mean": 11.78, "limit_sd": 2.14, "qmin": (0.4525, 0.4462, 0.1013)},
        "thompson":  {"limit_mean": 12.75, "limit_sd": 2.37, "qmin": (0.4450, 0.4050, 0.1500)},
        "stadium":   {"limit_mean": 13.81, "limit_sd": 2.60, "qmin": (0.2609, 0.4869, 0.2522)},
}

PRIOR_SETS = {
    "orig": {s: dict(c) for s, c in SPOTS.items()},
    "rmse": {s: dict(c) for s, c in FITTED_SPOTS.items()},
    "dfl": {s: dict(c) for s, c in DFL_SPOTS.items()},
}
_ACTIVE_PRIORS = "orig"


def use_priors(name):
    """Make `name` the active customer priors. Returns the name."""
    global SPOTS, _ACTIVE_PRIORS
    if name not in PRIOR_SETS:
        raise KeyError("unknown prior set %r (have: %s)"
                       % (name, ", ".join(sorted(PRIOR_SETS))))
    SPOTS = {s: dict(c) for s, c in PRIOR_SETS[name].items()}
    _ACTIVE_PRIORS = name
    return name

P = {
    # which PRIOR_SETS entry decide() uses; re-read every night
    "priors": "orig",
    # risk
    "risk_lambda": 0.55,     # weight on the bad tail
    "risk_tail": 0.30,       # what counts as the bad tail
    # uncertainty
    "crowd_mults": ((0.70, 0.15), (0.85, 0.20), (1.00, 0.30), (1.15, 0.20), (1.35, 0.15)),
    "intruder_p": 0.45,      # chance an unpredicted rival shows up at our spot
    # Night one has no history to predict from. Measured: every day-1 rival
    # offer in the probe data sits at exactly this value at reputation 50.
    # None restores the old flat $11.00/standard guess, for ablation.
    "day1_iso_value": 1.50,
    # Rivals do not repeat their price to the cent, and winning the value
    # contest is a cliff: a rival shaving 25c can take every customer. So
    # carry a spread of rival prices and prefer a pitch that survives it.
    "rival_price_shifts": ((-0.60, 0.30), (0.0, 0.50), (0.45, 0.20)),
    # reputation
    "rep_revisit_floor": 0.25,
    # Dollars of score charged for parking where we have already been parking,
    # scaled by the fraction of the last ten days spent there. 0.0 = off, the
    # default. Exists to test whether the bot's concentration on one spot
    # (200 of 240 trading days at High Street) is costing it anything: the
    # rep-value term rewards revisiting a spot *because* we revisit it, which
    # is a feedback loop, and this is the knob that opposes it.
    "spot_repeat_penalty": 0.0,
    "rep_max_horizon": 15,
    "rep_discount": 0.97,
    "rep_bonus": 5.0,        # only used when use_repvalue is off
    "rep_horizon": 12.0,
    # search
    "time_budget_s": 1.0,    # the final allows 2s a night, including the round trip
    "price_step": 0.25,
    "shortlist": 18,
    "limit_cells": 18,
    # feature flags, for ablation
    "use_reptracker": True,     # reconstruct rivals' per-spot reputation
    "use_mobility": True,       # probabilistic rival locations
    "use_repvalue": True,       # value-function reputation credit
    "stage1_newsvendor": True,  # inventory-aware screening
    "dedup_intruder": True,     # do not clone a rival onto its own spot
    # Measured: costs ~3.6% against the house bots (0 wins in 8 paired seeds),
    # because it gives up margin to guard against undercutting they never try.
    # Kept for the live field, where rivals do move their prices.
    "use_price_jitter": False,
    "use_herding": True,        # rivals pile onto the night's obvious spot
    "herd_p": 0.40,
}


def repfactor(rep):
    return 0.7 + 0.6 * rep / 100.0


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def norm_ppf(p):
    """Inverse normal CDF (Acklam's approximation)."""
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
    """Discretise the customers into (weight, min_rank, price_limit) cells."""
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


def budget_max(quality, budget=500.0):
    return int(budget // QCOST[quality])


# Each spot's share of the crowd on an ordinary clear day, from the docs.
SHARES = {"highst": 0.24, "oval": 0.20, "union": 0.18,
          "northres": 0.16, "thompson": 0.14, "stadium": 0.08}


def expected_crowds(state, spots):
    """Tomorrow's crowd per spot, with a sane floor if the forecast is absent.

    A missing or all-zero forecast is a data fault, not an empty campus: the
    market is about 110 customers per truck over at least 4 trucks. Falling
    back to the documented shares beats both guessing zero (sit out for
    nothing) and the old blind 133-meal default (risk the whole budget).
    """
    exp = ((state.get("forecast") or {}).get("expected_customers")) or {}
    out = {s: float(exp.get(s) or 0.0) for s in spots}
    if sum(out.values()) > 0:
        return out
    trucks = max(4, len(state.get("trucks") or []) or 4)
    total = 110.0 * trucks
    return {s: total * SHARES.get(s, 1.0 / max(1, len(spots))) for s in spots}


# ------------------------------------------------------------------ memory

class Memory:
    """What we have seen. Rivals' per-spot reputation and where they park.

    Reputation is private in the state we receive - `trucks` only reports an
    average across all six spots - but its dynamics are deterministic and
    every truck's spot and quality is public in last_day. So we replay the
    rules and recover each rival's reputation at each spot exactly.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.rep = defaultdict(dict)        # truck id -> spot -> reputation
        self.spots = defaultdict(list)      # truck id -> [spot, newest last]
        self.mine = []                      # our own spots, newest last
        self.key = None
        self.days = 0

    def _market_key(self, state):
        return (state.get("mode"), state.get("market"))

    def observe_day(self, last, me=None):
        """Fold one completed day into memory. `last` is a last_day object."""
        spots = last.get("spots") or {}
        sold_at = {}
        for spot, info in spots.items():
            for t in info.get("trucks", []):
                tid = t.get("id")
                if not tid:
                    continue
                sold_at[(tid, spot)] = (float(t.get("sold") or 0), t.get("quality", "standard"))
                self.spots[tid].append(spot)
                if me and tid == me:
                    self.mine.append(spot)
        seen = {tid for tid, _ in sold_at}
        for tid in set(list(self.rep.keys())) | seen:
            cur = self.rep[tid]
            for spot in SPOTS:
                r = cur.get(spot, 50.0)
                hit = sold_at.get((tid, spot))
                if hit is None:
                    r += (50.0 - r) * REP_DRIFT     # away: drift back toward 50
                elif hit[0] >= 1:
                    r = clamp(r + QREP[hit[1]], 0.0, 100.0)
                # Parked but sold nothing: reputation does not move at all.
                # The docs say unvisited spots drift, which reads as though a
                # shut-out day drifts too. It does not. Replaying 960 observed
                # truck-days: drifting on a shut-out mismatches the server
                # 175 times (max 12.07 points), holding steady mismatches 0.
                cur[spot] = r
        self.days += 1

    def sync(self, state):
        """Called each night. Resets between markets, folds in the new day."""
        k = self._market_key(state)
        last = state.get("last_day") or {}
        day = last.get("day")
        seen = getattr(self, "_last_seen", None)
        # Every practice market reports mode=practice, market=1, so the key
        # alone cannot tell two of them apart and one market's rivals would
        # leak into the next. A day that runs backwards, or a market with no
        # history when we hold some, means this is somewhere new.
        fresh = ((day is None and self.days > 0)
                 or (day is not None and seen is not None and day < seen))
        if k != self.key or fresh:
            self.reset()
            self.key = k
            seen = None
        if day and day != seen:
            self._last_seen = day
            self.observe_day(last, me=(state.get("you") or {}).get("id"))

    def rep_at(self, tid, spot, fallback=50.0):
        if not P["use_reptracker"]:
            return fallback
        got = self.rep.get(tid)
        if not got or spot not in got:
            return fallback
        return got[spot]

    def spot_probs(self, tid, half_life=4.0):
        """Where this truck parks, weighted to recent days, Laplace-smoothed."""
        hist = self.spots.get(tid) or []
        if not hist:
            return None
        w = defaultdict(float)
        total = 0.0
        for i, spot in enumerate(reversed(hist)):
            wt = 0.5 ** (i / half_life)
            w[spot] += wt
            total += wt
        alpha = 1.5
        n = len(SPOTS)
        return {s: (w.get(s, 0.0) + alpha / n) / (total + alpha) for s in SPOTS}

    def my_revisit_p(self, spot):
        """How often we come back to a spot, used to value reputation there."""
        if not self.mine:
            return 0.5
        recent = self.mine[-10:]
        p = sum(1 for s in recent if s == spot) / float(len(recent))
        return max(P["rep_revisit_floor"], p)


MEM = Memory()

# Diagnostics only. Set to a dict and decide() fills it with every candidate
# it scored; left None in play so the hot path stays allocation-free.
TRACE = None


def preload_memory(path="days.jsonl", mode="live", market=1):
    """Replay the archived live days so a restarted bot keeps its history."""
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return 0
    rows = []
    with f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("mode") == mode and r.get("market") == market and r.get("day"):
                rows.append(r)
    rows.sort(key=lambda r: r["day"])
    MEM.reset()
    MEM.key = (mode, market)
    me = None
    for r in rows:
        if not me:
            me = ((r.get("you") or {}).get("id"))
        MEM.observe_day(r, me=me)
        MEM._last_seen = r["day"]
    return len(rows)


# ------------------------------------------------------------------ market model

class Truck:
    __slots__ = ("value", "price", "qrank", "cap", "tid")

    def __init__(self, value, price, qrank, cap, tid=None):
        self.value = value
        self.price = price
        self.qrank = qrank
        self.cap = cap
        self.tid = tid


def beats(a, b):
    """Customer prefers a over b: higher value, ties to the cheaper truck."""
    if a.value > b.value + 1e-9:
        return True
    if a.value < b.value - 1e-9:
        return False
    return a.price < b.price


def allocate(cells, trucks, n_customers):
    """Assign customers to trucks, respecting sell-outs. Sales per truck."""
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
        frac = {ti: (1.0 if d <= cap[ti] else cap[ti] / d) for ti, d in demand.items()}
        for ti, d in demand.items():
            take = d * frac[ti]
            sales[ti] += take
            cap[ti] -= take
        spill = False
        for ci in range(len(weights)):
            ti = choice.get(ci)
            if ti is None:
                weights[ci] = 0.0
                continue
            weights[ci] *= (1.0 - frac[ti])
            if weights[ci] > 1e-9:
                spill = True
        if not spill:
            break
    return sales


def tail_mean(outcomes, frac):
    """Average profit over the worst `frac` of the probability mass (CVaR)."""
    xs = sorted(outcomes)
    acc, tot = 0.0, 0.0
    for p, w in xs:
        take = min(w, frac - tot)
        if take <= 1e-12:
            break
        acc += p * take
        tot += take
    return acc / tot if tot > 0 else xs[0][0]


def meal_grid(demands, hardcap):
    """Expected profit is piecewise linear in meals, kinked at each scenario's
    demand, so the optimum sits on one of those points (or 0, or the cap)."""
    pts = {0, hardcap}
    for d, _ in demands:
        pts.add(int(clamp(round(d), 0, hardcap)))
    return sorted(pts)


def best_inventory(demands, price, cost, hardcap):
    """Newsvendor over the scenario demands, penalising the bad tail.
    Returns (meals, risk_adjusted_score, expected_profit)."""
    best = (0, -1e18, 0.0)
    for m in meal_grid(demands, hardcap):
        outcomes = [(min(d, m) * price - m * cost, w) for d, w in demands]
        mean = sum(p * w for p, w in outcomes)
        score = mean - P["risk_lambda"] * (mean - tail_mean(outcomes, P["risk_tail"]))
        if score > best[1]:
            best = (m, score, mean)
    return best


# ------------------------------------------------------------------ rivals

def predict_rivals(state):
    """Each rival's likely pitch tomorrow, from what we have seen them do."""
    me = (state.get("you") or {}).get("id")
    rep_avg = {t["id"]: t.get("reputation_avg", 50.0) for t in state.get("trucks", [])}
    last = state.get("last_day") or {}
    out = []
    for spot, info in (last.get("spots") or {}).items():
        for t in info.get("trucks", []):
            if t["id"] == me:
                continue
            q = t.get("quality", "standard")
            out.append({
                "id": t["id"], "spot": spot, "price": float(t.get("price", 11.0)),
                "quality": q,
                "rep": MEM.rep_at(t["id"], spot, rep_avg.get(t["id"], 50.0)),
                "meals": float(t.get("meals") or budget_max(q)),
                "probs": MEM.spot_probs(t["id"]),
            })
    if not out:
        # Nothing observed yet: night one. The field spreads over the busy
        # spots, and - measured, not assumed - every opening offer sits on an
        # exact iso-value line. All 24 rival offers on day 1 of the eight
        # harvested markets price to value +1.50 at reputation 50: cheap at
        # $6.50, standard at $10.50, premium at $14.50.
        #
        # This matters more than it looks. value is a single scalar and
        # beats() gives a tie to the cheaper truck, so a price that lands us
        # *exactly* on the rivals' value line loses every contested cell
        # rather than splitting them. Assuming $11.00/standard (value +1.00)
        # told the old model it won comfortably anywhere below $11, and the
        # true cliff is at $10.50. Measured cost of that one error: $246 on
        # night one, which was the whole of the fitted-priors regression.
        others = [t for t in state.get("trucks", []) if t["id"] != me]
        ring = ["highst", "oval", "union", "northres", "thompson", "stadium"]
        opening = QUALITIES if P["day1_iso_value"] is not None else None
        for i, t in enumerate(others):
            if opening:
                q = opening[i % len(opening)]
                price = QVALUE[q] * repfactor(50.0) - float(P["day1_iso_value"])
            else:
                q, price = "standard", 11.0
            out.append({"id": t["id"], "spot": ring[i % len(ring)], "price": price,
                        "quality": q, "rep": 50.0, "meals": 133.0, "probs": None})
    return out


def rival_presence(rivals, spot, herd=False, magnet=None):
    """(rival, probability it is at this spot) for everyone who plausibly is.

    Rivals do not choose independently. They all read the same forecast, so
    on a night with an obvious draw - a concert, a game day - the field piles
    onto it together and leaves the rest of campus unserved. Treating their
    choices as independent makes every spot look mildly contested and hides
    the empty ones, which on such a night are where the money is.
    """
    out = []
    for r in rivals:
        if herd and magnet:
            p = 0.85 if spot == magnet else 0.05
        elif P["use_mobility"] and r.get("probs"):
            p = r["probs"].get(spot, 0.0)
        else:
            p = 1.0 if r["spot"] == spot else 0.0
        if p >= 0.12:
            out.append((r, p))
    return out


def strongest_rival(rivals):
    """The most dangerous truck in the field: what it costs us if it moves in."""
    best = None
    for r in rivals:
        v = QVALUE[r["quality"]] * repfactor(r["rep"]) - r["price"]
        if best is None or v > best[0]:
            best = (v, r)
    if best is None:
        return Truck(QVALUE["standard"] - 10.5, 10.5, 1, 133.0, tid="__generic__")
    v, r = best
    return Truck(v, r["price"], QRANK[r["quality"]], r["meals"], tid=r["id"])


def rival_trucks_at(rivals, spot, intruder=None, shift=0.0, herd=False, magnet=None):
    """The field we expect to face at a spot.

    A rival that only sometimes parks here is carried at a fraction of its
    capacity: over the scenarios that is the competition it actually removes.
    `shift` moves every rival's price, to test whether our pitch survives a
    field that turns out cheaper than it was yesterday.
    """
    ts = []
    here = set()
    for r, p in rival_presence(rivals, spot, herd, magnet):
        rep = MEM.rep_at(r["id"], spot, r["rep"])
        price = max(1.0, r["price"] + shift)
        v = QVALUE[r["quality"]] * repfactor(rep) - price
        ts.append(Truck(v, price, QRANK[r["quality"]], r["meals"] * p, tid=r["id"]))
        here.add(r["id"])
    if intruder is not None:
        # Do not invent a second copy of a truck that is already standing here.
        if not (P["dedup_intruder"] and intruder.tid in here):
            ip = max(1.0, intruder.price + shift)
            ts.append(Truck(intruder.value + (intruder.price - ip), ip,
                            intruder.qrank, intruder.cap, tid=intruder.tid))
    return ts


# ------------------------------------------------------------------ the decision

def scenarios():
    """(crowd multiplier, probability, intruder?, rival price shift, herd?)."""
    shifts = P["rival_price_shifts"] if P["use_price_jitter"] else ((0.0, 1.0),)
    herds = ((False, 1.0 - P["herd_p"]), (True, P["herd_p"])) if P["use_herding"] else ((False, 1.0),)
    out = []
    for mult, w in P["crowd_mults"]:
        for ds, ws in shifts:
            for hd, wh in herds:
                out.append((mult, w * (1 - P["intruder_p"]) * ws * wh, False, ds, hd))
                out.append((mult, w * P["intruder_p"] * ws * wh, True, ds, hd))
    return out


def magnet_spot(state, spots):
    """The spot the whole field can see is tonight's draw."""
    fc = state.get("forecast") or {}
    for ev in (fc.get("events") or []):
        if ev.get("spot") in spots:
            return ev["spot"]
    exp = fc.get("expected_customers") or {}
    cand = [(exp.get(s, 0), s) for s in spots]
    if not cand:
        return None
    n, best = max(cand)
    total = sum(x for x, _ in cand) or 1.0
    # Only a genuinely outsized crowd pulls the field together.
    return best if n > 0.28 * total else None


def rival_best_per_cell(cells, trucks):
    """The offer each kind of customer would otherwise take."""
    arr = []
    for (_, rank, limit) in cells:
        best = None
        for t in trucks:
            if t.qrank < rank or t.price > limit or t.value < 0:
                continue
            if best is None or beats(t, best):
                best = t
        arr.append(best)
    return arr


def fast_demand(cells, rbest, mine, n):
    """Customers who would pick us, ignoring who runs out of food first."""
    d = 0.0
    for ci, (cw, rank, limit) in enumerate(cells):
        if mine.qrank < rank or mine.price > limit:
            continue
        b = rbest[ci]
        if b is None or beats(mine, b):
            d += cw * n
    return d


def best_at_rep(spot, cells, rbest_modal, n_modal, rep, budget, pmin, pmax):
    """Best achievable day-profit at a spot for a given reputation.

    Used twice - at today's reputation and at tomorrow's - so the worth of a
    reputation point is read off the game's own mechanics instead of guessed.
    """
    best = 0.0
    for quality in QUALITIES:
        qv = QVALUE[quality] * repfactor(rep)
        cost = QCOST[quality]
        hardcap = int(budget // cost)
        price = max(pmin, 5.0)
        top = min(pmax, qv)
        while price <= top + 1e-9:
            mine = Truck(qv - price, price, QRANK[quality], hardcap)
            d = fast_demand(cells, rbest_modal, mine, n_modal)
            m = int(clamp(round(d), 0, hardcap))
            profit = min(d, m) * price - m * cost
            if profit > best:
                best = profit
            price += 0.5
    return best


def decide(state):
    """Pick tomorrow's spot, price, quality and meal count.

    Runs to a deadline. Measured cost is ~0.2s even against 40 rivals, but in
    the final a night is open for only 2 seconds and a missed night is a day
    of zero profit, so an overrun returns the best option found so far rather
    than the best option possible.
    """
    if P["priors"] != _ACTIVE_PRIORS:
        use_priors(P["priors"])
    t0 = time.time()
    deadline = t0 + float(P["time_budget_s"])
    MEM.sync(state)
    rules = state["rules"]
    budget = float(rules.get("budget", 500))
    spots = [s for s in rules.get("spots", list(SPOTS)) if s in SPOTS]
    pmin = float(rules.get("price_min", 1))
    pmax = float(rules.get("price_max", 30))
    expected = expected_crowds(state, spots)
    myrep = (state["you"].get("reputation") or {})
    rivals = predict_rivals(state)
    intruder = strongest_rival(rivals)

    # order_for_day is null outside a night, and `days` can be missing.
    day_now = int(state.get("order_for_day") or state.get("day") or 1)
    days_left = max(1, int(state.get("days") or 30) - day_now + 1)
    scen = scenarios()
    magnet = magnet_spot(state, spots) if P["use_herding"] else None

    # ---- stage 1: screen every spot/quality/price without capacity feedback.
    shortlist = []
    cellcache, rbestcache, modal = {}, {}, {}
    for spot in spots:
        n_base = float(expected.get(spot, 150))
        if n_base <= 0:
            continue
        # Half the budget for screening; leave the rest for the real search.
        if shortlist and time.time() > t0 + 0.5 * (deadline - t0):
            break
        cells = cellcache.setdefault(spot, cells_for(spot, P["limit_cells"]))
        rep = float(myrep.get(spot, 50.0))
        rf = repfactor(rep)
        rbest = {k: rival_best_per_cell(
                     cells, rival_trucks_at(rivals, spot, intruder if k[0] else None,
                                            k[1], k[2], magnet))
                 for k in {(e, ds, hd) for _, _, e, ds, hd in scen}}
        rbestcache[spot] = rbest
        modal[spot] = (cells, rival_best_per_cell(
            cells, rival_trucks_at(rivals, spot, None, 0.0)), n_base)
        for quality in QUALITIES:
            qv = QVALUE[quality] * rf
            cost = QCOST[quality]
            hardcap = int(budget // cost)
            price = max(pmin, 4.0)
            top = min(pmax, qv)
            while price <= top + 1e-9:
                mine = Truck(qv - price, price, QRANK[quality], hardcap)
                demands = [(fast_demand(cells, rbest[(e, ds, hd)], mine, n_base * mult), w)
                           for mult, w, e, ds, hd in scen]
                if P["stage1_newsvendor"]:
                    _, score, _ = best_inventory(demands, price, cost, hardcap)
                else:
                    exp_sales = sum(w * min(d, hardcap) for d, w in demands)
                    score = exp_sales * price - hardcap * cost
                shortlist.append((score, spot, quality, price))
                price += P["price_step"]
    if not shortlist:
        return {"spot": "union", "price": 11.0, "quality": "standard", "meals": 133}
    shortlist.sort(key=lambda x: -x[0])
    shortlist = shortlist[:P["shortlist"]]
    if TRACE is not None:
        TRACE["priors"] = _ACTIVE_PRIORS
        TRACE["magnet"] = magnet
        TRACE["expected"] = dict(expected)
        TRACE["myrep"] = {s: float(myrep.get(s, 50.0)) for s in spots}
        TRACE["rivals"] = [{"id": r["id"], "spot": r["spot"], "price": r["price"],
                            "quality": r["quality"], "rep": r["rep"]} for r in rivals]
        TRACE["n_screened"] = len(shortlist)
        TRACE["shortlist"] = [{"stage1_score": sc, "spot": sp, "quality": q, "price": pr}
                              for sc, sp, q, pr in shortlist]

    # ---- how much is a day's reputation gain worth at each shortlisted spot?
    repvalue = {}
    if P["use_repvalue"]:
        horizon = int(min(days_left - 1, P["rep_max_horizon"]))
        for spot in {s for _, s, _, _ in shortlist}:
            cells, rbest0, n_base = modal[spot]
            rep = float(myrep.get(spot, 50.0))
            base = best_at_rep(spot, cells, rbest0, n_base, rep, budget, pmin, pmax)
            visits = sum(MEM.my_revisit_p(spot) * (P["rep_discount"] ** k)
                         for k in range(horizon))
            for quality in QUALITIES:
                nxt = clamp(rep + QREP[quality], 0.0, 100.0)
                if abs(nxt - rep) < 1e-9 or visits <= 0:
                    repvalue[(spot, quality)] = 0.0
                    continue
                gain = best_at_rep(spot, cells, rbest0, n_base, nxt, budget, pmin, pmax) - base
                repvalue[(spot, quality)] = gain * visits
    else:
        w = P["rep_bonus"] * min(days_left, P["rep_horizon"]) / P["rep_horizon"]
        for _, spot, quality, _ in shortlist:
            repvalue[(spot, quality)] = w * QREP[quality]

    # ---- stage 2: full simulation with sell-outs, then choose the stock.
    best = None
    for _, spot, quality, price in shortlist:
        if best is not None and time.time() > deadline:
            break
        n_base = float(expected.get(spot, 150))
        cells = cellcache[spot]
        rep = float(myrep.get(spot, 50.0))
        myv = QVALUE[quality] * repfactor(rep) - price
        cost = QCOST[quality]
        hardcap = int(budget // cost)

        demands = []
        for mult, w, extra, ds, hd in scen:
            rts = rival_trucks_at(rivals, spot, intruder if extra else None, ds, hd, magnet)
            mine = Truck(myv, price, QRANK[quality], 1e9)
            demands.append((allocate(cells, [mine] + rts, n_base * mult)[0], w))

        meals, score, mean = best_inventory(demands, price, cost, hardcap)
        if meals > 0:
            score += repvalue.get((spot, quality), 0.0)
        if P["spot_repeat_penalty"]:
            score -= P["spot_repeat_penalty"] * MEM.my_revisit_p(spot)
        if TRACE is not None:
            TRACE.setdefault("stage2", []).append({
                "spot": spot, "quality": quality, "price": price, "meals": meals,
                "score": score, "mean_profit": mean,
                "repvalue": repvalue.get((spot, quality), 0.0),
                "demand_mean": sum(w * d for d, w in demands),
                "demand_p10": min(d for d, _ in demands),
                "demand_p90": max(d for d, _ in demands),
            })
        if best is None or score > best[0]:
            best = (score, mean, spot, quality, price, meals)

    _, mean, spot, quality, price, meals = best
    if TRACE is not None:
        TRACE["chosen"] = {"spot": spot, "quality": quality, "price": round(price, 2),
                           "meals": int(meals), "mean_profit": mean,
                           "sat_out": bool(mean < 0 and meals > 0)}
        TRACE["n_stage2"] = len(TRACE.get("stage2") or [])
    if mean < 0 and meals > 0:
        # Every option loses money today; sitting out costs nothing.
        return {"spot": spot, "price": round(price, 2), "quality": quality, "meals": 0}
    return {"spot": spot, "price": round(price, 2), "quality": quality, "meals": int(meals)}


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


def play(base, quiet=False, history=None):
    version, ordered_for, backoff, hangups, errors = 0, None, 0.25, 0, 0
    seen_days = set()
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
            # Any other network fault - a DNS blip, the wifi dropping - gets
            # the same treatment. An exponential back-off to 5s cost us a
            # whole night once: the link returned and we were still asleep.
            # We cannot reach the server at all here, so retrying hard is
            # not hammering it, and a night is only open for 5 minutes.
            errors += 1
            if errors <= 3 or errors % 40 == 0:
                print("network error (%s), retry %d" % (e, errors))
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 1.5)
            continue
        if errors:
            print("network recovered after %d failed attempt(s)" % errors)
        backoff, hangups, errors = 0.25, 0, 0
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
        day_done = (state.get("last_day") or {}).get("day")
        if day_done and day_done not in seen_days:
            seen_days.add(day_done)
            mine = state["last_day"].get("you") or {}
            if history is not None:
                row = dict(mine)
                row["day"] = day_done
                row["cash"] = state["you"]["cash"]
                history.append(row)
            # Sitting out is almost always a fault, not a choice: say so
            # loudly rather than letting a silent zero slide past.
            if mine.get("status") == "no_order" and not quiet:
                print("!! MISSED DAY %s - no order reached the server" % day_done)
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


def run_practice(seed=None, days=30, rivals=3, quiet=False, history=None):
    body = {"days": days, "rivals": rivals}
    if seed:
        body["seed"] = seed
    status, res = call("POST", "/api/v1/practice", body)
    if status not in (200, 201):
        print("could not start practice (%d): %s" % (status, res.get("error")))
        return None
    return play(res["base"], quiet=quiet, history=history)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="Food Truck Wars bot")
    ap.add_argument("--practice", action="store_true")
    ap.add_argument("--bench", type=int, metavar="N")
    ap.add_argument("--seed")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="override a tunable, e.g. --set use_mobility=False")
    args = ap.parse_args()
    if not KEY:
        sys.exit("Set NIGHTSHIFT_KEY first.")
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k in P:
            P[k] = ast.literal_eval(v)
        else:
            sys.exit("unknown tunable %r" % k)

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
        n = preload_memory()
        print("playing live at %s (Ctrl+C to stop)" % URL)
        if n:
            print("replayed %d archived days into memory" % n)
        play("/api/v1")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")

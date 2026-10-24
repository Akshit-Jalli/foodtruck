"""Shared harness for the decision-focused calibration experiments.

Everything here talks only to POST /api/v1/practice and the market handles it
returns. The live endpoint is unreachable from this module by construction:
`guard()` wraps bot.call so any path outside the practice prefix raises.

    import ftlab
    ftlab.guard()
    base = ftlab.new_market("tune-0")
    st   = ftlab.night_state(base)          # day-1 night, orders open
    ftlab.post_order(base, {...})
    row  = ftlab.resolved(base, day=1)      # what that order actually earned

Seed sets are kept apart on purpose (step 6 of the brief):
    TRAIN  cal-*    the only seeds the calibration search may see
    VAL    tune-*   model selection between finalists
    HOLD   hold-*   untouched; the merge gate
"""
import json
import os
import time
import urllib.error
import urllib.request

import bot

def _load_env(path=".env.local"):
    """Fill missing NIGHTSHIFT_* vars from .env.local, which is gitignored."""
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return
    with f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"")
            if k.startswith("NIGHTSHIFT_") and not os.environ.get(k):
                os.environ[k] = v


_load_env()

URL = os.environ.get("NIGHTSHIFT_URL", "https://nightshift.joinbuilders.org").rstrip("/")
KEY = os.environ.get("NIGHTSHIFT_KEY", "")
PRACTICE_PREFIX = "/api/v1/practice"

# bot was imported before _load_env() ran, so its module-level KEY is stale.
bot.KEY = bot.KEY or KEY
bot.URL = URL

TRAIN = ["cal-%d" % i for i in range(8)]
VAL = ["tune-%d" % i for i in range(8)]
HOLD = ["hold-%d" % i for i in range(8)]


def guard():
    """Make the live endpoint unreachable through bot.call."""
    if getattr(bot.call, "_ftlab_guarded", False):
        return
    original = bot.call

    def guarded(method, path, body=None, timeout=35, _o=original):
        if not str(path).startswith(PRACTICE_PREFIX):
            raise RuntimeError("ftlab refused %s %s: practice markets only" % (method, path))
        return _o(method, path, body, timeout)

    guarded._ftlab_guarded = True
    bot.call = guarded


def _req(method, path, body=None, timeout=35):
    if not path.startswith(PRACTICE_PREFIX):
        raise RuntimeError("ftlab refused %s %s: practice markets only" % (method, path))
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


def call(method, path, body=None, timeout=35, tries=6):
    """One request, retrying hang-ups and 429s the way bot.play does."""
    backoff = 1.0
    for attempt in range(tries):
        try:
            status, res = _req(method, path, body, timeout)
        except (urllib.error.URLError, OSError) as e:
            if attempt == tries - 1:
                raise
            time.sleep(min(backoff, 5.0))
            backoff *= 2
            continue
        if status == 429:
            time.sleep(2.0)
            continue
        return status, res
    return status, res


def new_market(seed, days=30, rivals=3):
    status, res = call("POST", PRACTICE_PREFIX, {"days": days, "rivals": rivals, "seed": seed})
    if status not in (200, 201):
        raise RuntimeError("could not start practice %r (%d): %s" % (seed, status, res.get("error")))
    return res["base"]


def state(base, since=0, timeout=35):
    status, st = call("GET", base + "/state?since=%d" % since, timeout=timeout)
    if status != 200:
        raise RuntimeError("state %s failed (%d): %s" % (base, status, st.get("error")))
    return st


def night_state(base, want_day=None, deadline_s=60.0):
    """Block until orders are open, and return that state."""
    t0 = time.time()
    version = 0
    while time.time() - t0 < deadline_s:
        st = state(base, version)
        version = st.get("version", version)
        if st.get("phase") == "over":
            return st
        if st.get("phase") == "night" and st.get("orders_open"):
            if want_day is None or st.get("order_for_day") == want_day:
                return st
        time.sleep(0.05)
    raise RuntimeError("no open night on %s within %.0fs" % (base, deadline_s))


def post_order(base, order):
    status, res = call("POST", base + "/orders", order)
    if status != 200:
        raise RuntimeError("order rejected (%d): %s" % (status, res.get("error")))
    return res


def resolved(base, day, deadline_s=60.0):
    """Block until `day` has been played, and return our row for it."""
    t0 = time.time()
    version = 0
    while time.time() - t0 < deadline_s:
        st = state(base, version)
        version = st.get("version", version)
        last = st.get("last_day") or {}
        if last.get("day") == day:
            row = dict(last.get("you") or {})
            row["day"] = day
            row["cash"] = (st.get("you") or {}).get("cash")
            row["weather"] = last.get("weather")
            row["events"] = [e.get("name") for e in last.get("events") or []]
            return row
        if st.get("phase") == "over":
            return None
        time.sleep(0.05)
    raise RuntimeError("day %s never resolved on %s" % (day, base))


def decide_traced(st, priors, time_budget_s=60.0):
    """decide() under a named prior set, with the full candidate trace.

    The time budget is lifted so the deadline never truncates the search:
    otherwise a slow night, not the priors, decides what got evaluated.
    """
    bot.MEM.reset()
    saved_priors, saved_budget = bot.P["priors"], bot.P["time_budget_s"]
    bot.P["priors"], bot.P["time_budget_s"] = priors, time_budget_s
    bot.TRACE = {}
    try:
        order = bot.decide(st)
        trace = bot.TRACE
    finally:
        bot.TRACE = None
        bot.P["priors"], bot.P["time_budget_s"] = saved_priors, saved_budget
    trace["order"] = order
    return order, trace


def register_priors(name, spots):
    """Make an arbitrary prior dict addressable by name."""
    bot.PRIOR_SETS[name] = {s: {"limit_mean": float(c["limit_mean"]),
                                "limit_sd": float(c["limit_sd"]),
                                "qmin": tuple(float(x) for x in c["qmin"])}
                            for s, c in spots.items()}
    return name

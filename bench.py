"""Paired, seeded benchmark harness for Food Truck Wars engines.

Runs the same seed through every engine under test (`bot`, `baseline`, ...),
so the two engines face identical weather / events / crowds, then reports the
per-seed differences plus aggregate quality and risk metrics.

    export NIGHTSHIFT_KEY=fk_...
    python bench.py --n 8 --seeds tune                       # bot vs baseline
    python bench.py --n 8 --seeds hold --engines bot         # held-out check
    python bench.py --n 6 --ablate risk_lambda=0.0 --label no_risk

SAFETY: this only ever posts orders to practice markets created by
POST /api/v1/practice. It never touches the live endpoint, and it runs one
market at a time so it cannot crowd out a live bot's requests.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as _dt
import importlib
import io
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LIVE_ORDERS_PATH = "/api/v1/orders"      # the one path we must never post to
PRACTICE_PREFIX = "/api/v1/practice"     # the only path prefix we allow


# ------------------------------------------------------------------ utilities

class Tee(io.TextIOBase):
    """Forward engine output to the console (indented) and keep a copy."""

    def __init__(self, real, prefix="    | "):
        self.real = real
        self.prefix = prefix
        self.buf = []
        self._line = ""

    def write(self, s):
        self.buf.append(s)
        self._line += s
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            try:
                self.real.write(self.prefix + line + "\n")
                self.real.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass

    def text(self):
        return "".join(self.buf)


def quantile(xs, q):
    """Linear-interpolation quantile; xs need not be sorted."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def worst_decile_mean(xs):
    """CVaR-style: mean of the worst 10% of runs (at least one run)."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(1, int(round(len(s) * 0.10)))
    return float(statistics.fmean(s[:k]))


def _betacf(a, b, x):
    tiny, eps, itmax = 1e-30, 3e-12, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided p-value for Student's t."""
    if df <= 0:
        return None
    return float(_betai(0.5 * df, 0.5, df / (df + t * t)))


# 95% two-sided t critical values, df 1..30 (df > 30 -> 1.96 is close enough).
_TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
          14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
          20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
          26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def sign_test_p(wins, losses):
    """Exact two-sided sign test; ties are dropped before calling."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def split_top_level(s, sep=","):
    """Split on `sep`, ignoring separators inside (), [], {} or quotes."""
    out, depth, cur, quote = [], 0, [], None
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [p.strip() for p in out if p.strip()]


def parse_ablation(spec):
    """'a=1,b=(2,3)' -> {'a': 1, 'b': (2, 3)} using ast.literal_eval."""
    out = {}
    if not spec:
        return out
    for part in split_top_level(spec):
        if "=" not in part:
            raise ValueError("bad --ablate entry %r (want KEY=VALUE)" % part)
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError) as e:
            raise ValueError("could not literal_eval %r for key %r: %s" % (v, k, e))
    return out


def safe_label(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "bench"


# ------------------------------------------------------------------ engine runs

def load_engine(name):
    mod = importlib.import_module(name)
    for attr in ("run_practice", "P", "LOG_PATH"):
        if not hasattr(mod, attr):
            raise SystemExit("engine %r has no %r - not a valid engine module" % (name, attr))
    # Never let an engine append to the live run_log.jsonl.
    mod.LOG_PATH = os.devnull

    # Hard safety net: this harness must never touch the live game. Wrap the
    # engine's HTTP helper so any request outside /api/v1/practice/... raises
    # instead of reaching the server. A live POST /api/v1/orders would corrupt
    # the real run, so we refuse it here rather than trusting call sites.
    if hasattr(mod, "call") and not getattr(mod.call, "_bench_guarded", False):
        original = mod.call

        def guarded(method, path, body=None, timeout=35, _o=original, _n=name):
            if not str(path).startswith(PRACTICE_PREFIX):
                raise RuntimeError(
                    "bench refused %s %s from engine %r: only %s* is allowed "
                    "(live endpoint %s is off limits)"
                    % (method, path, _n, PRACTICE_PREFIX, LIVE_ORDERS_PATH))
            return _o(method, path, body, timeout)

        guarded._bench_guarded = True
        mod.call = guarded
    return mod


def run_one(mod, seed, days, rivals, log_path, timeout):
    """Run one practice market. Returns (final_state|None, history, stdout, error)."""
    history = []
    if os.path.exists(log_path):
        os.remove(log_path)
    mod.LOG_PATH = log_path

    box = {}

    def worker():
        try:
            box["final"] = mod.run_practice(seed=seed, days=days, rivals=rivals,
                                            quiet=True, history=history)
        except BaseException as e:  # noqa: BLE001 - never let one seed kill the bench
            box["error"] = "%s: %s" % (type(e).__name__, e)

    real_stdout = sys.stdout
    tee = Tee(real_stdout)
    sys.stdout = tee
    try:
        if timeout and timeout > 0:
            th = threading.Thread(target=worker, daemon=True)
            th.start()
            th.join(timeout)
            timed_out = th.is_alive()
        else:
            worker()
            timed_out = False
    finally:
        sys.stdout = real_stdout

    if timed_out:
        return None, history, tee.text(), "timeout after %ds" % timeout
    return box.get("final"), history, tee.text(), box.get("error")


def read_decide_times(log_path):
    out = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("decide_s") is not None:
                    out.append(float(rec["decide_s"]))
    except OSError:
        pass
    return out


def traded(row):
    m = row.get("meals")
    return m is not None and float(m) > 0


def summarize_run(engine, seed, final, history, decide_times):
    """Collapse one market into a flat record of metrics."""
    cash = float(final["you"]["cash"])
    me = final["you"]["id"]
    field = sorted(final.get("trucks", []), key=lambda t: -float(t.get("cash", 0.0)))
    ids = [t["id"] for t in field]
    rank = ids.index(me) + 1 if me in ids else None
    rivals_cash = [float(t["cash"]) for t in field if t["id"] != me]

    trading, stockouts = 0, 0
    unsold_units, waste_cost, total_profit = 0.0, 0.0, 0.0
    sold_total, meals_total, revenue_total = 0.0, 0.0, 0.0
    per_spot = defaultdict(lambda: {"visits": 0, "profit": 0.0, "sold": 0.0,
                                    "meals": 0.0, "stockouts": 0, "revenue": 0.0})
    curve = {}
    for row in history:
        day = row.get("day")
        profit = float(row.get("profit") or 0.0)
        total_profit += profit
        if day is not None:
            curve[int(day)] = profit
        if not traded(row):
            continue
        meals = float(row["meals"])
        sold = float(row.get("sold") or 0.0)
        cost = float(row.get("cost") or 0.0)
        trading += 1
        meals_total += meals
        sold_total += sold
        revenue_total += float(row.get("revenue") or 0.0)
        unsold = max(0.0, meals - sold)
        unsold_units += unsold
        if meals > 0 and cost > 0:
            waste_cost += unsold * (cost / meals)
        so = sold >= meals - 1e-9
        if so:
            stockouts += 1
        spot = row.get("spot") or "?"
        ps = per_spot[spot]
        ps["visits"] += 1
        ps["profit"] += profit
        ps["sold"] += sold
        ps["meals"] += meals
        ps["revenue"] += float(row.get("revenue") or 0.0)
        ps["stockouts"] += 1 if so else 0

    n_days = len(history)
    return {
        "engine": engine,
        "seed": seed,
        "ok": True,
        "final_cash": cash,
        "rank": rank,
        "field_size": len(field),
        "best_rival_cash": max(rivals_cash) if rivals_cash else None,
        "days_recorded": n_days,
        "trading_days": trading,
        "sat_out_days": n_days - trading,
        "profit_from_history": total_profit,
        "stockout_days": stockouts,
        "stockout_rate": (stockouts / trading) if trading else None,
        "unused_units_total": unsold_units,
        "unused_units_per_trading_day": (unsold_units / trading) if trading else None,
        "wasted_cost_total": waste_cost,
        "wasted_cost_per_trading_day": (waste_cost / trading) if trading else None,
        "meals_total": meals_total,
        "sold_total": sold_total,
        "revenue_total": revenue_total,
        "sell_through": (sold_total / meals_total) if meals_total else None,
        "profit_curve": [curve.get(d) for d in range(1, (max(curve) + 1) if curve else 1)],
        "per_spot": {k: dict(v) for k, v in per_spot.items()},
        "decide_s_mean": statistics.fmean(decide_times) if decide_times else None,
        "decide_s_max": max(decide_times) if decide_times else None,
        "decide_s_n": len(decide_times),
    }


# ------------------------------------------------------------------ aggregation

def aggregate(runs):
    """Aggregate a list of per-run records for one engine."""
    ok = [r for r in runs if r.get("ok")]
    if not ok:
        return {"n": 0}
    cash = [r["final_cash"] for r in ok]
    ranks = [r["rank"] for r in ok if r["rank"]]
    trading = sum(r["trading_days"] for r in ok)
    stock = sum(r["stockout_days"] for r in ok)
    unsold = sum(r["unused_units_total"] for r in ok)
    waste = sum(r["wasted_cost_total"] for r in ok)
    meals = sum(r["meals_total"] for r in ok)
    sold = sum(r["sold_total"] for r in ok)

    dts = []
    for r in ok:
        if r["decide_s_mean"] is not None:
            dts.append((r["decide_s_mean"], r["decide_s_n"]))
    dmean = (sum(m * n for m, n in dts) / sum(n for _, n in dts)) if dts else None
    dmax = max((r["decide_s_max"] for r in ok if r["decide_s_max"] is not None), default=None)

    spots = defaultdict(lambda: {"visits": 0, "profit": 0.0, "sold": 0.0,
                                 "meals": 0.0, "stockouts": 0, "revenue": 0.0})
    for r in ok:
        for spot, v in r["per_spot"].items():
            d = spots[spot]
            for k in ("visits", "profit", "sold", "meals", "stockouts", "revenue"):
                d[k] += v.get(k, 0)
    per_spot = {}
    for spot, v in sorted(spots.items(), key=lambda kv: -kv[1]["profit"]):
        per_spot[spot] = dict(v)
        per_spot[spot]["profit_per_visit"] = v["profit"] / v["visits"] if v["visits"] else None
        per_spot[spot]["stockout_rate"] = v["stockouts"] / v["visits"] if v["visits"] else None

    maxlen = max((len(r["profit_curve"]) for r in ok), default=0)
    curve = []
    for i in range(maxlen):
        vals = [r["profit_curve"][i] for r in ok
                if i < len(r["profit_curve"]) and r["profit_curve"][i] is not None]
        curve.append({"day": i + 1,
                      "mean_profit": statistics.fmean(vals) if vals else None,
                      "median_profit": statistics.median(vals) if vals else None,
                      "n": len(vals)})

    return {
        "n": len(ok),
        "final_cash_mean": statistics.fmean(cash),
        "final_cash_median": statistics.median(cash),
        "final_cash_sd": statistics.stdev(cash) if len(cash) > 1 else 0.0,
        "final_cash_min": min(cash),
        "final_cash_max": max(cash),
        "final_cash_p10": quantile(cash, 0.10),
        "final_cash_worst_decile_mean": worst_decile_mean(cash),
        "win_rate": (sum(1 for r in ranks if r == 1) / len(ranks)) if ranks else None,
        "wins": sum(1 for r in ranks if r == 1),
        "mean_rank": statistics.fmean(ranks) if ranks else None,
        "trading_days": trading,
        "sat_out_days": sum(r["sat_out_days"] for r in ok),
        "stockout_rate": (stock / trading) if trading else None,
        "unused_units_per_trading_day": (unsold / trading) if trading else None,
        "unused_units_per_run": unsold / len(ok),
        "wasted_cost_per_trading_day": (waste / trading) if trading else None,
        "wasted_cost_per_run": waste / len(ok),
        "sell_through": (sold / meals) if meals else None,
        "decide_s_mean": dmean,
        "decide_s_max": dmax,
        "per_spot": per_spot,
        "profit_curve": curve,
        "seeds": [r["seed"] for r in ok],
        "final_cash_by_seed": {r["seed"]: r["final_cash"] for r in ok},
    }


def paired(runs_by_engine, engines):
    """Per-seed differences for every ordered engine pair (a vs b)."""
    out = []
    for i, a in enumerate(engines):
        for b in engines[i + 1:]:
            ra = {r["seed"]: r for r in runs_by_engine.get(a, []) if r.get("ok")}
            rb = {r["seed"]: r for r in runs_by_engine.get(b, []) if r.get("ok")}
            seeds = [s for s in ra if s in rb]
            seeds.sort()
            if not seeds:
                continue
            diffs = [ra[s]["final_cash"] - rb[s]["final_cash"] for s in seeds]
            wins_a = sum(1 for d in diffs if d > 1e-9)
            wins_b = sum(1 for d in diffs if d < -1e-9)
            ties = len(diffs) - wins_a - wins_b
            n = len(diffs)
            mean = statistics.fmean(diffs)
            sd = statistics.stdev(diffs) if n > 1 else 0.0
            se = sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
            t = (mean / se) if se > 0 else None
            tcrit = _TCRIT.get(n - 1, 1.96) if n > 1 else None
            out.append({
                "a": a, "b": b, "n_paired": n,
                "per_seed": [{"seed": s,
                              "a_cash": ra[s]["final_cash"],
                              "b_cash": rb[s]["final_cash"],
                              "diff": ra[s]["final_cash"] - rb[s]["final_cash"]}
                             for s in seeds],
                "mean_diff": mean,
                "median_diff": statistics.median(diffs),
                "sd_diff": sd,
                "se_diff": se,
                "ci95_diff": [mean - tcrit * se, mean + tcrit * se] if tcrit else None,
                "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
                "paired_t": t,
                "paired_t_df": n - 1 if n > 1 else None,
                "paired_t_p": t_two_sided_p(t, n - 1) if t is not None and n > 1 else None,
                "sign_test_p": sign_test_p(wins_a, wins_b),
                "note": ("Small n: read the per-seed diffs, not the p-value. "
                         "Tests are paired, n=%d (%d non-tied)." % (n, wins_a + wins_b)),
            })
    return out


# ------------------------------------------------------------------ reporting

def fmt(v, spec="%.2f", dash="-"):
    return dash if v is None else (spec % v)


def print_report(summary):
    cfg = summary["config"]
    print()
    print("=" * 94)
    print("BENCH %s   seeds=%s n=%d days=%d rivals=%d" % (
        summary["label"], cfg["seed_set"], cfg["n"], cfg["days"], cfg["rivals"]))
    if cfg["ablate"]:
        print("ablation on bot.P: %s" % json.dumps(cfg["ablate"]))
    print("=" * 94)

    hdr = ("%-10s %4s %10s %10s %10s %10s %6s %5s %7s %7s %8s %8s" % (
        "engine", "n", "mean$", "median$", "wdec10$", "min$", "win%", "rank",
        "stockout", "unused", "waste$/d", "dec_s"))
    print(hdr)
    print("-" * len(hdr))
    for eng in cfg["engines"]:
        a = summary["engines"].get(eng, {})
        if not a.get("n"):
            print("%-10s %4d   (no successful runs)" % (eng, 0))
            continue
        print("%-10s %4d %10.2f %10.2f %10.2f %10.2f %6.0f %5.2f %7s %7s %8s %8s" % (
            eng, a["n"], a["final_cash_mean"], a["final_cash_median"],
            a["final_cash_worst_decile_mean"], a["final_cash_min"],
            100.0 * (a["win_rate"] or 0.0), a["mean_rank"] or 0.0,
            fmt(a["stockout_rate"], "%.3f"),
            fmt(a["unused_units_per_trading_day"], "%.1f"),
            fmt(a["wasted_cost_per_trading_day"], "%.1f"),
            fmt(a["decide_s_mean"], "%.3f")))
    for eng in cfg["engines"]:
        a = summary["engines"].get(eng, {})
        if a.get("n"):
            print("  %-8s p10=%s  max_decide_s=%s  sell_through=%s  trading_days=%d  sat_out=%d"
                  % (eng, fmt(a["final_cash_p10"]), fmt(a["decide_s_max"], "%.3f"),
                     fmt(a["sell_through"], "%.3f"), a["trading_days"], a["sat_out_days"]))

    # ---- per-seed cash, side by side
    engines_ok = [e for e in cfg["engines"] if summary["engines"].get(e, {}).get("n")]
    if engines_ok:
        print()
        print("per-seed final cash")
        row = "%-14s" % "seed" + "".join("%12s" % e for e in engines_ok)
        print(row)
        print("-" * len(row))
        seeds = sorted({s for e in engines_ok
                        for s in summary["engines"][e]["final_cash_by_seed"]})
        for s in seeds:
            line = "%-14s" % s
            for e in engines_ok:
                v = summary["engines"][e]["final_cash_by_seed"].get(s)
                line += "%12s" % ("-" if v is None else "%.2f" % v)
            print(line)

    # ---- paired
    for p in summary["paired"]:
        print()
        print("paired: %s vs %s   (n=%d seeds run on both)" % (p["a"], p["b"], p["n_paired"]))
        print("%-14s %12s %12s %12s" % ("seed", p["a"], p["b"], "diff"))
        print("-" * 52)
        for r in p["per_seed"]:
            print("%-14s %12.2f %12.2f %+12.2f" % (r["seed"], r["a_cash"], r["b_cash"], r["diff"]))
        print("-" * 52)
        print("mean diff %+.2f   median diff %+.2f   sd %.2f" % (
            p["mean_diff"], p["median_diff"], p["sd_diff"]))
        if p["ci95_diff"]:
            print("95%% CI (paired t, df=%s): [%+.2f, %+.2f]" % (
                p["paired_t_df"], p["ci95_diff"][0], p["ci95_diff"][1]))
        print("seeds won: %s=%d  %s=%d  ties=%d" % (
            p["a"], p["wins_a"], p["b"], p["wins_b"], p["ties"]))
        if p["paired_t_p"] is not None:
            print("paired t = %.3f, df=%d, two-sided p = %.4f ; sign test p = %s"
                  % (p["paired_t"], p["paired_t_df"], p["paired_t_p"],
                     fmt(p["sign_test_p"], "%.4f")))
        print(p["note"])

    # ---- per-spot
    for eng in cfg["engines"]:
        a = summary["engines"].get(eng, {})
        if not a.get("n") or not a.get("per_spot"):
            continue
        print()
        print("per-spot breakdown [%s] (pooled over %d runs)" % (eng, a["n"]))
        print("%-10s %7s %12s %12s %9s %9s" % (
            "spot", "visits", "profit$", "$/visit", "sell-thru", "stockout"))
        print("-" * 64)
        for spot, v in a["per_spot"].items():
            st = (v["sold"] / v["meals"]) if v["meals"] else None
            print("%-10s %7d %12.2f %12.2f %9s %9s" % (
                spot, v["visits"], v["profit"], v["profit_per_visit"] or 0.0,
                fmt(st, "%.3f"), fmt(v["stockout_rate"], "%.3f")))

    # ---- profit curve
    for eng in cfg["engines"]:
        a = summary["engines"].get(eng, {})
        if not a.get("n") or not a.get("profit_curve"):
            continue
        print()
        print("mean profit per day [%s]" % eng)
        cells = ["d%02d %8.1f" % (c["day"], c["mean_profit"] or 0.0) for c in a["profit_curve"]]
        for i in range(0, len(cells), 5):
            print("  " + "  ".join(cells[i:i + 5]))

    if summary["failures"]:
        print()
        print("failures / skipped (%d):" % len(summary["failures"]))
        for f in summary["failures"]:
            print("  %-10s %-14s %s" % (f["engine"], f["seed"], f["error"]))
    print()
    print("JSON summary: %s" % summary["output_path"])
    print("Per-run records: %s" % summary["runs_path"])


# ------------------------------------------------------------------ main

def build_summary(label, cfg, runs_by_engine, failures, out_path, runs_path):
    return {
        "label": label,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "config": cfg,
        "engines": {e: aggregate(runs_by_engine.get(e, [])) for e in cfg["engines"]},
        "paired": paired(runs_by_engine, cfg["engines"]),
        "runs": {e: runs_by_engine.get(e, []) for e in cfg["engines"]},
        "failures": failures,
        "output_path": out_path,
        "runs_path": runs_path,
    }


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="baseline,bot",
                    help="comma-separated module names (default: baseline,bot)")
    ap.add_argument("--n", type=int, default=8, help="number of seeds (default 8)")
    ap.add_argument("--seeds", default="tune", choices=["tune", "hold"],
                    help="seed set: tune-0.. or hold-0.. (disjoint; default tune)")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--ablate", default="",
                    help="KEY=VALUE[,KEY=VALUE...] applied to bot.P before running")
    ap.add_argument("--label", default="", help="output label (default auto)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "bench_results"))
    ap.add_argument("--market-timeout", type=int, default=900,
                    help="seconds before a stuck market aborts the bench (0 = no limit)")
    ap.add_argument("--start-retries", type=int, default=3,
                    help="attempts to start each practice market")
    ap.add_argument("--max-429", type=int, default=6,
                    help="consecutive rate-limit failures before giving up entirely")
    ap.add_argument("--pause", type=float, default=1.0,
                    help="seconds to idle between markets (be kind to the API)")
    args = ap.parse_args(argv)

    if not os.environ.get("NIGHTSHIFT_KEY"):
        sys.exit("Set NIGHTSHIFT_KEY first (export NIGHTSHIFT_KEY=fk_...).")

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if not engines:
        sys.exit("--engines is empty")
    if args.n < 1:
        sys.exit("--n must be >= 1")

    try:
        ablate = parse_ablation(args.ablate)
    except ValueError as e:
        sys.exit(str(e))

    seeds = ["%s-%d" % (args.seeds, i) for i in range(args.n)]
    for s in seeds:
        if s.startswith("bench-") or s.startswith("smoke-"):
            sys.exit("refusing to use burned seed %r" % s)

    label = args.label or "%s_%s_n%d%s" % (
        "-".join(engines), args.seeds, args.n,
        ("_" + "_".join("%s=%s" % (k, v) for k, v in sorted(ablate.items())))
        if ablate else "")
    label = safe_label(label)

    out_dir = args.out_dir
    log_dir = os.path.join(out_dir, "logs", label)
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(out_dir, label + ".json")
    runs_path = os.path.join(out_dir, label + ".runs.jsonl")
    open(runs_path, "w", encoding="utf-8").close()  # fresh per invocation

    mods = {}
    for e in engines:
        try:
            mods[e] = load_engine(e)
        except Exception as ex:  # noqa: BLE001
            sys.exit("could not load engine %r: %s" % (e, ex))

    # Apply the ablation to bot.P only (that is the engine under development).
    originals = {}
    if ablate:
        target = mods.get("bot")
        if target is None:
            sys.exit("--ablate given but 'bot' is not in --engines")
        unknown = [k for k in ablate if k not in target.P]
        if unknown:
            print("warning: --ablate introduces key(s) not already in bot.P: %s"
                  % ", ".join(unknown))
        originals = copy.deepcopy(target.P)
        target.P.update(ablate)
        print("applied ablation to bot.P: %s" % json.dumps(
            {k: repr(v) for k, v in ablate.items()}))

    cfg = {"engines": engines, "n": args.n, "seed_set": args.seeds, "seeds": seeds,
           "days": args.days, "rivals": args.rivals, "ablate": {k: repr(v) for k, v in ablate.items()},
           "ablate_applies_to": "bot" if ablate else None,
           "market_timeout_s": args.market_timeout,
           "worst_decile_definition": "mean of the worst max(1, round(0.10*n)) runs; p10 is the interpolated 10th percentile"}

    runs_by_engine = {e: [] for e in engines}
    failures = []
    consecutive_429 = 0
    aborted = None

    def flush_summary():
        summary = build_summary(label, cfg, runs_by_engine, failures, out_path, runs_path)
        if aborted:
            summary["aborted"] = aborted
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        os.replace(tmp, out_path)
        return summary

    print("bench %s: %d seeds x %d engines, %d-day practice markets, one at a time"
          % (label, len(seeds), len(engines), args.days))
    print("writing to %s" % out_path)

    t_start = time.time()
    try:
        for seed in seeds:
            if aborted:
                break
            for eng in engines:
                if aborted:
                    break
                mod = mods[eng]
                log_path = os.path.join(log_dir, "%s__%s.jsonl" % (eng, seed))
                final = None
                last_err = "unknown"
                for attempt in range(1, args.start_retries + 1):
                    t0 = time.time()
                    print("  [%s] %s attempt %d ..." % (eng, seed, attempt))
                    final, history, out, err = run_one(
                        mod, seed, args.days, args.rivals, log_path, args.market_timeout)
                    if final is not None:
                        consecutive_429 = 0
                        break
                    rate_limited = "(429)" in out or "429" in (err or "")
                    last_err = err or (out.strip().splitlines()[-1] if out.strip()
                                       else "run_practice returned None")
                    if err and err.startswith("timeout"):
                        aborted = ("market timed out after %ds (%s/%s); stopping so no "
                                   "orphaned request loop keeps hitting the API"
                                   % (args.market_timeout, eng, seed))
                        break
                    if rate_limited:
                        consecutive_429 += 1
                        if consecutive_429 >= args.max_429:
                            aborted = ("%d consecutive rate-limit failures; stopping so the "
                                       "live bot's requests take priority" % consecutive_429)
                            break
                        wait = min(60.0, 10.0 * attempt)
                        print("    rate limited; sleeping %.0fs" % wait)
                        time.sleep(wait)
                    else:
                        if attempt < args.start_retries:
                            print("    failed (%s); retrying in 5s" % last_err)
                            time.sleep(5)
                if final is None:
                    failures.append({"engine": eng, "seed": seed, "error": last_err})
                    print("    SKIP %s/%s: %s" % (eng, seed, last_err))
                    with open(runs_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"engine": eng, "seed": seed, "ok": False,
                                            "error": last_err}) + "\n")
                    flush_summary()
                    continue

                try:
                    rec = summarize_run(eng, seed, final, history,
                                        read_decide_times(log_path))
                except Exception as ex:  # noqa: BLE001 - one bad state must not end the bench
                    msg = "finished but could not summarize: %s: %s" % (type(ex).__name__, ex)
                    failures.append({"engine": eng, "seed": seed, "error": msg})
                    print("    SKIP %s/%s: %s" % (eng, seed, msg))
                    flush_summary()
                    continue
                rec["wall_s"] = round(time.time() - t0, 2)
                rec["log_path"] = log_path
                runs_by_engine[eng].append(rec)
                with open(runs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                flush_summary()
                print("    cash $%.2f  rank %s/%s  stockout %s  unused %s/day  %.1fs"
                      % (rec["final_cash"], rec["rank"], rec["field_size"],
                         fmt(rec["stockout_rate"], "%.2f"),
                         fmt(rec["unused_units_per_trading_day"], "%.1f"), rec["wall_s"]))
                if args.pause > 0:
                    time.sleep(args.pause)
    except KeyboardInterrupt:
        aborted = "interrupted by user"
        print("\ninterrupted - writing what we have")
    finally:
        if ablate and originals:
            mods["bot"].P.clear()
            mods["bot"].P.update(originals)

    summary = flush_summary()
    summary["wall_s"] = round(time.time() - t_start, 1)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    try:
        print_report(summary)
    except Exception as ex:  # noqa: BLE001 - the JSON is already on disk
        import traceback
        traceback.print_exc()
        print("report rendering failed (%s); results are intact in %s" % (ex, out_path))
    if aborted:
        print("ABORTED: %s" % aborted)
        return 2
    return 0 if any(runs_by_engine[e] for e in engines) else 1


if __name__ == "__main__":
    sys.exit(main())

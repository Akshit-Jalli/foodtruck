"""Steps 3-4: decision-focused calibration of the customer priors.

Not a re-fit of demand. The objective here is the thing we are actually paid
in -- mean final cash over a fixed set of practice markets, measured by
playing them, with the existing optimiser untouched inside the loop. That
makes this a black-box decision loss in the spirit of the task-based
end-to-end learning of Donti, Amos and Kolter (NeurIPS 2017), and
deliberately not SPO+.

Why not SPO+. Elmachtoub and Grigas derive the SPO+ surrogate for decisions
of the form argmin over a fixed convex feasible region of a cost that is
*linear* in the predicted parameter. Our decision is an argmax over a
discrete (spot x quality x price) grid, and realised profit is not linear in
the predicted demand: it passes through a newsvendor with a CVaR penalty and
a capacity-constrained allocation whose profit surface is piecewise constant
with cliffs wherever our offer crosses a rival's value. None of the structure
SPO+ needs is present, so its convexity and consistency guarantees would not
transfer. We optimise the real loss directly instead.

Design, in brief:

* Low-dimensional. Three scalars, not 24. The priors move as a family:
      alpha  blend from the established priors (0) to the RMSE fit (1)
      delta  additive shift applied to every spot's limit_mean
      kappa  multiplicative scale applied to every spot's limit_sd
  Twenty-four free parameters searched against eight noisy market outcomes
  would fit the seeds, not the market.

* Common random numbers. Every evaluation plays the same seeds, so the
  difference between two thetas is a paired difference, not two draws.

* Regularised toward the established priors. theta = (0, 0, 1) is the proven
  configuration, and the penalty is what the search has to beat to leave it.

* Train seeds only. cal-* here; tune-* is kept for choosing between
  finalists, and hold-* is not touched until the merge gate.

    python calibrate.py --evals 24 --parallel 4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import bot
import ftlab

HERE = os.path.dirname(os.path.abspath(__file__))

BASE = {s: dict(c) for s, c in bot.PRIOR_SETS["orig"].items()}
FIT = {s: dict(c) for s, c in bot.PRIOR_SETS["rmse"].items()}

# (lo, hi, starting compass step)
SPACE = {"alpha": (0.0, 1.0, 0.50), "delta": (-1.5, 1.5, 0.75), "kappa": (0.6, 1.6, 0.25)}
THETA0 = {"alpha": 0.0, "delta": 0.0, "kappa": 1.0}
LIMIT_SD_FLOOR = 0.5


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def priors_of(theta):
    """theta -> a full SPOTS dict."""
    a = clamp(theta["alpha"], *SPACE["alpha"][:2])
    d = clamp(theta["delta"], *SPACE["delta"][:2])
    k = clamp(theta["kappa"], *SPACE["kappa"][:2])
    out = {}
    for s in BASE:
        b, f = BASE[s], FIT[s]
        lm = (1 - a) * b["limit_mean"] + a * f["limit_mean"] + d
        sd = ((1 - a) * b["limit_sd"] + a * f["limit_sd"]) * k
        qm = tuple((1 - a) * x + a * y for x, y in zip(b["qmin"], f["qmin"]))
        tot = sum(qm) or 1.0
        out[s] = {"limit_mean": round(lm, 4),
                  "limit_sd": round(max(LIMIT_SD_FLOOR, sd), 4),
                  "qmin": tuple(round(x / tot, 4) for x in qm)}
    return out


def penalty(theta, weight):
    """Cost, in dollars, of standing away from the established priors."""
    a = (theta["alpha"] - THETA0["alpha"]) / 1.0
    d = (theta["delta"] - THETA0["delta"]) / 1.5
    k = (theta["kappa"] - THETA0["kappa"]) / 0.5
    return weight * (a * a + d * d + k * k)


def run_market(seed, priors, days, rivals, extra):
    cmd = [sys.executable, os.path.join(HERE, "_calworker.py"),
           "--seed", seed, "--priors", json.dumps(priors),
           "--days", str(days), "--rivals", str(rivals)]
    for k, v in (extra or {}).items():
        cmd += ["--set", "%s=%s" % (k, json.dumps(v))]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"seed": seed, "ok": False, "cash": None, "err": "timeout"}
    lines = [l for l in (out.stdout or "").strip().splitlines() if l.startswith("{")]
    if not lines:
        return {"seed": seed, "ok": False, "cash": None, "err": (out.stderr or "")[-300:]}
    return json.loads(lines[-1])


def evaluate(priors, seeds, days=30, rivals=3, parallel=4, extra=None):
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        res = list(ex.map(lambda s: run_market(s, priors, days, rivals, extra), seeds))
    ok = [r for r in res if r.get("ok") and r.get("cash") is not None]
    mean = sum(r["cash"] for r in ok) / len(ok) if ok else float("-inf")
    return mean, {r["seed"]: r.get("cash") for r in res}, res


def compass(seeds, evals, parallel, reg_weight, days, rivals, extra, log):
    """First-improvement compass search, the shape fit.py already uses."""
    theta = dict(THETA0)
    steps = {k: SPACE[k][2] for k in SPACE}
    cache = {}
    budget = {"n": 0}

    def score(th):
        key = tuple(round(th[k], 4) for k in sorted(th))
        if key in cache:
            return cache[key]
        if budget["n"] >= evals:
            return None
        budget["n"] += 1
        t0 = time.time()
        mean, per, _ = evaluate(priors_of(th), seeds, days, rivals, parallel, extra)
        pen = penalty(th, reg_weight)
        val = mean - pen
        cache[key] = val
        rec = {"eval": budget["n"], "theta": dict(th), "mean_cash": mean,
               "penalty": pen, "objective": val, "per_seed": per,
               "took_s": round(time.time() - t0, 1)}
        log.append(rec)
        print("  eval %2d  a=%.2f d=%+.2f k=%.2f   cash $%9.2f  pen %6.2f  obj $%9.2f  (%.0fs)"
              % (budget["n"], th["alpha"], th["delta"], th["kappa"],
                 mean, pen, val, rec["took_s"]))
        return val

    best = score(theta)
    print("  start: objective $%.2f" % best)
    while budget["n"] < evals and max(steps.values()) > 1e-3:
        improved = False
        for k in ("alpha", "delta", "kappa"):
            for sgn in (+1, -1):
                if budget["n"] >= evals:
                    break
                cand = dict(theta)
                cand[k] = clamp(theta[k] + sgn * steps[k], *SPACE[k][:2])
                if abs(cand[k] - theta[k]) < 1e-9:
                    continue
                val = score(cand)
                if val is None:
                    break
                if val > best + 1e-9:
                    theta, best, improved = cand, val, True
                    print("    -> moved %s to %.3f" % (k, theta[k]))
                    break
            if improved:
                break
        if not improved:
            for k in steps:
                steps[k] *= 0.5
            print("    (no improvement; steps halved to a=%.3f d=%.3f k=%.3f)"
                  % (steps["alpha"], steps["delta"], steps["kappa"]))
    return theta, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=24)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--reg", type=float, default=120.0,
                    help="dollars charged for a full-scale move off the established priors")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--seeds", default="cal", choices=["cal", "tune", "hold"])
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="bot.P override held fixed across the whole search")
    ap.add_argument("--out", default="calibration_dfl.json")
    args = ap.parse_args()

    extra = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        extra[k] = json.loads(v)

    seeds = {"cal": ftlab.TRAIN, "tune": ftlab.VAL, "hold": ftlab.HOLD}[args.seeds]
    print("decision-focused calibration")
    print("  objective : mean final cash over %s-* (%d markets per evaluation)"
          % (args.seeds, len(seeds)))
    print("  held fixed: %s" % (extra or "nothing"))
    print("  penalty   : $%.0f for a full-scale move off the established priors" % args.reg)
    print("  budget    : %d evaluations, %d markets in parallel" % (args.evals, args.parallel))
    print("")

    log = []
    t0 = time.time()
    theta, best = compass(seeds, args.evals, args.parallel, args.reg,
                          args.days, args.rivals, extra, log)
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "seeds": seeds, "seed_set": args.seeds, "extra": extra,
           "reg_weight": args.reg, "evals": len(log), "wall_s": round(time.time() - t0, 1),
           "theta": theta, "objective": best, "priors": priors_of(theta), "log": log}
    json.dump(out, open(args.out, "w"), indent=1)
    print("")
    print("best theta: alpha=%.3f delta=%+.3f kappa=%.3f   objective $%.2f"
          % (theta["alpha"], theta["delta"], theta["kappa"], best))
    print("priors:")
    for s, c in out["priors"].items():
        print("    %-10s limit_mean %6.2f  limit_sd %5.2f  qmin %s"
              % (s, c["limit_mean"], c["limit_sd"], c["qmin"]))
    print("")
    print("wrote %s  (%.1f min)" % (args.out, out["wall_s"] / 60.0))


if __name__ == "__main__":
    main()

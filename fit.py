"""Fit the per-spot customer priors in bot.py to observed spot-days.

    export NIGHTSHIFT_KEY=...        # not needed; this script is offline
    python fit.py                    # fit probe_days.jsonl (+ days.jsonl)
    python fit.py --holdout 2        # how many harvested markets to hold back

Each spot-day is an experiment: we know the realised crowd, every truck's
price, quality, meal count and reputation, how many meals each one sold, and
how many customers left hungry. bot.py's market model turns candidate priors
into predicted sales, so we search the priors for the values that reproduce
what actually happened -- then pull them back toward the documented priors by
an amount that depends on how much evidence the spot actually has.

Four things this gets right that a naive least-squares fit does not:

1. Censoring. A truck with sold == meals sold out; true demand was *at least*
   that. Predicting more demand than it sold must not be penalised, so its
   residual is one-sided: only underprediction counts.
2. The hungry count. `customers - hungry` is the total served at that spot.
   It is never censored (the meal caps are inside the model), it constrains
   the whole population rather than one truck, and it is the single strongest
   signal available. It gets its own residual.
3. Shrinkage. The fit is penalised toward bot.py's documented priors with the
   weight of n0 pseudo-observations, so a spot with 50 rows moves a little and
   a spot with 240 moves a lot. The report prints the implied shrinkage.
4. Honest validation. Some harvested markets are held out, the fit is done on
   the rest, and out-of-sample error is compared against the hand-written
   priors. If the fit is not better out of sample, the report says so.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import bot
from harvest import RepBook

PROBE = os.environ.get("NIGHTSHIFT_PROBE", "probe_days.jsonl")
DAYS = os.environ.get("NIGHTSHIFT_DAYS", "days.jsonl")
OUT = "calibration.json"

NCELLS = 20          # price-limit cells per quality rank inside the model
# Strength of the prior, in pseudo-observations. Chosen by the cross-validated
# sweep in --sweep-prior: on 854 harvested spot-days the out-of-sample error is
# flat from 0 to ~20 and degrades steadily above it, so the data is already
# strong enough that heavy shrinkage only costs accuracy. 20 sits at the top of
# that flat region: a spot with 3 observations still moves only 13% of the way
# to its fitted value, while a spot with 240 moves 92%.
PSEUDO_N = 20.0
# How far one pseudo-observation of disagreement lets each parameter move.
SCALES = {"limit_mean": 1.5, "limit_sd": 0.75, "qcheap": 0.12, "qprem": 0.12}
BOUNDS = {"limit_mean": (6.0, 20.0), "limit_sd": (0.5, 7.0)}

# bot.py's priors as they stood when this process imported it, deep-copied so
# that one run stays self-consistent even if bot.py is edited underneath us.
# This is the shrinkage target and the "before" in every comparison.
BASE = {s: {"limit_mean": float(c["limit_mean"]), "limit_sd": float(c["limit_sd"]),
            "qmin": tuple(float(x) for x in c["qmin"])}
        for s, c in bot.SPOTS.items()}

# The priors as originally documented. Pinned, so the headline comparison has
# a fixed anchor even once bot.py has been recalibrated by someone else.
DOC = {
    "highst":   {"limit_mean": 12.0, "limit_sd": 3.0, "qmin": (0.50, 0.35, 0.15)},
    "oval":     {"limit_mean": 13.0, "limit_sd": 2.3, "qmin": (0.30, 0.50, 0.20)},
    "union":    {"limit_mean": 13.0, "limit_sd": 2.2, "qmin": (0.22, 0.53, 0.25)},
    "northres": {"limit_mean": 11.0, "limit_sd": 2.1, "qmin": (0.45, 0.45, 0.10)},
    "thompson": {"limit_mean": 12.0, "limit_sd": 2.3, "qmin": (0.45, 0.40, 0.15)},
    "stadium":  {"limit_mean": 13.0, "limit_sd": 2.6, "qmin": (0.25, 0.50, 0.25)},
}


# ------------------------------------------------------------------ loading

def load_probe(path=PROBE):
    """Harvested practice markets. Reputation is already reconstructed."""
    obs = []
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return obs
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("spot") not in bot.SPOTS or not r.get("trucks"):
                continue
            obs.append({
                "source": "probe",
                "market": r.get("seed") or r.get("practice_id"),
                "day": r.get("day"),
                "spot": r["spot"],
                "customers": float(r.get("customers") or 0),
                "hungry": float(r.get("hungry") or 0),
                "expected": r.get("expected_customers"),
                "trucks": [{"quality": t.get("quality", "standard"),
                            "price": float(t["price"]),
                            "meals": float(t["meals"]),
                            "sold": float(t["sold"]),
                            "rep": float(t.get("rep", 50.0))} for t in r["trucks"]],
            })
    return obs


def load_live(path=DAYS):
    """The live game's archive. Sparse, and reputation must be rebuilt here."""
    recs = []
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return []
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    recs.sort(key=lambda r: (r.get("mode") or "", r.get("market") or 0, r.get("day") or 0))
    obs = []
    books = {}
    for r in recs:
        spots = list((r.get("spots") or {}).keys()) or list(bot.SPOTS)
        key = (r.get("mode"), r.get("market"))
        book = books.setdefault(key, RepBook(spots))
        fc = (r.get("forecast_before") or {}).get("expected_customers") or {}
        for spot, s in (r.get("spots") or {}).items():
            if spot not in bot.SPOTS or not (s.get("trucks") or []):
                continue
            obs.append({
                "source": "live",
                "market": "live-%s" % (r.get("market"),),
                "day": r.get("day"),
                "spot": spot,
                "customers": float(s.get("customers") or 0),
                "hungry": float(s.get("hungry") or 0),
                "expected": fc.get(spot),
                "trucks": [{"quality": t.get("quality", "standard"),
                            "price": float(t.get("price") or 0),
                            "meals": float(t.get("meals") or 0),
                            "sold": float(t.get("sold") or 0),
                            "rep": float(book.of(t["id"], spot))} for t in (s["trucks"])],
            })
        book.apply_day(r.get("spots"))
    return obs


# ------------------------------------------------------------------ the model

def predict(ob, cfg, ncells=NCELLS):
    """Predicted sales per truck at one spot-day, under candidate priors."""
    spot = ob["spot"]
    saved = bot.SPOTS[spot]
    bot.SPOTS[spot] = cfg
    try:
        cells = bot.cells_for(spot, ncells)
        trucks = [
            bot.Truck(bot.QVALUE[t["quality"]] * bot.repfactor(t["rep"]) - t["price"],
                      t["price"], bot.QRANK[t["quality"]], t["meals"])
            for t in ob["trucks"]
        ]
        return bot.allocate(cells, trucks, ob["customers"])
    finally:
        bot.SPOTS[spot] = saved


def obs_loss(ob, cfg, ncells=NCELLS):
    """Mean squared residual for one spot-day, in meals^2.

    Per-truck residuals are one-sided when that truck sold out: predicted
    sales are capped at its meal count anyway, so for a sold-out truck the
    only way to be wrong is to predict that it would *not* have sold out.
    The served-total residual is two-sided -- the caps are already in the
    model, so `customers - hungry` is an exact, uncensored target.
    """
    pred = predict(ob, cfg, ncells)
    tot = 0.0
    for p, t in zip(pred, ob["trucks"]):
        r = p - t["sold"]
        if t["sold"] >= t["meals"] - 1e-9:      # censored: sold out
            r = min(r, 0.0)
        tot += r * r
    served = ob["customers"] - ob["hungry"]
    d = sum(pred) - served
    tot += d * d
    return tot / (len(ob["trucks"]) + 1)


def data_loss(obs, cfg, ncells=NCELLS):
    return sum(obs_loss(o, cfg, ncells) for o in obs)


def rmse(obs, cfg, ncells=NCELLS):
    if not obs:
        return float("nan")
    return math.sqrt(data_loss(obs, cfg, ncells) / len(obs))


# ------------------------------------------------------------------ the fit

def theta_of(cfg):
    return [cfg["limit_mean"], cfg["limit_sd"], cfg["qmin"][0], cfg["qmin"][2]]


def cfg_of(th):
    m = min(BOUNDS["limit_mean"][1], max(BOUNDS["limit_mean"][0], th[0]))
    sd = min(BOUNDS["limit_sd"][1], max(BOUNDS["limit_sd"][0], th[1]))
    qc = min(0.97, max(0.01, th[2]))
    qp = min(0.97, max(0.01, th[3]))
    if qc + qp > 0.98:
        k = 0.98 / (qc + qp)
        qc, qp = qc * k, qp * k
    return {"limit_mean": m, "limit_sd": sd, "qmin": (qc, 1.0 - qc - qp, qp)}


def fit_spot(spot, obs, prior, sigma2, pseudo_n=PSEUDO_N, ncells=NCELLS, max_evals=400):
    """Penalised pattern search. Returns (cfg, n, shrinkage, evals)."""
    mine = [o for o in obs if o["spot"] == spot]
    n = len(mine)
    shrink = pseudo_n / (pseudo_n + n)
    if n == 0:
        return dict(prior), 0, 1.0, 0
    th0 = theta_of(prior)
    sc = [SCALES["limit_mean"], SCALES["limit_sd"], SCALES["qcheap"], SCALES["qprem"]]
    evals = [0]

    def J(th):
        cfg = cfg_of(th)
        evals[0] += 1
        t = theta_of(cfg)
        pen = pseudo_n * sigma2 * sum(((t[j] - th0[j]) / sc[j]) ** 2 for j in range(4))
        return data_loss(mine, cfg, ncells) + pen

    th = list(th0)
    best = J(th)
    steps = [1.0, 0.5, 0.06, 0.06]
    while steps[0] >= 0.05 and evals[0] < max_evals:
        improved = False
        for j in range(4):
            for s in (steps[j], -steps[j]):
                if evals[0] >= max_evals:
                    break
                cand = list(th)
                cand[j] += s
                v = J(cand)
                if v < best - 1e-9:
                    best, th, improved = v, cand, True
                    break
        if not improved:
            steps = [s * 0.5 for s in steps]
    return cfg_of(th), n, shrink, evals[0]


def fit_all(obs, sigma2, pseudo_n=PSEUDO_N, ncells=NCELLS):
    out = {}
    for spot in BASE:
        cfg, n, shrink, ev = fit_spot(spot, obs, BASE[spot], sigma2, pseudo_n, ncells)
        out[spot] = {"cfg": cfg, "n": n, "shrinkage": shrink, "evals": ev}
    return out


def cross_validate(obs, markets, k, pseudo_n, ncells):
    """Leave-markets-out CV. A single split is a coin toss on six spots; this
    rotates every harvested market through the test set exactly once.

    Live-archive rows are tiny in number and always stay in training.
    Returns pooled squared error per spot for each candidate prior set.
    """
    folds = [markets[i::k] for i in range(k)]
    folds = [f for f in folds if f]
    acc = {s: {"n": 0, "base": 0.0, "fit": 0.0, "doc": 0.0} for s in BASE}
    for fold in folds:
        held = set(fold)
        train = [o for o in obs if o["market"] not in held]
        test = [o for o in obs if o["market"] in held]
        if not test:
            continue
        f = fit_all(train, scale_sigma2(train, ncells), pseudo_n, ncells)
        for o in test:
            s = o["spot"]
            a = acc[s]
            a["n"] += 1
            a["base"] += obs_loss(o, BASE[s], ncells)
            a["fit"] += obs_loss(o, f[s]["cfg"], ncells)
            a["doc"] += obs_loss(o, DOC[s], ncells)
    return acc, len(folds)


def scale_sigma2(obs, ncells=NCELLS):
    """Typical per-observation loss under the documented priors, in meals^2.

    Used to put the shrinkage penalty in the same units as the data term, so
    that `pseudo_n` really does behave like a count of prior observations.
    The data term is a *sum* over observations, so its natural per-observation
    unit is the mean, not the median -- the loss is heavily right-skewed
    (median ~7, mean ~26 meals^2 here) and a median-based scale would leave
    the prior far too weak to hold a big-n spot back.
    """
    vals = [obs_loss(o, BASE[o["spot"]], ncells) for o in obs]
    return max(1.0, statistics.mean(vals)) if vals else 1.0


# ------------------------------------------------------------------ crowd noise

def crowd_mults(obs, probs=(0.15, 0.20, 0.30, 0.20, 0.15)):
    """Multiplicative crowd noise: realised customers / forecast customers."""
    rs = sorted(o["customers"] / float(o["expected"])
                for o in obs if o.get("expected"))
    if len(rs) < 20:
        return None, {"n": len(rs)}

    def q(p):
        i = p * (len(rs) - 1)
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return rs[lo] + (rs[hi] - rs[lo]) * (i - lo)

    # Put each atom at the centre of its probability band, so the shape of
    # the empirical distribution survives, then stretch the atoms about their
    # mean until the discrete variance matches the empirical one. Five atoms
    # at band centres always under-disperse, and under-dispersion is the
    # dangerous direction: bot.py's newsvendor sizes inventory off these
    # scenarios and prices the bad tail with a CVaR, so a crowd distribution
    # with the tails shaved off quietly makes every day look safer than it is.
    cum, raw = 0.0, []
    for p in probs:
        raw.append((q(cum + p / 2.0), p))
        cum += p
    emp_sd = statistics.pstdev(rs)
    mu = sum(m * p for m, p in raw)
    var = sum(p * (m - mu) ** 2 for m, p in raw)
    stretch = (emp_sd / math.sqrt(var)) if var > 1e-12 else 1.0
    atoms = [(round(max(0.05, mu + (m - mu) * stretch), 3), p) for m, p in raw]
    mean = sum(m * p for m, p in atoms)
    stats = {
        "n": len(rs),
        "empirical_mean": round(statistics.mean(rs), 4),
        "empirical_sd": round(statistics.pstdev(rs), 4),
        "empirical_min": round(rs[0], 3),
        "empirical_max": round(rs[-1], 3),
        "p05": round(q(0.05), 3), "p50": round(q(0.50), 3), "p95": round(q(0.95), 3),
        "discretised_mean": round(mean, 4),
        "variance_stretch": round(stretch, 4),
    }
    cur = bot.P["crowd_mults"]
    stats["current_mean"] = round(sum(m * p for m, p in cur), 4)
    stats["current_sd"] = round(math.sqrt(
        sum(p * m * m for m, p in cur) - stats["current_mean"] ** 2), 4)
    stats["discretised_sd"] = round(math.sqrt(
        sum(p * m * m for m, p in atoms) - mean ** 2), 4)
    return atoms, stats


# ------------------------------------------------------------------ reporting

def spots_block(cfgs):
    lines = ["SPOTS = {"]
    for spot in BASE:
        c = cfgs[spot]
        lines.append('    "%s":%s {"limit_mean": %5.2f, "limit_sd": %4.2f, '
                     '"qmin": (%.2f, %.2f, %.2f)},'
                     % (spot, " " * (9 - len(spot)), c["limit_mean"], c["limit_sd"],
                        c["qmin"][0], c["qmin"][1], c["qmin"][2]))
    lines.append("}")
    return "\n".join(lines)




def rms(total_sq, n):
    return math.sqrt(total_sq / n) if n else float("nan")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="calibrate bot.py's customer priors")
    ap.add_argument("--folds", type=int, default=4,
                    help="leave-markets-out CV folds (0 to skip validation)")
    ap.add_argument("--pseudo-n", type=float, default=PSEUDO_N)
    ap.add_argument("--cells", type=int, default=NCELLS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sweep-prior", action="store_true",
                    help="cross-validate the shrinkage strength and exit")
    args = ap.parse_args()

    t0 = time.time()
    probe = load_probe()
    live = load_live()
    obs = probe + live
    if not obs:
        sys.exit("no observations in %s or %s yet" % (PROBE, DAYS))
    markets = sorted({o["market"] for o in probe})
    print("%d spot-day observations with >=1 truck  (%d harvested over %d markets, "
          "%d from the live archive)" % (len(obs), len(probe), len(markets), len(live)))

    drifted = [s for s in BASE
               if BASE[s] != {"limit_mean": DOC[s]["limit_mean"],
                              "limit_sd": DOC[s]["limit_sd"],
                              "qmin": tuple(DOC[s]["qmin"])}]
    if drifted:
        print("note: bot.py SPOTS already differ from the documented priors at %s; "
              "shrinking toward bot.py as it stands, and reporting both baselines."
              % ", ".join(sorted(drifted)))

    sigma2 = scale_sigma2(obs, args.cells)
    print("residual scale under bot.py current priors: %.2f meals (sigma^2 = %.1f); "
          "prior strength = %.0f pseudo-observations\n"
          % (math.sqrt(sigma2), sigma2, args.pseudo_n))

    if args.sweep_prior:
        print("\ncross-validated shrinkage strength (%d folds)" % args.folds)
        print("  %10s %9s %10s" % ("pseudo_n", "CV RMSE", "vs bot.py"))
        for pn in (0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, float("inf")):
            use = 1e12 if pn == float("inf") else pn
            acc, _ = cross_validate(obs, markets, args.folds, use, args.cells)
            n = sum(a["n"] for a in acc.values())
            rf = rms(sum(a["fit"] for a in acc.values()), n)
            rb = rms(sum(a["base"] for a in acc.values()), n)
            print("  %10s %9.3f %+9.2f%%"
                  % ("inf" if pn == float("inf") else "%.0f" % pn, rf,
                     100.0 * (rf - rb) / rb))
        return

    # ---- honest validation: every market takes a turn in the test set.
    cv = {}
    if args.folds >= 2 and len(markets) >= 2:
        k = min(args.folds, len(markets))
        print("cross-validating: %d folds over %d harvested markets" % (k, len(markets)))
        acc, k = cross_validate(obs, markets, k, args.pseudo_n, args.cells)
        print("\nout-of-sample RMSE (meals per spot-day residual, pooled over folds)")
        print("  %-9s %6s  %10s %10s %10s   %s"
              % ("spot", "n", "documented", "bot.py now", "fitted", "fitted vs bot.py"))
        tb = tf = td = 0.0
        tc = 0
        for spot in BASE:
            a = acc[spot]
            if not a["n"]:
                continue
            tc += a["n"]
            tb += a["base"]
            tf += a["fit"]
            td += a["doc"]
            rb, rf, rd = rms(a["base"], a["n"]), rms(a["fit"], a["n"]), rms(a["doc"], a["n"])
            cv[spot] = {"n": a["n"], "rmse_documented": round(rd, 3),
                        "rmse_botpy": round(rb, 3), "rmse_fitted": round(rf, 3)}
            print("  %-9s %6d  %10.2f %10.2f %10.2f   %+6.1f%%  %s"
                  % (spot, a["n"], rd, rb, rf, 100.0 * (rf - rb) / rb,
                     "better" if rf < rb else "WORSE"))
        rb, rf, rd = rms(tb, tc), rms(tf, tc), rms(td, tc)
        cv["_all"] = {"n": tc, "rmse_documented": round(rd, 3),
                      "rmse_botpy": round(rb, 3), "rmse_fitted": round(rf, 3), "folds": k}
        print("  %-9s %6d  %10.2f %10.2f %10.2f   %+6.1f%%  %s"
              % ("ALL", tc, rd, rb, rf, 100.0 * (rf - rb) / rb,
                 "better" if rf < rb else "WORSE"))
        wins = sum(1 for s in BASE
                   if s in cv and cv[s]["rmse_fitted"] < cv[s]["rmse_botpy"])
        print("  %d of %d spots improve out of sample" % (wins, len(cv) - 1))

    # ---- final fit on everything, plus an unshrunk fit as a diagnostic.
    print("\nfinal fit on all %d observations" % len(obs))
    final = fit_all(obs, sigma2, args.pseudo_n, args.cells)
    raw = fit_all(obs, sigma2, 0.0, args.cells)
    cfgs = {s: final[s]["cfg"] for s in BASE}

    def fmt(x):
        return "%5.2f/%4.2f/%2.0f-%2.0f-%2.0f" % (
            x["limit_mean"], x["limit_sd"],
            100 * x["qmin"][0], 100 * x["qmin"][1], 100 * x["qmin"][2])

    print("\n  (limit_mean / limit_sd / qmin%, cheap-standard-premium)")
    print("  %-9s %5s %6s  %-18s %-18s %-18s  %6s %6s"
          % ("spot", "n", "shrink", "bot.py now", "fitted", "unshrunk", "rmse0", "rmse1"))
    per_spot = {}
    for spot in BASE:
        f = final[spot]
        mine = [o for o in obs if o["spot"] == spot]
        p, c = BASE[spot], f["cfg"]
        r0, r1 = rmse(mine, p, args.cells), rmse(mine, c, args.cells)
        print("  %-9s %5d %5.0f%%  %-18s %-18s %-18s  %6.2f %6.2f"
              % (spot, f["n"], 100 * f["shrinkage"], fmt(p), fmt(c),
                 fmt(raw[spot]["cfg"]), r0, r1))
        per_spot[spot] = {
            "n": f["n"], "shrinkage": round(f["shrinkage"], 4), "evals": f["evals"],
            "documented": {"limit_mean": DOC[spot]["limit_mean"],
                           "limit_sd": DOC[spot]["limit_sd"],
                           "qmin": list(DOC[spot]["qmin"])},
            "botpy_now": {"limit_mean": p["limit_mean"], "limit_sd": p["limit_sd"],
                          "qmin": list(p["qmin"])},
            "fitted": {"limit_mean": round(c["limit_mean"], 3),
                       "limit_sd": round(c["limit_sd"], 3),
                       "qmin": [round(x, 4) for x in c["qmin"]]},
            "unshrunk": {"limit_mean": round(raw[spot]["cfg"]["limit_mean"], 3),
                         "limit_sd": round(raw[spot]["cfg"]["limit_sd"], 3),
                         "qmin": [round(x, 4) for x in raw[spot]["cfg"]["qmin"]]},
            "rmse_in_sample_botpy": round(r0, 3),
            "rmse_in_sample_fitted": round(r1, 3),
            "holdout": cv.get(spot),
        }

    # ---- crowd noise.
    atoms, cstats = crowd_mults(obs)
    print("\ncrowd multiplier (realised customers / forecast customers), n=%d" % cstats["n"])
    if atoms:
        print("  empirical mean %.3f sd %.3f, range %.2f-%.2f  (p05 %.2f, p50 %.2f, p95 %.2f)"
              % (cstats["empirical_mean"], cstats["empirical_sd"], cstats["empirical_min"],
                 cstats["empirical_max"], cstats["p05"], cstats["p50"], cstats["p95"]))
        print("  current bot.P: mean %.3f sd %.3f   ->   suggested: mean %.3f sd %.3f"
              % (cstats["current_mean"], cstats["current_sd"],
                 cstats["discretised_mean"], cstats["discretised_sd"]))

    print("\n# paste into bot.py")
    print(spots_block(cfgs))
    if atoms:
        print('    "crowd_mults": (%s),'
              % ", ".join("(%.2f, %.2f)" % (m, p) for m, p in atoms))

    verdict = "no validation run"
    if "_all" in cv:
        a = cv["_all"]
        d = 100.0 * (a["rmse_fitted"] - a["rmse_botpy"]) / a["rmse_botpy"]
        wins = sum(1 for s in BASE
                   if s in cv and cv[s]["rmse_fitted"] < cv[s]["rmse_botpy"])
        verdict = ("fitted priors are %.1f%% %s than bot.py current priors out of sample "
                   "(%d of %d spots improve)"
                   % (abs(d), "better" if d < 0 else "WORSE", wins, len(cv) - 1))
    print("\nverdict: %s" % verdict)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {"probe_rows": len(probe), "live_rows": len(live), "markets": markets},
        "settings": {"cells": args.cells, "pseudo_n": args.pseudo_n,
                     "sigma2": round(sigma2, 3), "folds": args.folds},
        "spots": per_spot,
        "holdout_overall": cv.get("_all"),
        "crowd_mults": [[m, p] for m, p in (atoms or [])],
        "crowd_stats": cstats,
        "spots_block": spots_block(cfgs),
        "verdict": verdict,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("wrote %s  (%.0fs)" % (args.out, time.time() - t0))


if __name__ == "__main__":
    main()

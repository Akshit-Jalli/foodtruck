"""Paired comparison across bench runs that differ only in configuration.

bench.py pairs engines against each other within one run. This pairs whole
runs against each other by seed, which is what an ablation comparison needs:
same engine, same seeds, one knob moved.

    python compare_runs.py d1_orig_old d1_rmse_old d1_rmse_fix
    python compare_runs.py --base d1_orig_old d1_rmse_old d1_rmse_fix
"""
import argparse
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "bench_results")


def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def t_two_sided_p(t, df):
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    return min(1.0, front * _betacf(a, b, x))


def load(label, engine="bot"):
    path = os.path.join(RESULTS, label + ".runs.jsonl")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("engine") == engine and r.get("ok"):
                out[r["seed"]] = r["final_cash"]
    if not out:
        raise SystemExit("no ok %r runs in %s" % (engine, path))
    return out


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = [b[s] - a[s] for s in seeds]
    n = len(d)
    if n < 2:
        return seeds, d, None
    m, sd = statistics.mean(d), statistics.stdev(d)
    se = sd / math.sqrt(n)
    if se == 0:
        # Every seed identical: the knob changed nothing at all. There is no
        # test to run, and no uncertainty to report.
        return seeds, d, {"n": n, "mean": m, "sd": 0.0, "t": float("nan"),
                          "df": n - 1, "p": float("nan"), "ci": (m, m),
                          "wins": 0, "losses": 0, "identical": True}
    t = m / se
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n - 1, 1.96)
    return seeds, d, {"n": n, "mean": m, "sd": sd, "t": t, "df": n - 1,
                      "p": t_two_sided_p(abs(t), n - 1),
                      "ci": (m - tcrit * se, m + tcrit * se),
                      "wins": sum(1 for x in d if x > 0),
                      "losses": sum(1 for x in d if x < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--base", default=None, help="label everything is compared against")
    ap.add_argument("--engine", default="bot")
    args = ap.parse_args()

    data = {lab: load(lab, args.engine) for lab in args.labels}
    base = args.base or args.labels[0]
    seeds = sorted(set.intersection(*[set(v) for v in data.values()]))

    print("paired over %d common seeds: %s" % (len(seeds), ", ".join(seeds)))
    print("")
    print("%-22s %11s %11s %11s" % ("config", "mean$", "median$", "min$"))
    print("-" * 58)
    for lab in args.labels:
        v = [data[lab][s] for s in seeds]
        print("%-22s %11.2f %11.2f %11.2f"
              % (lab, statistics.mean(v), statistics.median(v), min(v)))

    print("")
    print("paired differences against %s  (positive = better than base)" % base)
    print("-" * 78)
    print("%-22s %10s %9s %8s %7s %18s" % ("config", "mean diff", "sd", "t", "p", "95% CI"))
    for lab in args.labels:
        if lab == base:
            continue
        _, d, st = paired(data[base], data[lab])
        if st is None:
            continue
        if st.get("identical"):
            print("%-22s %+10.2f %9s %8s %7s  %s"
                  % (lab, st["mean"], "-", "-", "-", "identical on every seed"))
            continue
        print("%-22s %+10.2f %9.2f %8.3f %7.4f  [%+8.2f,%+8.2f]  %d-%d"
              % (lab, st["mean"], st["sd"], st["t"], st["p"],
                 st["ci"][0], st["ci"][1], st["wins"], st["losses"]))

    print("")
    print("per-seed final cash")
    print("%-9s" % "seed" + "".join("%13s" % l[:12] for l in args.labels))
    for s in seeds:
        print("%-9s" % s + "".join("%13.2f" % data[l][s] for l in args.labels))


if __name__ == "__main__":
    main()

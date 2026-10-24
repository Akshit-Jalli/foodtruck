"""Step 1: paired Day-1 diagnostic, orig priors vs rmse priors.

Day 1 is the clean experiment. There is no history, so Memory is empty and
the order is a pure function of the priors and the opening state -- and the
regression showed up hardest there (mean day-1 profit $325.10 -> $86.60).

For each seed this opens one practice market, takes the day-1 night state,
and runs decide() under both prior sets with the deadline lifted, recording
every candidate it scored. Nothing is ordered: the market is abandoned, so
the two runs see byte-identical inputs.

    python diag_day1.py --seeds tune --out diag_day1.json
"""
import argparse
import json
import time

import ftlab


def component_diff(a, b):
    out = []
    for k in ("spot", "quality", "price", "meals"):
        if a[k] != b[k]:
            out.append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="tune", choices=["tune", "hold", "cal"])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--out", default="diag_day1.json")
    args = ap.parse_args()

    ftlab.guard()
    seeds = {"tune": ftlab.VAL, "hold": ftlab.HOLD, "cal": ftlab.TRAIN}[args.seeds]
    rows = []
    for seed in seeds:
        t0 = time.time()
        base = ftlab.new_market(seed, args.days, args.rivals)
        st = ftlab.night_state(base, want_day=1)
        rec = {"seed": seed, "base": base, "days": args.days, "rivals": args.rivals,
               "forecast": (st.get("forecast") or {}).get("expected_customers"),
               "events": [(e.get("name"), e.get("spot"))
                          for e in (st.get("forecast") or {}).get("events") or []],
               "by_priors": {}}
        for pr in ("orig", "rmse"):
            order, tr = ftlab.decide_traced(st, pr)
            rec["by_priors"][pr] = {"order": order, "trace": tr}
        a = rec["by_priors"]["orig"]["order"]
        b = rec["by_priors"]["rmse"]["order"]
        rec["diff"] = component_diff(a, b)
        rows.append(rec)
        print("%-8s orig %-9s %-9s $%6.2f x%-4d | rmse %-9s %-9s $%6.2f x%-4d | differs: %s  (%.1fs)"
              % (seed, a["spot"], a["quality"], a["price"], a["meals"],
                 b["spot"], b["quality"], b["price"], b["meals"],
                 ",".join(rec["diff"]) or "nothing", time.time() - t0))

    json.dump(rows, open(args.out, "w"), indent=1)

    print("\n%s\nwhich component moves, over %d seeds\n%s" % ("=" * 62, len(rows), "=" * 62))
    tally = {}
    for r in rows:
        for k in r["diff"]:
            tally[k] = tally.get(k, 0) + 1
    for k in ("spot", "quality", "price", "meals"):
        print("  %-8s changed on %d/%d seeds" % (k, tally.get(k, 0), len(rows)))
    nch = sum(1 for r in rows if r["diff"])
    print("  %d/%d seeds get a different day-1 order at all" % (nch, len(rows)))

    print("\nprice and meals detail")
    print("  %-8s %10s %10s %8s | %7s %7s %7s" % (
        "seed", "orig", "rmse", "dprice", "o_meals", "r_meals", "dmeals"))
    for r in rows:
        a, b = r["by_priors"]["orig"]["order"], r["by_priors"]["rmse"]["order"]
        print("  %-8s %10s %10s %+8.2f | %7d %7d %+7d" % (
            r["seed"], "%s/%s" % (a["spot"][:6], a["quality"][:4]),
            "%s/%s" % (b["spot"][:6], b["quality"][:4]),
            b["price"] - a["price"], a["meals"], b["meals"], b["meals"] - a["meals"]))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()

"""One practice market under an arbitrary prior set. Used by calibrate.py.

Runs in its own process so that bot's module globals (SPOTS, MEM, P) cannot
leak between concurrent markets. Prints one JSON line on stdout.

    python _calworker.py --seed cal-0 --priors '{"highst": {...}, ...}'
"""
import argparse
import json
import os
import sys

import ftlab          # loads .env.local, sets bot.KEY
import bot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--priors", required=True, help="JSON dict of spot -> prior")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=3)
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    args = ap.parse_args()

    ftlab.guard()
    bot.LOG_PATH = os.devnull

    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in bot.P:
            sys.exit("unknown tunable %r" % k)
        bot.P[k] = json.loads(v)

    ftlab.register_priors("_cal", json.loads(args.priors))
    bot.P["priors"] = "_cal"

    final = bot.run_practice(seed=args.seed, days=args.days,
                             rivals=args.rivals, quiet=True)
    if not final:
        print(json.dumps({"seed": args.seed, "ok": False, "cash": None}))
        return
    you = final.get("you") or {}
    ranks = sorted((t.get("cash", 0) for t in final.get("trucks") or []), reverse=True)
    cash = you.get("cash")
    print(json.dumps({"seed": args.seed, "ok": True, "cash": cash,
                      "rank": (ranks.index(cash) + 1) if cash in ranks else None,
                      "field": len(ranks)}))


if __name__ == "__main__":
    main()

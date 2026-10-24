"""Explain a decision: sweep the options and show what the model thinks.

    python explain.py                      # fetch the live state and explain it
    python explain.py state.json           # explain a saved state
    python explain.py --spot union --quality premium

Read-only. It never sends an order.
"""
import argparse
import json
import sys

import bot


def fetch_state():
    status, state = bot.call("GET", "/api/v1/state?since=0")
    if status != 200:
        sys.exit("could not read state (%d): %s" % (status, state.get("error")))
    return state


def evaluate(state, spot, quality, prices):
    """Stage-2 quality numbers for one spot/quality across a price sweep."""
    rules = state["rules"]
    budget = float(rules.get("budget", 500))
    expected = (state.get("forecast") or {}).get("expected_customers") or {}
    myrep = float((state["you"].get("reputation") or {}).get(spot, 50.0))
    rivals = bot.predict_rivals(state)
    intruder = bot.strongest_rival(rivals)
    cells = bot.cells_for(spot, bot.P["limit_cells"])
    n_base = float(expected.get(spot, 150))
    cost = bot.QCOST[quality]
    hardcap = int(budget // cost)
    scen = bot.scenarios()
    spots = [s for s in rules.get("spots", list(bot.SPOTS)) if s in bot.SPOTS]
    magnet = bot.magnet_spot(state, spots) if bot.P["use_herding"] else None

    rows = []
    for price in prices:
        myv = bot.QVALUE[quality] * bot.repfactor(myrep) - price
        if myv < 0:
            continue
        demands = []
        for mult, w, extra, ds, hd in scen:
            rts = bot.rival_trucks_at(rivals, spot, intruder if extra else None, ds, hd, magnet)
            mine = bot.Truck(myv, price, bot.QRANK[quality], 1e9)
            demands.append((bot.allocate(cells, [mine] + rts, n_base * mult)[0], w))
        meals, score, mean = bot.best_inventory(demands, price, cost, hardcap)
        exp_d = sum(d * w for d, w in demands)
        rows.append((price, myv, exp_d, meals, mean, score))
    return rows, n_base, myrep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", nargs="?")
    ap.add_argument("--spot")
    ap.add_argument("--quality")
    args = ap.parse_args()

    state = json.load(open(args.state, encoding="utf-8")) if args.state else fetch_state()
    bot.preload_memory(mode=state.get("mode", "live"), market=state.get("market", 1))
    bot.MEM.sync(state)

    order = bot.decide(state)
    print("the bot would send: %s\n" % order)

    rivals = bot.predict_rivals(state)
    print("rivals as modelled (reputation is reconstructed, not reported):")
    for r in rivals:
        probs = r.get("probs")
        where = ("  likely: " + ", ".join(
            "%s %.0f%%" % (s, p * 100) for s, p in sorted(probs.items(), key=lambda x: -x[1])[:3])
        ) if probs else "  (first day, assumed)"
        print("  %-10s %-9s $%-6.2f rep@%s %.1f%s" % (
            r["id"][:10], r["quality"], r["price"], r["spot"],
            bot.MEM.rep_at(r["id"], r["spot"], r["rep"]), where))

    spot = args.spot or order["spot"]
    quality = args.quality or order["quality"]
    prices = [x * 0.25 for x in range(28, 61)]        # $7.00 - $15.00
    rows, n_base, myrep = evaluate(state, spot, quality, prices)
    print("\n%s / %s   forecast crowd %.0f   our reputation %.1f" % (
        spot, quality, n_base, myrep))
    print("%8s %7s %9s %7s %11s %11s" % (
        "price", "value", "demand", "meals", "E[profit]", "risk-adj"))
    best = max(rows, key=lambda r: r[5]) if rows else None
    for price, myv, exp_d, meals, mean, score in rows:
        mark = "  <-- picked" if best and price == best[0] else ""
        print("%8.2f %7.2f %9.1f %7d %11.2f %11.2f%s" % (
            price, myv, exp_d, meals, mean, score, mark))
    if best:
        top = max(rows, key=lambda r: r[4])
        print("\nhighest expected profit  $%.2f at $%.2f" % (top[4], top[0]))
        print("highest risk-adjusted    $%.2f at $%.2f (what the bot optimises)" % (
            best[4], best[0]))
        print("cost of the hedge:       $%.2f" % (top[4] - best[4]))


if __name__ == "__main__":
    main()

"""Fit the per-spot customer priors in bot.py to the real game.

    python fit.py              # read days.jsonl, print calibrated SPOTS

Every archived spot-day is an experiment: we know the crowd, every truck's
price, quality and reputation, and how many meals each one actually sold.
The model in bot.py predicts those sales from the priors, so we search the
priors for the values that reproduce what really happened.
"""
import itertools
import json
import os
import sys

import bot

DAYS = os.environ.get("NIGHTSHIFT_DAYS", "days.jsonl")


def observations(path=DAYS):
    """One record per spot-day that had at least one truck parked."""
    obs = []
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return obs
    with f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rep_avg = {t["id"]: t.get("reputation_avg", 50.0) for t in (r.get("trucks") or [])}
            my_rep = r.get("your_rep") or {}
            me = (r.get("you") or {}).get("id")
            for spot, s in (r.get("spots") or {}).items():
                if spot not in bot.SPOTS:
                    continue
                trucks = s.get("trucks") or []
                if not trucks:
                    # Still informative: everyone went hungry. No truck to fit.
                    continue
                rows = []
                for t in trucks:
                    rep = my_rep.get(spot, 50.0) if t["id"] == me else rep_avg.get(t["id"], 50.0)
                    rows.append({
                        "quality": t.get("quality", "standard"),
                        "price": float(t.get("price", 11.0)),
                        "meals": float(t.get("meals") or 0),
                        "sold": float(t.get("sold") or 0),
                        "rep": float(rep),
                    })
                obs.append({
                    "spot": spot,
                    "customers": float(s.get("customers", 0)),
                    "hungry": float(s.get("hungry", 0)),
                    "trucks": rows,
                })
    return obs


def predict(ob, cfg):
    """Predicted sales per truck at one spot-day, under candidate priors."""
    saved = bot.SPOTS[ob["spot"]]
    bot.SPOTS[ob["spot"]] = cfg
    try:
        cells = bot.cells_for(ob["spot"], 24)
        trucks = [
            bot.Truck(
                bot.QVALUE[t["quality"]] * bot.repfactor(t["rep"]) - t["price"],
                t["price"], bot.QRANK[t["quality"]], t["meals"])
            for t in ob["trucks"]
        ]
        return bot.allocate(cells, trucks, ob["customers"])
    finally:
        bot.SPOTS[ob["spot"]] = saved


def error(obs, cfg):
    """Squared error in meals, per spot-day, on each truck and on the total."""
    total = 0.0
    for ob in obs:
        pred = predict(ob, cfg)
        for p, t in zip(pred, ob["trucks"]):
            total += (p - t["sold"]) ** 2
        served_pred = sum(pred)
        served_act = ob["customers"] - ob["hungry"]
        total += (served_pred - served_act) ** 2
    return total


def fit_spot(spot, obs):
    """Coordinate search over the mean, spread and the premium-minimum share."""
    mine = [o for o in obs if o["spot"] == spot]
    if not mine:
        return bot.SPOTS[spot], 0, None
    base = bot.SPOTS[spot]
    best = (error(mine, base), base)
    means = [round(x * 0.25, 2) for x in range(36, 61)]        # $9.00 - $15.00
    sds = [round(x * 0.25, 2) for x in range(4, 21)]           # $1.00 - $5.00
    prem = [round(x * 0.05, 2) for x in range(0, 13)]          # 0% - 60% premium-min
    cheapshare = [round(x * 0.05, 2) for x in range(0, 15)]    # 0% - 70% cheap-ok

    for _ in range(3):
        for m in means:
            cfg = dict(best[1], limit_mean=m)
            e = error(mine, cfg)
            if e < best[0]:
                best = (e, cfg)
        for s in sds:
            cfg = dict(best[1], limit_sd=s)
            e = error(mine, cfg)
            if e < best[0]:
                best = (e, cfg)
        for pc in prem:
            for ch in cheapshare:
                if pc + ch > 1.0:
                    continue
                cfg = dict(best[1], qmin=(ch, round(1.0 - ch - pc, 2), pc))
                e = error(mine, cfg)
                if e < best[0]:
                    best = (e, cfg)
    return best[1], len(mine), (error(mine, base), best[0])


def main():
    obs = observations()
    if not obs:
        sys.exit("no observations in %s yet" % DAYS)
    print("%d spot-days observed\n" % len(obs))
    out = {}
    for spot in bot.SPOTS:
        cfg, n, errs = fit_spot(spot, obs)
        out[spot] = cfg
        if errs:
            rmse_before = (errs[0] / max(1, n)) ** 0.5
            rmse_after = (errs[1] / max(1, n)) ** 0.5
            print("%-9s n=%-3d rmse %6.1f -> %6.1f meals   mean $%.2f sd $%.2f qmin %s" % (
                spot, n, rmse_before, rmse_after, cfg["limit_mean"], cfg["limit_sd"], cfg["qmin"]))
        else:
            print("%-9s no data, keeping prior" % spot)

    print("\n# paste into bot.py\nSPOTS = {")
    for spot, cfg in out.items():
        print('    "%s": {"limit_mean": %.2f, "limit_sd": %.2f, "qmin": (%.2f, %.2f, %.2f)},' % (
            spot, cfg["limit_mean"], cfg["limit_sd"], *cfg["qmin"]))
    print("}")


if __name__ == "__main__":
    main()

"""Step 2: decompose the day-1 regression with controlled component swaps.

The day-1 diagnostic showed the two prior sets differ on exactly two
components: price (+$0.25) and meals (-2). This plays each variant for real,
one fresh market per (seed, variant), so every outcome is a realised outcome
under matched conditions. Logged sales from a different order are NOT used as
a counterfactual -- the whole point is that at a value cliff they are not one.

Matching is asserted, not assumed: each market's day-1 forecast and rival
field are compared against the first market opened for that seed, and any
mismatch is reported rather than averaged away.

    python swap_day1.py --out swap_day1.json
"""
import argparse
import json
import time

import ftlab


def field_signature(st):
    """What day 1 looks like before anyone orders: crowd and the rival field."""
    fc = (st.get("forecast") or {}).get("expected_customers") or {}
    me = (st.get("you") or {}).get("id")
    trucks = sorted((t.get("id"), t.get("cash")) for t in st.get("trucks") or []
                    if t.get("id") != me)
    return {"forecast": {k: fc[k] for k in sorted(fc)}, "trucks": trucks,
            "events": sorted((e.get("name"), e.get("spot"))
                             for e in (st.get("forecast") or {}).get("events") or [])}


def rival_day1(st_after):
    last = st_after.get("last_day") or {}
    out = []
    for spot, sp in sorted((last.get("spots") or {}).items()):
        for t in sp.get("trucks") or []:
            out.append((spot, t.get("id"), t.get("price"), t.get("quality"),
                        t.get("meals"), t.get("sold")))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default="diag_day1_tune.json")
    ap.add_argument("--out", default="swap_day1.json")
    ap.add_argument("--rivals", type=int, default=3)
    args = ap.parse_args()

    ftlab.guard()
    diag = json.load(open(args.diag))

    rows = []
    for d in diag:
        seed = d["seed"]
        a = d["by_priors"]["orig"]["order"]
        b = d["by_priors"]["rmse"]["order"]
        spot, quality = a["spot"], a["quality"]

        variants = []
        # the price ladder, held at the orig inventory
        for price in (9.75, 10.00, 10.25, 10.50, 10.75, 11.00):
            tag = "ladder@%.2f" % price
            if abs(price - a["price"]) < 1e-9:
                tag = "A orig (price+meals)"
            elif abs(price - b["price"]) < 1e-9:
                tag = "D rmse-price, orig-meals"
            variants.append((tag, {"spot": spot, "quality": quality,
                                   "price": price, "meals": a["meals"]}))
        # and the two that move inventory to the rmse level
        variants.append(("B rmse (price+meals)",
                         {"spot": spot, "quality": quality, "price": b["price"],
                          "meals": b["meals"]}))
        variants.append(("C orig-price, rmse-meals",
                         {"spot": spot, "quality": quality, "price": a["price"],
                          "meals": b["meals"]}))

        sig0 = None
        for tag, order in variants:
            t0 = time.time()
            base = ftlab.new_market(seed, 30, args.rivals)
            st = ftlab.night_state(base, want_day=1)
            sig = field_signature(st)
            if sig0 is None:
                sig0 = sig
            matched = (sig == sig0)
            ftlab.post_order(base, order)
            row = ftlab.resolved(base, 1)
            after = ftlab.state(base)
            rows.append({"seed": seed, "variant": tag, "order": order,
                         "profit": row.get("profit"), "sold": row.get("sold"),
                         "revenue": row.get("revenue"), "spoiled": row.get("spoiled"),
                         "matched_field": matched, "rival_day1": rival_day1(after),
                         "took_s": round(time.time() - t0, 2)})
            print("  %-8s %-26s $%-6.2f x%-3d -> sold %-4s profit $%8.2f %s"
                  % (seed, tag, order["price"], order["meals"], row.get("sold"),
                     row.get("profit") or 0.0, "" if matched else "  !! FIELD MISMATCH"))

    json.dump(rows, open(args.out, "w"), indent=1)
    print("\nwrote %s  (%d markets)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()

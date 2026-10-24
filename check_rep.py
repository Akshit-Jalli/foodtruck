"""Verify the reputation reconstruction against ground truth.

Our own reputation is reported per spot in state.you.reputation, so replaying
the public rules over last_day must reproduce it exactly. If it does, the same
replay applied to rivals is sound too - that is the whole basis for trusting
MEM.rep_at() for trucks whose per-spot reputation we never get told.

    python check_rep.py [days]
"""
import sys

import bot


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    hist = []
    final = bot.run_practice(seed="repcheck-1", days=days, rivals=3,
                             quiet=True, history=hist)
    if not final:
        sys.exit("could not run a practice market")

    # The market ends without another decide(), so the closing day is still
    # unfolded. Fold it before comparing.
    bot.MEM.sync(final)

    me = final["you"]["id"]
    truth = final["you"]["reputation"]
    mine = bot.MEM.rep.get(me, {})
    print("days played: %d   memory holds %d days\n" % (len(hist), bot.MEM.days))
    print("%-10s %10s %10s %8s" % ("spot", "actual", "rebuilt", "diff"))
    worst = 0.0
    for spot in sorted(truth):
        a = truth[spot]
        b = mine.get(spot, float("nan"))
        d = abs(a - b)
        worst = max(worst, d)
        print("%-10s %10.3f %10.3f %8.3f" % (spot, a, b, d))
    print("\nworst error: %.4f reputation points" % worst)

    # The same replay, now applied to the rivals we cannot see directly.
    print("\nreconstructed rival reputation (no ground truth available):")
    for t in final["trucks"]:
        if t["id"] == me:
            continue
        reps = bot.MEM.rep.get(t["id"], {})
        if not reps:
            continue
        avg = sum(reps.values()) / len(reps)
        print("  %-22s avg rebuilt %6.2f   reported avg %6.2f   %s" % (
            t["truck_name"], avg, t.get("reputation_avg", float("nan")),
            " ".join("%s=%.0f" % (s[:4], r) for s, r in sorted(reps.items()))))

    if worst > 0.01:
        print("\nFAIL: reconstruction does not match ground truth")
        return 1
    print("\nPASS: reconstruction matches our own reputation exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())

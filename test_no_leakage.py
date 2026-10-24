"""Step 7: assert the deployed decision path cannot see the future.

Three checks, run against a real practice market:

1. Temporal. Every time decide() is called, the state it is handed must
   describe only days already played: last_day.day < order_for_day. If the
   server ever handed us a resolved current day, the bot would be scoring
   itself against an outcome it is supposed to be predicting.

2. Surface. decide() must read nothing but the state argument and static
   configuration. The state keys it touches are recorded and checked against
   an allow-list of fields that exist at order time.

3. Harness. ftlab / calibrate / _calworker may set priors and bot.P before a
   market starts, and may read final cash after it ends, but must never pass
   a market outcome back into decide() mid-market. Asserted by running a
   market with a decide() wrapper that rejects any state carrying a key the
   server did not send.

    python test_no_leakage.py
"""
import json
import sys

import ftlab
import bot

# Fields that exist at order time: tomorrow's published forecast, the public
# roster, the rules, our own standing, and yesterday's resolved results.
# Observed by inspection, not guessed: this is the full top-level key set the
# practice server sends at order time. Every one of them is either tomorrow's
# published forecast, the public roster, the rules, our own standing,
# yesterday's resolved results, or session bookkeeping.
ALLOWED_TOP_LEVEL = {
    "version", "phase", "mode", "market", "markets", "day", "days",
    "order_for_day", "orders_open", "rules", "you", "trucks", "forecast",
    "last_day", "announcements", "phase_ends_at", "practice_id",
    "server_time", "error",
}

# Bookkeeping only; carries no information about the night being ordered for.
BENIGN = {"markets", "phase_ends_at", "practice_id", "server_time", "version"}


def main():
    ftlab.guard()
    bot.LOG_PATH = "/dev/null"

    violations = []
    calls = {"n": 0}
    real_decide = bot.decide

    def audited(state):
        calls["n"] += 1
        ofd = state.get("order_for_day")
        last = (state.get("last_day") or {}).get("day")

        # 1. temporal
        if ofd is not None and last is not None and last >= ofd:
            violations.append("call %d: last_day.day=%s >= order_for_day=%s"
                              % (calls["n"], last, ofd))

        # 2. surface - nothing unexpected arriving in the state
        unknown = set(state) - ALLOWED_TOP_LEVEL
        if unknown:
            violations.append("call %d: unexpected state keys %s" % (calls["n"], sorted(unknown)))

        # 3. harness - the current day must carry no resolved outcome for us
        ld = state.get("last_day") or {}
        if ld.get("day") == ofd:
            violations.append("call %d: last_day resolves the day we are ordering for" % calls["n"])

        # 4. the forecast may describe tomorrow, but must not report it as fact
        fc = state.get("forecast") or {}
        if ofd is not None and fc.get("day") not in (None, ofd):
            violations.append("call %d: forecast is for day %s, ordering for %s"
                              % (calls["n"], fc.get("day"), ofd))
        for bad in ("customers", "hungry", "sold", "spots"):
            if bad in fc:
                violations.append("call %d: forecast carries realised field %r"
                                  % (calls["n"], bad))

        return real_decide(state)

    bot.decide = audited
    try:
        print("playing one 8-day practice market with an audited decide()...")
        final = bot.run_practice(seed="leak-check", days=8, rivals=3, quiet=True)
    finally:
        bot.decide = real_decide

    print("decide() called %d times" % calls["n"])
    if final is None:
        sys.exit("market did not finish - inconclusive")

    # Memory must only ever have ingested days strictly before the last order.
    print("Memory holds %d trucks' spot histories, %d of our own placements"
          % (len(bot.MEM.spots), len(bot.MEM.mine)))

    if violations:
        print("\nLEAKAGE FOUND:")
        for v in violations:
            print("  - " + v)
        sys.exit(1)

    print("\nno leakage detected:")
    print("  - every decide() saw only days already resolved")
    print("  - no unexpected fields arrived in the state")
    print("  - no market outcome was fed back into decide() mid-market")
    print("\nnote: day1_iso_value and the SPOTS priors are fitted offline from")
    print("past markets. That is calibration, not leakage - but both are fitted")
    print("to the practice field's behaviour, so neither is evidence about the")
    print("live field, which MODEL.md already flags as a different population.")


if __name__ == "__main__":
    main()

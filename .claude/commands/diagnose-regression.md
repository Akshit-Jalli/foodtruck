---
description: Decompose a regression into the decision component that causes it
---
Something got better on a proxy metric and worse on money. Decompose it before
theorising. $ARGUMENTS names the two configurations.

1. **Find where they diverge.** Compare mean profit per day between the two
   configurations across the per-run logs in `bench_results/logs/<label>/`.
   Report the cumulative difference by day. If one day carries most of it, the
   problem is that day, not the model.

2. **Diff the chosen action** on the divergent day, component by component
   (spot, price, quality, meals), with `diag_day1.py` if it is day 1.

3. **Play each one-component swap for real**, one fresh market per (seed,
   variant), as `swap_day1.py` does. Assert the field matched across variants
   instead of assuming it. **Do not use logged sales from a different order as
   a counterfactual** — at a value cliff they are not one.

4. **Attribute the gap**: what share does each component carry? Check that the
   endpoints reproduce the benched figures; if they do not, the harness is
   wrong before the model is.

5. **Then** look for the mechanism. For every rival compute
   `value = QVALUE[quality] * repfactor(rep) - price` and compare it to ours.
   `beats()` gives ties to the cheaper truck, so landing exactly on a rival's
   value loses every contested cell rather than splitting it.

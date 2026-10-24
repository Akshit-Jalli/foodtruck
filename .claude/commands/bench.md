---
description: Bench a change against baseline on the tune seeds, paired, with honest statistics
---
Bench the current working tree against `baseline` on the tune seeds.

```
python bench.py --engines baseline,bot --n 8 --seeds tune --label $ARGUMENTS
```

Then report, in this order:

1. The **per-seed paired differences**, not just the means.
2. Mean difference, sd, paired t, df, two-sided p and the 95% CI.
3. If the CI spans zero, say so plainly. Do not describe a result as an
   improvement when it does.
4. How many seeds would be needed for 80% power at the effect size actually
   measured: `n = (1.96 + 0.8416)^2 * sd^2 / delta^2`.

Move exactly one thing per run. If more than one knob changed, bench them
separately or the result is uninterpretable.

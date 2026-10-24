---
description: The merge gate - prove a configuration wins on untouched holdout seeds
---
Run the holdout gate for the configuration in $ARGUMENTS.

```
python bench.py --engines bot --n 8 --seeds hold --ablate "<config>" --label gate_<name>
python bench.py --engines bot --n 8 --seeds hold --label gate_default
python compare_runs.py --base gate_default gate_default gate_<name>
```

`hold-*` seeds exist only for this. If they have been used to choose anything,
say so and stop — the gate is void.

Report mean difference, sd, t, p, 95% CI and the per-seed table.

**Do not recommend merging unless it wins on `hold-*`.** If the holdout gain is
much smaller than the gain on train or tune seeds, that gap is selection noise
being stripped out; quantify how much and say so. A 24-evaluation search over
8 shared seeds has an evaluation spread around $835 (sd ~$235), so its maximum
looks good whether or not anything real happened.

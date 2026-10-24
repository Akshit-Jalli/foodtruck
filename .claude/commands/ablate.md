---
description: Sweep one bot.P knob alone across several values and compare paired by seed
---
Sweep the knob named in $ARGUMENTS across a sensible range, one value per bench
run, everything else held fixed.

```
for v in <values>; do
  python bench.py --engines bot --n 8 --seeds tune --ablate "<knob>=$v" --label "<knob>_$v"
done
python compare_runs.py --base <knob>_<current> <all labels>
```

Rules:

- One knob. Never bundle. `abl_smoke` moved two at n=1 and proved nothing.
- Include the current value as the comparison base so every number is a delta
  from today's behaviour.
- Report the paired table from `compare_runs.py`, and say whether the response
  is monotone. A monotone sweep is much stronger evidence than one good point.
- A winner on `tune-*` is a candidate, not a decision. Send it to `/gate`.

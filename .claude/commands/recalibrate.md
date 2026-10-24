---
description: Refit the customer priors, demand-first then money-second
---
Two loops. They optimise different things; do not conflate them.

**Demand fit** (what reproduces observed sales):

```
python harvest.py                 # designed experiments -> probe_days.jsonl
python fit.py --folds 4           # -> calibration.json, with out-of-sample RMSE
```

Report out-of-sample RMSE against both the current and the documented priors.

**Then stop.** An RMSE improvement is a *hypothesis* about profit, not
evidence of it. Before anything is pasted into `bot.py`:

```
python bench.py --engines bot --n 8 --seeds tune --ablate "priors='rmse'" --label cand_rmse
python compare_runs.py --base d1_orig_old d1_orig_old cand_rmse
```

A 13.4% RMSE improvement once cost $238/market — see `DECISION_CALIBRATION.md`.

**Money fit** (what actually maximises cash), if the demand fit does not win:

```
python calibrate.py --evals 24 --parallel 4 --seeds cal --set day1_iso_value=1.50
```

Train on `cal-*` only, then send the result to `/gate`. Keep the
parameterisation low-dimensional and regularised toward the established
priors; 24 free parameters against 8 noisy markets fits the seeds, not the
market.

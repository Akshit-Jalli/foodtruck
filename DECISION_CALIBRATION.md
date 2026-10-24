# The fitted-priors regression: diagnosis, cause, fix

Why the RMSE-fitted customer priors improved demand prediction and cost
$238/market, what actually went wrong, and what the evidence says to do.

Companion to [MODEL.md](MODEL.md), which describes the model itself.

## Summary

The regression was not a calibration failure. It was a **tie-break on one
night**.

The fitted priors moved the day-1 price by one $0.25 step, from $10.25 to
$10.50. At $10.50 our offer lands on *exactly* the value the rival field
opens at, and `beats()` gives a tie to the cheaper truck — so instead of
splitting the contested customers we lose all of them. Half the crowd
disappears for twenty-five cents.

| | |
|---|---|
| Day 1 share of the 30-day regression | **107%** |
| Days 2–30 combined | **+$15.78** (slightly *better*) |
| Cause, by controlled swap | price 103%, inventory 3% |
| Fix | correct the day-1 rival prior |
| Measured effect of the fix on the fitted priors | **+$230.59/market, 8/8 seeds, p < 0.0001** |

With the fix in place the fitted priors are statistically indistinguishable
from the established ones (−$13.16, p = 0.79, 3–5), rather than $244 behind.

## 1. Day-1 diagnostic

[diag_day1.py](diag_day1.py) opens one practice market per seed, takes the
day-1 night state, and runs `decide()` under both prior sets with the
deadline lifted so that timing cannot confound the comparison. Nothing is
ordered, so both runs see identical inputs.

Day 1 is the clean experiment: `Memory` is empty, so the order is a pure
function of the priors and the opening state.

Result, over all eight `tune-*` seeds:

```
spot     changed on 0/8 seeds
quality  changed on 0/8 seeds
price    changed on 8/8 seeds     always +$0.25  ($10.25 -> $10.50)
meals    changed on 8/8 seeds     always -2
```

Perfectly consistent. One price step and two meals, every seed.

## 2. Decomposing it

[swap_day1.py](swap_day1.py) plays each component for real — one fresh market
per (seed, variant), 64 markets — because at a value cliff a logged sale from
a different order is not a valid counterfactual. Matching is asserted, not
assumed: every market's day-1 forecast and rival field is compared against
the first market opened for that seed. All 64 matched, and the rival field was
byte-identical across our eight variants on all eight seeds, confirming that
day-1 rival choices do not react to our order.

Mean over 8 seeds:

| variant | profit | sold |
|---|---|---|
| A  orig price, orig meals | $325.09 | 50.4 |
| C  orig price, **RMSE meals** | $317.22 | 48.9 |
| D  **RMSE price**, orig meals | $79.12 | 25.8 |
| B  RMSE price, RMSE meals | $86.62 | 25.8 |

```
total A -> B                      -238.47
attributable to PRICE  (A -> D)   -245.97   (103% of the gap)
attributable to MEALS  (A -> C)     -7.88   (3%)
interaction                        +15.38
```

The price step carries all of it. The inventory change is noise. Note that
A → B reproduces the bench's day-1 figures ($325.10 → $86.60) to within four
cents, which is the harness validating itself.

The price ladder at fixed inventory shows the shape:

| price | sold | profit |
|---|---|---|
| $9.75 | 51.0 | $306.00 |
| $10.00 | 50.5 | $313.75 |
| $10.25 | 50.4 | **$325.09** |
| $10.50 | 25.8 | $79.12 |
| $10.75 | 24.6 | $73.47 |
| $11.00 | 23.8 | $70.00 |

Not a demand curve. A cliff between $10.25 and $10.50, and the established
priors happen to stand on the last profitable step of it.

## 3. The cause

Every rival offer on night one sits on an **exact iso-value line of +1.50**:

| spot | offer | value at rep 50 |
|---|---|---|
| highst | cheap @ $6.50 | 8.00 − 6.50 = **1.50** |
| union | standard @ $10.50 | 12.00 − 10.50 = **1.50** |
| thompson | premium @ $14.50 | 16.00 − 14.50 = **1.50** |

This is not a coincidence of one seed. Across the whole probe dataset, **all
24 day-1 rival offers in all 8 harvested markets sit at exactly +1.50**. By
day 2 they spread out (1.356, 1.5, 1.644, 1.98) and by day 5 they are fully
dispersed, so it is specifically an opening convention.

Our value at standard quality and reputation 50 is `12.00 − price`:

- $10.25 → **+1.75** > 1.50 → we win every cell we qualify for → ~50 sold
- $10.50 → **+1.50** = 1.50 → *tie*, and `beats()` breaks ties to the cheaper
  truck. The rival at High Street is $6.50. We lose every contested cell.

And `predict_rivals` could not see any of this, because on night one it has no
history and fell back to a flat guess of **$11.00 standard — value +1.00**.
That told the model it won comfortably anywhere below $11.00. The true
threshold is strictly below $10.50. Both prior sets were blind to the cliff;
the established ones simply landed on the safe side of it by luck.

## 4. The fix

`P["day1_iso_value"] = 1.50` (set to `None` to restore the old guess). The
night-one fallback now places rivals on the measured iso-value line, cycling
quality and pricing each to that value, instead of a flat $11.00/standard.

Benched over the eight `tune-*` seeds, one knob at a time
([compare_runs.py](compare_runs.py)):

| config | mean$ | vs orig_old | p | won–lost |
|---|---|---|---|---|
| orig priors, old fallback | 9877.19 | — | — | — |
| orig priors, **fix** | 9877.19 | +0.00 | — | identical on every seed |
| RMSE priors, old fallback | 9633.44 | −243.75 | 0.0023 | 0–8 |
| RMSE priors, **fix** | 9864.03 | −13.16 | 0.79 | 3–5 |

The fix applied to the fitted priors alone: **+$230.59, sd 72.49,
t = 8.997, df 7, p < 0.0001, won 8 of 8.**

It is a no-op for the established priors, which already price below the cliff.
That is the expected signature of a fix that removes a trap rather than
changing strategy.

## 5. Decision-focused calibration

The framing matters, so it is worth stating what was and was not done.

**Why not SPO+.** Elmachtoub and Grigas derive the SPO+ surrogate for
decisions of the form `argmin` over a *fixed convex feasible region* of a cost
that is *linear in the predicted parameter*. Our decision is an argmax over a
discrete (spot × quality × price) grid, and realised profit is not linear in
predicted demand: it passes through a newsvendor with a CVaR tail penalty and
a capacity-constrained allocation whose profit surface is piecewise constant,
with cliffs exactly where our offer crosses a rival's value — §3 is a
measurement of one such cliff. None of the structure SPO+ needs is present,
so its convexity and consistency results would not transfer. Implementing it
mechanically would have produced a surrogate with no claim on this geometry.

**What was done instead**, closer to the task-based end-to-end learning of
Donti, Amos and Kolter: optimise the real decision loss as a black box.
[calibrate.py](calibrate.py) scores a candidate prior set by *playing* markets
with the existing optimiser untouched in the loop, and searches with a
derivative-free compass method.

Three design choices worth defending:

- **Low-dimensional.** Three scalars, not 24. `alpha` blends established →
  fitted, `delta` shifts every `limit_mean`, `kappa` scales every `limit_sd`.
  Twenty-four free parameters against eight noisy market outcomes would fit
  the seeds, not the market.
- **Common random numbers.** Every evaluation plays the same seeds, so two
  thetas differ by a paired difference rather than two independent draws.
- **Regularised toward the established priors.** `theta = (0, 0, 1)` is the
  proven configuration; the penalty is what the search must beat to leave it.

Seed hygiene: the search sees only `cal-*`. `tune-*` is reserved for choosing
between finalists and `hold-*` is untouched until the merge gate.

### What it found

24 evaluations, 192 markets, 18.2 minutes, with the day-1 fix held fixed so
the search could not simply rediscover "do not price onto the cliff".

| | alpha | delta | kappa | train cash |
|---|---|---|---|---|
| start (established priors) | 0.000 | +0.00 | 1.00 | $10025.19 |
| best objective | 0.125 | +0.75 | 1.00 | **$10595.35** |

Almost all of the movement is `delta`: a **$0.75 global shift upward in every
spot's `limit_mean`**. `alpha` settled at 0.125, i.e. the search stayed close to
the established priors and did *not* want the RMSE fit. `kappa` did not move
at all.

Read this with care. The reported gain is +$570.16 over the start point, but it
is the maximum of 24 evaluations whose own spread is $835.41 (sd $235.00) on eight
shared seeds. The expected maximum of 24 draws that noisy is inflated by
several hundred dollars even when the true effect is zero. On the train seeds
this is a candidate, not a result, which is exactly why step 6 of the brief
demanded an untouched holdout.

### The holdout gate

Registered as `PRIOR_SETS["dfl"]` and run against seeds the search never saw.

| config | seeds | mean$ | vs orig | p | won-lost |
|---|---|---|---|---|---|
| orig | hold | 10128.62 | - | - | - |
| **dfl** | **hold** | **10449.25** | **+320.62** | **0.0026** | **7-1** |
| rmse + day-1 fix | hold | 10086.22 | -42.41 | 0.73 | 6-2 |
| **dfl** | tune | **10325.44** | **+448.25** | **<0.0001** | **8-0** |
| dfl | cal (train) | 10595.35 | +570.16 | - | - |

The calibrated priors **pass the gate**: +$320.62 on eight seeds the search
never saw, p = 0.0026, winning 7 of 8.

The shrinking sequence across seed sets is the signature to look for:

```
train       +570.16      <- inflated: the max of 24 evaluations on these seeds
validation  +448.25
holdout     +320.62      <- the honest number
```

About $250 of the train-set gain was selection. What survives is real.

Also worth recording: the RMSE priors, with the day-1 fix, come out at
-$42.41 on holdout with p = 0.73. They are neither better nor worse than the
established priors once the cliff is removed, which is the final confirmation
that the original $238 regression was a tie-break and not a calibration
failure.

### What the search actually learned

`alpha` settled at 0.125 and `kappa` did not move, so the answer is not "the
RMSE fit was right after all". Essentially all of the gain is `delta = +0.75`:
a uniform $0.75 upward shift in every spot's `limit_mean`. Customers pay more
than both prior sets believed. That raises predicted demand at every price,
which the newsvendor turns into more inventory - and the measured stockout
rate falls from 0.667 to 0.517 while unused units rise from 3.9 to 5.7/day.

Which points at the same conclusion as §8 from a second direction: this bot
was under-stocking.

## 8. The two follow-up ablations

Run one knob at a time on `tune-*`, as the brief required.

### risk_lambda was over-tuned

| risk_lambda | mean$ | vs 0.55 | p | won-lost | stockout | unused/day |
|---|---|---|---|---|---|---|
| **0.0** | **10243.78** | **+366.59** | **0.0039** | **7-1** | 0.371 | 9.9 |
| 0.275 | 10125.47 | +248.28 | 0.0370 | 6-2 | 0.479 | 8.2 |
| 0.55 *(current)* | 9877.19 | - | - | - | 0.629 | 4.3 |
| 0.825 | 9835.81 | -41.38 | 0.1769 | 2-6 | 0.633 | 4.1 |
| 1.1 | 9580.59 | -296.59 | 0.0004 | 0-8 | 0.692 | 3.3 |

Cleanly monotone, which is far stronger evidence than a single good point.
The CVaR tail penalty costs money against this field: it is charged for the
spoilage that over-stocking causes, so it pushes inventory down, and the bot
then sells out on two nights in three. Turning it off raises waste from $17.70
to $40.70 a day and still nets +$366.

Note this is a *practice-field* result. The penalty is insurance against a
bad tail that the house bots never inflict. Before setting it to zero in the
live game, bench it in a larger field, where the tail is real.

### Spot concentration is not a problem

The bot spends 200 of 240 trading days at High Street and switches spots 7.4
times in 30 days against baseline's 16.5. Its per-day profit at Union ($398)
and Oval ($411) both exceed High Street ($326), which looks like money left
on the table.

It is not. Forcing diversification loses:

| spot_repeat_penalty | mean$ | vs off | p | won-lost |
|---|---|---|---|---|
| 0 *(off)* | 9877.19 | - | - | - |
| 100 | 9654.56 | -222.62 | 0.0192 | 2-6 |
| 300 | 9590.28 | -286.91 | 0.0061 | 2-6 |

The per-visit gap was selection, not opportunity: the bot goes to Union and
Oval only on the nights those spots look unusually good, so their average is
conditioned on being good. Profit at High Street also does not decay with
streak length ($349 / $302 / $330 / $279 / $369 / $328 for streaks 1 to 6+),
so it is not saturating its own spot either. Question settled; the
`spot_repeat_penalty` knob stays at 0 and exists only to have answered it.

## 9. Recommendation

| change | evidence | call |
|---|---|---|
| `day1_iso_value = 1.50` | +$230.59 on the fitted priors, 8/8, p<0.0001; no-op on established priors | **merge** |
| `priors = 'dfl'` | +$320.62 on untouched holdout, 7/8, p=0.0026 | **merge** |
| `risk_lambda` 0.55 -> 0.0 | +$366.59 on tune, monotone sweep, p=0.0039 | **gate on hold-\* first** |
| `priors = 'rmse'` | -$42.41 on holdout, p=0.73 | do not merge; keep for reference |
| `spot_repeat_penalty` | -$222 to -$287 on tune | do not merge; leave at 0 |

The first two have cleared the bar the brief set. `risk_lambda = 0.0` has only
been swept on validation seeds and should pass `hold-*` before it goes in -
and ideally a larger field too, since a tail penalty is insurance and this
field inflicts no tail.

### The general fix still outstanding

`day1_iso_value` removes the tie-break trap on night one only. The trap itself
is general: `beats()` gives ties to the cheaper truck, so on *any* night our
offer lands exactly on a modelled rival's value we lose every contested cell
instead of splitting it. A guard that refuses a price whose value ties a
modelled rival - stepping to the next grid point below instead - would protect
all thirty nights. Not built, not benched. It is the highest-value piece of
work left, because the failure it prevents cost $246 in a single night.


## 6. Leakage audit

[test_no_leakage.py](test_no_leakage.py) wraps `decide()` for a whole market
and asserts, on every call, that

1. the state describes only days already played (`last_day.day < order_for_day`),
2. no unexpected top-level field arrives — the allow-list was built by
   inspecting the live state, not guessed,
3. `last_day` never resolves the day being ordered for,
4. the forecast is for the day being ordered for and carries no realised
   field (`customers`, `hungry`, `sold`, `spots`).

Passes. Every state key the decision path reads is tomorrow's published
forecast, the public roster, the rules, our own standing, or yesterday's
resolved results. `Memory` ingests only `last_day`. The harness sets priors
and `bot.P` before a market starts and reads cash after it ends; nothing is
fed back mid-market.

One honest caveat: `day1_iso_value = 1.50` and the `SPOTS` priors are both
fitted offline against the practice field. That is calibration, not leakage —
but it is calibration to *this* opponent population, so it is not evidence
about the live field, which MODEL.md §6 already flags as a different animal.
If the live field opens somewhere other than +1.50, the constant is wrong
there; the mechanism (never price onto a rival's value line) is what
generalises, not the number.

## 7. Two incidental findings

**Rival per-spot reputation is public.** MODEL.md §3 states that "rival
reputation is private — the server reports only a six-spot average". It does
not. Every entry in `state["trucks"]` carries a full per-spot `reputation`
dict. `Memory`'s reconstruction is *exact* — audited at 234 (rival, spot)
values over 13 nights, 0 mismatches, worst error 0.00 points — so this is a
simplification opportunity, not a bug. It would also close the "rivals enter
and leave invisibly" gap in MODEL.md §7 for free, since a direct read does
not depend on having observed the truck.

**`announcements` exists and pre-announces tomorrow's event in prose**, e.g.
"Tomorrow: Midterms. Thompson is packed and stressed students treat
themselves." It is redundant with `forecast.events`, which already carries the
same event structurally, so this is not a new information source — but it
confirms the dashboard's News panel is reading a real field.

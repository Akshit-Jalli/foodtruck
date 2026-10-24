# How the model works

A map of the decision engine in [bot.py](bot.py) — what it believes about the
world, how it turns those beliefs into a nightly order, and how the beliefs
themselves get calibrated.

## First, a naming note

There is no classifier here, in the machine-learning sense — nothing is trained
to sort inputs into labelled buckets, and there are no learned weights anywhere
in the repo. What the bot has is a **structural demand model** plus a
**per-night optimiser**. The game's rules are known, so instead of learning a
mapping from features to outcomes, the bot reconstructs the mechanism and
simulates it.

Two parts do classification-*like* work, and they are probably what the word
was reaching for:

- **Customer discretisation** (`cells_for`) — sorts the crowd into typed
  buckets by willingness to pay and minimum acceptable quality.
- **The choice rule** (`beats` / `allocate`) — assigns each bucket to whichever
  truck wins it.

Those two are the heart of the thing, so they get the most space below.

## 1. What a customer is

The game gives every customer two private numbers. The model mirrors that:

| Trait | Distribution | Meaning |
|---|---|---|
| Price limit | Normal(`limit_mean`, `limit_sd`) per spot | Won't pay above this |
| Minimum quality | Categorical over cheap/standard/premium (`qmin`) | Won't buy below this |

Those priors live in the `SPOTS` dict ([bot.py:49-56](bot.py#L49-L56)) and are
per-spot — the Stadium crowd pays more than North Res, and the Union skews
toward wanting at least standard food.

`cells_for(spot, n)` ([bot.py:110-121](bot.py#L110-L121)) turns that continuous
population into a finite set of **cells**, which is the unit the whole engine
works in. For each of the 3 quality tiers it slices the price-limit normal into
`n` equal-probability strata and takes each stratum's midpoint via the inverse
normal CDF:

    limit_i = limit_mean + limit_sd * invnorm((i + 0.5) / n)

With `limit_cells = 18` that is **3 x 18 = 54 cells**, each a triple of
`(weight, min_quality_rank, price_limit)`. Stratifying by equal probability
rather than on a uniform price grid puts the resolution where customers
actually are.

## 2. How a customer picks a truck

Every truck on a spot presents one offer. Its **value** to a customer is

    value     = QVALUE[quality] * repfactor(reputation) - price
    repfactor = 0.7 + 0.6 * reputation / 100      # rep 0..100 -> 0.7x .. 1.3x
    QVALUE    = cheap 8.0 | standard 12.0 | premium 16.0

A customer in a given cell considers a truck only if **quality rank >= their
minimum**, **price <= their limit**, and **value >= 0**. Among those they take
the highest value, ties going to the cheaper truck (`beats`,
[bot.py:277-283](bot.py#L277-L283)).

This is why the game is cliff-shaped rather than a smooth demand curve: value
is a single scalar, so the truck that wins a cell takes *all* of it. A rival
shaving 25c can flip whole cells at once. That one fact drives most of the
design decisions below.

`allocate(cells, trucks, n)` ([bot.py:286-326](bot.py#L286-L326)) runs the
scramble with capacity. Each round, every cell picks its favourite among trucks
that still have food; an oversubscribed truck serves a `cap/demand` fraction
and the rest of those customers spill to their next choice. Five rounds, or
until nothing spills. Our demand is whatever is *left over* once everyone
else's offer has been scored — the central modelling claim of the whole bot.

## 3. What we believe about rivals

Rival reputation is private — the server reports only a six-spot average — but
reputation **dynamics are deterministic and public**: a truck that sells >= 1
meal at a spot gains `QREP[quality]` there, otherwise its reputation at that
spot drifts 3% of the way back toward 50. So `Memory`
([bot.py:135-230](bot.py#L135-L230)) replays the public `last_day` feed and
recovers every rival's per-spot reputation exactly.

That claim is tested, not assumed. [check_rep.py](check_rep.py) replays the
same rules against *our own* reputation, which the server does report per spot,
and asserts agreement to 0.01 points.

On top of that, four hedges model what rivals might do tomorrow:

| Hedge | Flag | What it assumes |
|---|---|---|
| **Mobility** | `use_mobility` | Rivals park where they have been parking — exponentially-weighted spot frequency, 4-day half-life, Laplace-smoothed (`spot_probs`). A rival present 40% of the time is carried at 40% of its capacity there. |
| **Intruder** | `intruder_p = 0.45` | With p=0.45 the strongest truck in the field shows up at whatever spot we chose (`strongest_rival`). Deduped, so we never clone a truck onto a spot it already occupies. |
| **Price jitter** | `use_price_jitter` | Rivals don't repeat last night's price to the cent. The whole rival price vector shifts by -$0.60 / $0.00 / +$0.45 with weights 0.30 / 0.50 / 0.20. |
| **Herding** | `use_herding`, `herd_p = 0.40` | With p=0.40 the field ignores its own history and piles onto the night's obvious draw (`magnet_spot`: an event spot, or any spot forecast to hold > 28% of all customers). On a herd night a rival sits at the magnet with p=0.85 and anywhere else with p=0.05. |

The last two exist because the practice opponents and the live opponents are
different animals — see §6.

## 4. The scenario set

Those hedges multiply out into the distribution the optimiser integrates over
(`scenarios()`, [bot.py:479-490](bot.py#L479-L490)):

      5 crowd multipliers  (0.70 .. 1.35, weights .15/.20/.30/.20/.15)
    x 2 intruder states    (.55 / .45)
    x 3 price shifts       (.30 / .50 / .20)
    x 2 herd states        (.60 / .40)
    = 60 weighted scenarios per candidate

Every candidate pitch is scored against all 60, so what gets maximised is a
distribution over profit, never a point estimate.

## 5. Choosing the order

`decide()` ([bot.py:563-680](bot.py#L563-L680)) runs in two stages, because a
full capacity-aware simulation of every (spot x quality x price) combination is
far too slow for a 2-second order window.

**Stage 1 — screen.** For all 6 spots x 3 qualities x every price on a $0.25
grid, compute demand with `fast_demand`, which ignores sell-outs: a cell is
ours if we beat the best rival offer for it. Score each candidate by newsvendor
and keep the top `shortlist = 18`.

**Stage 2 — simulate.** For each of the 18 survivors, run the real `allocate`
over all 60 scenarios, with spill and sell-outs, for a proper demand
distribution. Then pick stock.

**Inventory** is a newsvendor with a tail penalty (`best_inventory`,
[bot.py:351-363](bot.py#L351-L363)). Expected profit is piecewise linear in
meals with kinks exactly at the scenario demands, so the optimum always sits on
one of those points and the search grid can be tiny. The objective is not
expected profit but

    score = mean - risk_lambda * (mean - CVaR_30%)        # risk_lambda = 0.55

mean profit penalised by 0.55x the gap to the average of the worst 30% of
outcomes. That is what trades a little mean for a much tighter downside.

**Reputation credit.** Selling today raises reputation, which is worth money on
future visits. Rather than guess a bonus, `best_at_rep` re-solves the best
achievable day-profit at that spot at today's reputation and again at
tomorrow's, and books the difference, multiplied by how often we actually
return there (`my_revisit_p`) discounted over the remaining horizon. The worth
of a reputation point is read off the game's own mechanics instead of assumed.

**Sitting out.** If the best option still loses money in expectation, the bot
orders 0 meals. Spoilage is real; doing nothing is free.

## 6. Calibration

The `SPOTS` priors are the only numbers in the model that come from data
rather than from the rulebook. They are fitted offline, in a three-step loop
that is run by hand between markets — nothing re-fits during play.

    harvest.py  ->  probe_days.jsonl  ->  fit.py  ->  edit SPOTS  ->  bench.py

### harvest.py — generating identification

Observational data from a *good* bot is useless for calibration: it always
parks near the same price, so the price-limit distribution is never probed.
[harvest.py](harvest.py) throws cash away and runs a designed experiment
instead.

The grid is spot x quality x price ladder — cheap 6 prices ($3.00–$7.75),
standard 8 ($4.00–$11.75), premium 11 ($5.00–$15.75), each ladder stopping just
below `QVALUE` where value hits zero at reputation 50. That is 25 cells per
spot, 150 overall. Each market shuffles its own queue and places one cell per
day, preferring a spot that was empty yesterday *and* a price we can actually
sell at — a spot where we are the only truck is the cleanest experiment there
is, because everyone who walks away reveals that their limit is under our price
or their minimum quality is over ours.

Meals are deliberately generous, `1.8x` the forecast crowd, so our own rows are
rarely censored by a sell-out.

The committed [probe_days.jsonl](probe_days.jsonl) holds **8 markets, 1440
spot-day rows, 240 placements covering 115 of the 150 cells**. Coverage is
aggregate, not balanced — each market reshuffles independently, so no cell is
guaranteed a fixed number of hits.

### fit.py — the estimator

[fit.py](fit.py) searches the priors for the values that best reproduce what
actually happened. Four free parameters per spot (`limit_mean`, `limit_sd`,
`qcheap`, `qpremium`; `qstandard` is implied), fitted independently per spot.
Everything else — QVALUE, repfactor, reputations, crowd — is held fixed.

Per spot-day with T trucks, predicted sales `p`, observed `sold`, crowd `C`
and hungry count `H`:

    L = [ sum_i (p_i - sold_i)^2  +  (sum_i p_i - (C - H))^2 ] / (T + 1)

The second term is the **served-total residual**: `customers - hungry` is the
total served at that spot, it constrains the whole population rather than one
truck, and the meal caps are already inside the model, so it is an exact
target.

The penalised objective adds shrinkage toward the current `SPOTS` values:

    J(theta) = data_loss + n0 * sigma^2 * sum_j ((theta_j - theta0_j) / s_j)^2

with `n0 = PSEUDO_N = 40`, scale `s = (1.5, 0.75, 0.12, 0.12)`, and `sigma^2`
estimated as the mean per-observation loss under the current priors. A spot
with few rows barely moves; a spot with hundreds moves a lot.

The optimiser is a hand-rolled **first-improvement compass search** — no
scipy, no gradients, no restarts. One sweep over the four coordinates trying
+/- step, halving all steps when a sweep fails, stopping at 400 evaluations.
Deterministic and local-only.

Validation is **k-fold leave-markets-out** (default 4 folds, stride slices over
the probe markets; live rows always stay in training). Each fold refits on the
training markets and scores held-out RMSE against both the current priors and
the documented ones. If the fit is not better out of sample, the report says so.

### Known gaps in the calibration code

Documented here because the docstrings currently oversell a few things:

- **The censoring correction is inert.** `predict` passes each truck's meal
  count as `Truck.cap` ([fit.py:165](fit.py#L165)) and `allocate` caps sales at
  `cap`, so predicted sales never exceed meals. For a sold-out truck the
  residual is therefore already `<= 0`, and the `min(r, 0.0)` at
  [fit.py:186](fit.py#L186) only strips floating-point noise. The fit cannot
  learn "this truck would have sold far more with stock" — that would need an
  uncapped demand prediction.
- **The served total is not weighted as the strongest signal** despite the
  docstring saying so; dividing by `T + 1` gives it exactly one truck's weight.
- **`sigma^2` is computed under the current `SPOTS`**, not the documented
  priors as the comment claims (`main()` does flag the drift).
- **The printed "shrinkage" figure is notional** — a conjugate weight
  `n0/(n0+n)` that the optimiser never uses. Real shrinkage comes only from
  the penalty term.
- **Latent crash**: `shrink = pseudo_n / (pseudo_n + n)` at
  [fit.py:224](fit.py#L224) runs *before* the `n == 0` guard, so a spot with no
  rows divides by zero during the unshrunk diagnostic fit. Not reachable with
  current data.
- `--holdout` is documented but does not exist; the flag is `--folds`.
- Market identity is `seed or practice_id`, so reusing `--prefix` across
  harvest runs would collide seeds and leak rows across CV folds.

### bench.py — confirming it helped

[bench.py](bench.py) replays the same seeded practice markets across engine
versions and pairs the results by seed. It refuses to touch the live endpoint:
the engine's HTTP helper is wrapped so any request outside `/api/v1/practice/`
raises instead of reaching the server.

The critical caveat is that **the practice field cannot validate every
change.** Measured over 576 rival day-to-day transitions in the probe data, the
house bots hold price and quality 100% of the time and spot 92.4% of the time,
and they move no more often when we park on top of them (6.9%) than when we do
not (7.7%). They neither re-price nor herd. So the `use_price_jitter` and
`use_herding` hedges can only ever *cost* money in bench — what bench measures
for them is the insurance premium, not the benefit. The live field does both
(see the day-3 record in [days.jsonl](days.jsonl), where all seven trucks piled
onto one event spot and rivals re-priced overnight).

## 7. What the model does *not* do

Worth being explicit, because these are the known edges:

- **No best-response reasoning.** Rival beliefs are built from observed history
  only. Nothing anticipates that a rival will react to *our* price or spot.
  There is no equilibrium computation and no fictitious play beyond the
  frequency counting in `spot_probs`.
- **No within-market learning of rival type.** Every rival gets the same
  mobility / jitter / herd treatment. A rival that is visibly a fixed-price
  camper and one that visibly undercuts nightly are modelled identically.
- **Rivals enter and leave invisibly.** `predict_rivals` only sees trucks
  present in `last_day`; a truck that sits out a night disappears from our
  beliefs entirely, even though `Memory` still holds its history.
- **Priors are static during play.** `SPOTS` is fixed at process start. Nothing
  re-fits mid-market; re-fitting is an offline step between runs.
- **The crowd forecast is taken at face value**, widened only by the five crowd
  multipliers — the bot never concludes the forecast itself is biased.

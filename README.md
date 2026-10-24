# Food Truck Wars — bot and experiment infrastructure

A first-party client and research harness for
[Builders Night Shift](https://nightshift.joinbuilders.org).

Six trucks compete for a campus crowd each night. Every customer has a private
price ceiling and a private minimum quality, and buys from whichever truck they
can afford that offers the best `value = quality_value × reputation_factor −
price`. Our sales are whatever is left once every rival's offer has been scored
against ours, customer by customer, until somebody sells out.

- [MODEL.md](MODEL.md) — how the decision engine works
- [DECISION_CALIBRATION.md](DECISION_CALIBRATION.md) — the fitted-priors
  regression: diagnosis, root cause, fix, and the holdout results

## Setup

No dependencies. Python 3.9+ and the standard library only (tested on 3.13).

```bash
git clone https://github.com/Akshit-Jalli/foodtruck.git
cd foodtruck
printf 'NIGHTSHIFT_KEY=fk_your_key_here\n' > .env.local   # gitignored
```

`.env.local` is matched by the `.env.*` rule in `.gitignore`. **Never commit a
key.** `ftlab.py` loads the file automatically; `bot.py`, `bench.py` and
`harvest.py` read the environment, so export it for those:

```bash
export NIGHTSHIFT_KEY="$(grep NIGHTSHIFT_KEY .env.local | cut -d= -f2)"
```

On Windows PowerShell:

```powershell
$env:NIGHTSHIFT_KEY = (Select-String -Path .env.local -Pattern 'NIGHTSHIFT_KEY=(.*)').Matches.Groups[1].Value
```

Verify:

```bash
python bot.py --practice        # one 30-day practice market, prints each night
```

### Environment variables

| Variable | Default | Used by |
|---|---|---|
| `NIGHTSHIFT_KEY` | *(required)* | everything that talks to the server |
| `NIGHTSHIFT_URL` | `https://nightshift.joinbuilders.org` | all clients |
| `NIGHTSHIFT_LOG` | `run_log.jsonl` | `bot.py`, `baseline.py` |
| `NIGHTSHIFT_PROBE` | `probe_days.jsonl` | `fit.py` |
| `NIGHTSHIFT_DAYS` | `days.jsonl` | `fit.py` |
| `DASH_PORT` | `8000` | `dashboard.py` |

## The pieces

### Playing

| Command | What it does |
|---|---|
| `python bot.py` | play the live game (leave running) |
| `python bot.py --practice` | one 30-day practice market, verbose |
| `python bot.py --bench 8` | 8 seeded practice markets, averaged |
| `python bot.py --set use_herding=False --practice` | override any tunable in `bot.P` |
| `python supervise.py` | keep the live bot running all night, restarting on crash |

`baseline.py` is a second, simpler engine kept as a control. Never delete it —
every performance claim in this repo is relative to it.

### Watching

```bash
python dashboard.py            # then open http://localhost:8000
python recorder.py             # archive every live day to days.jsonl
python explain.py              # why did the model choose that? sweeps the options
```

The dashboard is strictly read-only and never sends an order.

### Measuring

`bench.py` is the only thing that should ever decide whether a change helped.
It replays the same seeded practice markets across engine versions and pairs
the results by seed.

```bash
# bot vs baseline on the tuning seeds
python bench.py --engines baseline,bot --n 8 --seeds tune

# one knob moved, everything else held
python bench.py --engines bot --n 8 --seeds tune \
    --ablate "risk_lambda=0.0" --label rl_zero

# a bigger field - where the bot's edge is largest
python bench.py --engines baseline,bot --n 6 --seeds hold --rivals 8
```

Then compare whole runs against each other, paired by seed:

```bash
python compare_runs.py --base rl_0.55 rl_0.55 rl_0.0 rl_1.1
```

`bench.py` refuses to touch the live endpoint: the engine's HTTP helper is
wrapped so any request outside `/api/v1/practice/` raises instead of reaching
the server.

### Calibrating

Two separate loops. They optimise different things, and conflating them is
what caused the regression documented in `DECISION_CALIBRATION.md`.

**Demand calibration** — fits the priors to reproduce observed sales:

```bash
python harvest.py              # designed experiments -> probe_days.jsonl
python fit.py                  # -> calibration.json + a SPOTS block to paste
python fit.py --folds 4        # leave-markets-out cross-validation
```

**Decision calibration** — tunes the priors to maximise realised profit
through the optimiser:

```bash
python calibrate.py --evals 24 --parallel 4 --seeds cal --set day1_iso_value=1.50
```

`fit.py` minimises demand RMSE. `calibrate.py` maximises money. **A better fit
is not a better decision** — see `DECISION_CALIBRATION.md`, where a 13.4% RMSE
improvement cost $238/market.

### Diagnosing

```bash
python diag_day1.py --seeds tune        # what changes between two prior sets
python swap_day1.py                     # play each component swap for real
python test_no_leakage.py               # assert decide() cannot see the future
python check_rep.py                     # verify reputation reconstruction
```

## Seed discipline

Three disjoint sets. Keeping them apart is the only thing standing between us
and tuning on our own test set.

| Set | Seeds | May be used for |
|---|---|---|
| **train** | `cal-0` … `cal-7` | the calibration search, and nothing else |
| **validation** | `tune-0` … `tune-7` | choosing between finalists, sweeping knobs |
| **holdout** | `hold-0` … `hold-7` | the merge gate only |

**Do not merge a configuration into the default without a paired win on
`hold-*`.** Train-set gains are routinely several hundred dollars of pure
selection noise: a 24-evaluation search over 8 seeds has an evaluation spread
of ~$835 (sd ~$235), so its maximum looks good whether or not anything real
happened.

## Configuration

Everything tunable lives in `bot.P`, and anything in it can be moved from the
command line (`--set K=V`) or from the bench (`--ablate K=V`). The customer
priors are addressed by name through `bot.PRIOR_SETS`:

| Set | What it is |
|---|---|
| `orig` | hand-tuned from the rulebook. The proven default. |
| `rmse` | `fit.py`'s demand fit. Neutral once the day-1 bug is fixed. |
| `dfl` | `calibrate.py`'s profit-tuned result. **+$320.62 on holdout, p = 0.0026.** |

```bash
python bench.py --engines bot --n 8 --seeds hold --ablate "priors='dfl'"
```

### Measured so far

All paired over 8 seeds, one knob at a time.

| Change | Seeds | Effect | p | won–lost |
|---|---|---|---|---|
| `priors='dfl'` | **hold** | **+$320.62** | 0.0026 | 7–1 |
| `risk_lambda=0.0` (from 0.55) | tune | **+$366.59** | 0.0039 | 7–1 |
| `day1_iso_value=1.50` (on `rmse` priors) | tune | **+$230.59** | <0.0001 | 8–0 |
| `priors='rmse'` with the day-1 fix | hold | −$42.41 | 0.73 | 6–2 |
| `spot_repeat_penalty=100` | tune | −$222.62 | 0.0192 | 2–6 |
| `risk_lambda=1.1` | tune | −$296.59 | 0.0004 | 0–8 |

Two settled questions: the bot's concentration on High Street is **not**
costing it anything (forcing diversification loses money), and the CVaR tail
penalty is **over**-tuned — it was making the bot under-stock.

## Safety rails

- `bench.py`, `ftlab.py` and `calibrate.py` can only reach
  `/api/v1/practice*`. A live order from a harness raises instead of sending.
- `bench.py` points each engine's `LOG_PATH` at `os.devnull` so a benchmark
  never pollutes `run_log.jsonl`.
- `.env*` is gitignored. Keys go in `.env.local`, never in a commit or a log.
- `dashboard.py` only ever issues `GET`.
- `test_no_leakage.py` asserts the decision path sees no future information.

## Working prompts for Claude Code

Paste these. They are also installed as slash commands in
[.claude/commands/](.claude/commands/), so `/bench`, `/ablate`, `/gate`,
`/diagnose-regression` and `/recalibrate` work out of the box.

**Before you trust a change**

```
Bench this change against baseline on the tune seeds, one knob at a time, then
show me the paired per-seed differences rather than the means. If the 95% CI
spans zero, say so plainly and tell me how many seeds I would need for 80%
power at the effect size you measured.
```

**Before you merge anything**

```
Run the holdout gate: bench this configuration against the current default on
hold-* only, paired, and report mean difference, sd, t, p, CI and the per-seed
table. Do not recommend merging unless it wins on hold-*. If the holdout gain
is much smaller than the train gain, tell me how much of it you think is
selection noise and why.
```

**When a metric improves but money does not**

```
Our <metric> improved but profit fell. Decompose it before theorising: find the
first day the two configurations diverge, diff the chosen action component by
component (spot, price, quality, meals), then play each one-component swap for
real in a fresh market per variant. Do not use logged sales from a different
order as a counterfactual. Tell me what share of the gap each component carries.
```

**When you suspect a cliff**

```
Sweep our price on a $0.25 grid at fixed inventory and play each point for
real on the same seed. Then compute every rival's value = QVALUE[quality] *
repfactor(rep) - price and check whether our chosen price lands on or near one
of theirs. beats() gives ties to the cheaper truck, so an exact tie loses every
contested customer instead of splitting them.
```

**Recalibrating**

```
Harvest fresh designed experiments, refit the priors with fit.py, and report
out-of-sample demand RMSE. Then — separately — bench the fitted priors for
money on tune-* before anyone pastes them into bot.py. Treat an RMSE
improvement as a hypothesis about profit, not evidence of it.
```

**Adding a tunable**

```
Add this as a key in bot.P with a default that reproduces today's behaviour
exactly, so it can be ablated with bench.py --ablate. Then prove the default
is a no-op by benching it against the current default and showing the per-seed
diffs are all zero.
```

### House rules for agents working here

1. One knob per bench run. Bundled changes are uninterpretable — the
   `abl_smoke` run moved two at once at n=1 and taught us nothing.
2. `n=8` cannot see a $150 effect. Quote the CI, not just the mean.
3. Never paste a `fit.py` block into `bot.py` without benching it for money.
4. Keep `train` / `validation` / `holdout` seeds apart.
5. Don't delete `baseline.py`.
6. Re-read `bot.py` before editing it. More than one agent works in this repo.

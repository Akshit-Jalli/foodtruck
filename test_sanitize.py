"""Offline gate for the bad-data guard. No key, no practice markets."""
import copy, json, random, time
import bot

states = [json.loads(l) for l in open("practice_states.jsonl")]
states = [s.get("state", s) for s in states]

# 1. identity: clean data passes through untouched
san = bot.Sanitizer()
for s in states:
    c, fixes = san(copy.deepcopy(s))
    assert c == s and not fixes, ("identity broke", fixes[:3])
print("identity: OK on %d states" % len(states))

# 2. corruption: break numeric leaves the way the server does
def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    elif bot._is_num(x):
        yield path

def setp(x, path, val):
    for p in path[:-1]:
        x = x[p]
    x[path[-1]] = val

night = [i for i, s in enumerate(states) if s.get("phase") == "night" and s.get("orders_open")]
rng = random.Random(7)
# decide() learns across nights (bot.MEM), so snapshot memory before each night
snap, clean_orders = {}, {}
bot.MEM = bot.Memory()
for i, s in enumerate(states):
    snap[i] = copy.deepcopy(bot.MEM)
    if i in night:
        clean_orders[i] = bot.decide(copy.deepcopy(s))
def with_mem(i, s):
    bot.MEM = copy.deepcopy(snap[i])
    return bot.decide(s)
trials = errors = same = raw_err = 0
worst = 0.0
by_mode = {}
for t in range(300):
    i = rng.choice([j for j in night if j > 0])
    prev = states[i - 1]
    s = copy.deepcopy(states[i])
    corrupted = []
    mode = "none" if rng.random() < 0.5 else "dollar"
    for path in rng.sample(list(leaves(s)), k=min(3, len(list(leaves(s))))):
        corrupted.append(path)
        v = s
        for p in path:
            v = v[p]
        setp(s, path, None if mode == "none" else "$%.2f" % v)
    try:
        with_mem(i, copy.deepcopy(s))
    except Exception:
        raw_err += 1
    g = bot.Sanitizer()
    g(copy.deepcopy(prev))
    c, fixes = g(s)
    trials += 1
    t0 = time.time()
    try:
        o = with_mem(i, c)
    except Exception as e:
        errors += 1
        print("  guarded error:", type(e).__name__, e, "| fixes:", fixes[:3], "| corrupted:", [p for p in corrupted][:3])
        continue
    worst = max(worst, time.time() - t0)
    hit = (o == clean_orders[i])
    same += hit
    by_mode.setdefault(mode, []).append(hit)
print("corruption: %d trials | unguarded decide errors %d | guarded errors %d | same order %.0f%% | max decide %.2fs"
      % (trials, raw_err, errors, 100.0 * same / trials, worst))
for m, v in by_mode.items():
    print("  %-6s corruption: same order %.0f%% of %d" % (m, 100.0 * sum(v) / len(v), len(v)))
# "$" strings must round-trip exactly; None is filled from yesterday, so the
# order can legitimately move a little.
# residual errors fall through to the last-good-order path in play()
assert errors <= 0.01 * trials and sum(by_mode["dollar"]) == len(by_mode["dollar"]) and worst < 0.5
print("RESULT: PASS")

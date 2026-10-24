"""Offline fault-injection test for bot.play(): no key, no server.

    python3 test_transport.py                 # hardened bot.py: must PASS
    python3 test_transport.py bot_frozen.py   # the 6244d30 copy, for contrast

A fake server runs 2 markets x 4 nights with final-style 2-second semantics:
a long-poll on an unchanged version sleeps until the window closes, and the
night is then lost. It injects the faults seen in tonight's logs - connection
resets, read timeouts, a 502, a truncated body, bad JSON - plus a state that
makes decide() itself raise. PASS = no crash and every night gets an order.
"""
import http.client
import importlib.util
import json
import os
import sys
import types
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))


class Done(BaseException):
    pass


def load(path):
    spec = importlib.util.spec_from_file_location("engine_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LOG_PATH = os.devnull
    mod.time = types.SimpleNamespace(time=__import__("time").time, sleep=lambda s: None)
    return mod


def sample_day():
    with open(os.path.join(HERE, "days.jsonl"), encoding="utf-8") as f:
        return json.loads(f.readlines()[1])


class FakeServer:
    MARKETS, DAYS = 2, 4

    def __init__(self, me, row):
        self.me, self.row = me, row
        self.market, self.day, self.version = 1, 1, 1
        self.orders, self.missed, self.gets, self.posts = {}, [], 0, {}
        # (kind, market, day, attempt) -> fault
        self.faults = {
            ("GET", 1, 1, 2): http.client.IncompleteRead(b"{"),
            ("GET", 1, 2, 1): ValueError("Expecting value: line 1 column 1"),
            ("POST", 1, 2, 1): urllib.error.URLError(ConnectionResetError(54, "reset by peer")),
            ("POST", 1, 3, 1): 502,
            ("POST", 1, 4, 1): TimeoutError("The read operation timed out"),
            ("POST", 2, 1, 1): urllib.error.URLError("_ssl.c:1015: The handshake operation timed out"),
        }
        self.get_n = {}

    def state(self):
        r = self.row
        st = {
            "version": self.version, "mode": "final", "market": self.market,
            "phase": "night", "orders_open": True, "order_for_day": self.day,
            "day": self.day, "days": 30,
            "rules": {"budget": 500, "price_min": 1, "price_max": 30,
                      "spots": ["oval", "thompson", "union", "stadium", "highst", "northres"]},
            "you": {"id": self.me, "cash": 0.0, "reputation": dict(r["your_rep"])},
            "trucks": r["trucks"],
            "forecast": r["forecast_before"],
            "last_day": None if self.day == 1 else {
                "day": self.day - 1, "spots": r["spots"], "you": r["you"]},
        }
        if self.market == 2 and self.day == 2:
            del st["rules"]                 # decide() raises KeyError on this one
        return st

    def advance(self, ordered):
        if not ordered:
            self.missed.append((self.market, self.day))
        self.day += 1
        if self.day > self.DAYS:
            self.market, self.day = self.market + 1, 1
        if self.market > self.MARKETS:
            raise Done()
        self.version += 1

    def call(self, method, path, body=None, timeout=35):
        if method == "GET":
            k = (self.market, self.day)
            self.get_n[k] = self.get_n.get(k, 0) + 1
            f = self.faults.get(("GET",) + k + (self.get_n[k],))
            if isinstance(f, BaseException):
                raise f
            since = int(path.rsplit("=", 1)[1])
            if since == self.version:       # long-poll: the window closes on us
                self.advance(ordered=False)
            return 200, self.state()
        k = (self.market, self.day)
        self.posts[k] = self.posts.get(k, 0) + 1
        f = self.faults.get(("POST",) + k + (self.posts[k],))
        if isinstance(f, BaseException):
            raise f
        if isinstance(f, int):
            return f, {"error": "HTTP Error %d: Bad Gateway" % f}
        self.orders[k] = body
        self.advance(ordered=True)
        return 200, {"ok": True}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bot.py")
    mod = load(path)
    row = sample_day()
    srv = FakeServer(row["you"].get("id") or "t_e2de7d", row)
    # days.jsonl stores our id on the spot entry, not on `you`
    for info in row["spots"].values():
        for t in info["trucks"]:
            if t.get("truck_name") == "IBE":
                srv.me = t["id"]
    mod.call = srv.call
    crashed = None
    try:
        mod.play("/api/v1", quiet=True)
    except Done:
        pass
    except BaseException as e:  # noqa: BLE001
        crashed = "%s: %s" % (type(e).__name__, e)
    total = srv.MARKETS * srv.DAYS
    print("engine:   %s" % os.path.basename(path))
    print("orders:   %d/%d nights" % (len(srv.orders), total))
    print("missed:   %s" % (srv.missed or "none"))
    print("crashed:  %s" % (crashed or "no"))
    ok = crashed is None and not srv.missed and len(srv.orders) == total
    print("RESULT:   %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

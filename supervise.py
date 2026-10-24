"""Keep the bot running all night.

    python supervise.py            # runs bot.py, restarts it if it dies

The live game is 48 days and the final is 20 markets at 8:30 AM; a crash at
03:00 with nobody watching would cost every day until morning. So: restart on
any unexpected exit, back off if it is failing repeatedly, and do NOT restart
on an authentication failure, which only a human can fix.
"""
import subprocess
import sys
import time
from datetime import datetime

CMD = [sys.executable, "-u", "bot.py"]
AUTH_EXIT = 1          # bot.py exits 1 on 401/404
MAX_BACKOFF = 60


def stamp():
    return datetime.now().strftime("%H:%M:%S")


def main():
    backoff = 2
    starts = 0
    while True:
        starts += 1
        print("[%s] supervisor: start #%d" % (stamp(), starts), flush=True)
        began = time.time()
        try:
            rc = subprocess.call(CMD)
        except KeyboardInterrupt:
            print("[%s] supervisor: stopped by hand" % stamp(), flush=True)
            return 0
        ran = time.time() - began

        if rc == 0:
            print("[%s] supervisor: bot exited cleanly after %.0fs" % (stamp(), ran), flush=True)
            return 0
        print("[%s] supervisor: bot exited %s after %.0fs" % (stamp(), rc, ran), flush=True)

        # A long run that then died is a blip; restart promptly. A run that
        # dies immediately is a real fault, so slow down instead of spinning.
        if ran > 120:
            backoff = 2
        else:
            backoff = min(backoff * 2, MAX_BACKOFF)
            if rc == AUTH_EXIT and ran < 15:
                print("[%s] supervisor: looks like a key or market problem, "
                      "not restarting. Fix NIGHTSHIFT_KEY and start again."
                      % stamp(), flush=True)
                return rc
        print("[%s] supervisor: restarting in %ds" % (stamp(), backoff), flush=True)
        time.sleep(backoff)


if __name__ == "__main__":
    sys.exit(main())

"""Test job for the queue's stop rules: just sleeps (no data, no model)."""
import sys
import time

secs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
for i in range(secs):
    time.sleep(1)
    if i % 30 == 0:
        print(f"sleeping {i}/{secs}", flush=True)

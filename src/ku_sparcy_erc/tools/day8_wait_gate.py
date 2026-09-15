#!/usr/bin/env python3
"""Wait for a single physical checkpoint; a failure never advances motion."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('gate',choices=['pre_place_ready','pre_place','support','release','deposit']);p.add_argument('--wall-timeout',type=float,default=900.);a=p.parse_args()
end=time.monotonic()+a.wall_timeout
wanted={'pre_place_ready':'WAIT_APPROVAL_PRE_PLACE','pre_place':'WAIT_APPROVAL_LOWER',
        'support':'WAIT_APPROVAL_RELEASE','release':'WAIT_APPROVAL_RETRACT','deposit':'DONE'}[a.gate]
while time.monotonic()<end:
    try:d=json.loads(Path(a.result).read_text())
    except (OSError,ValueError):time.sleep(.5);continue
    if d.get('state')=='FAILED':
        print('[DAY8 '+a.gate+'][FAIL] '+d.get('reason',''));raise SystemExit(1)
    if d.get('state')==wanted:
        print('[DAY8 '+a.gate+'][PASS] '+d.get('reason',''));raise SystemExit(0)
    time.sleep(.5)
print('[DAY8 '+a.gate+'][FAIL] wall watchdog; inspect live node and /clock')
raise SystemExit(1)

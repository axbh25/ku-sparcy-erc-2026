#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
import subprocess
BASE='cb284b8f7188ece2d15f0b5461b02ffb686bd19b'
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
if git('rev-parse','day7-stable^{commit}')!=BASE:raise SystemExit('[DAY7 CHECKPOINT][FAIL]')
original=set(git('ls-tree','-r','--name-only',BASE).splitlines())
changed=set(git('diff','--name-only',BASE,'--').splitlines())
allowed={'.gitignore','src/ku_sparcy_erc/setup.py','src/ku_sparcy_erc/launch/solution.launch.py'}
bad=sorted((changed & original)-allowed)
if bad:
    print('\n'.join(bad));raise SystemExit('[DAY7 FROZEN BASELINE][FAIL]')
print('[DAY7 CHECKPOINT][PASS]')
print('[DAY5-7 TRANSPORT FROZEN][PASS]')
print('[DAY7 FROZEN BASELINE][PASS]')
PY
bash ./scripts/check_day567_frozen_files.sh

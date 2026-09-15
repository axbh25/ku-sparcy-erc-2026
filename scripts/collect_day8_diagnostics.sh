#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="/tmp/ku_sparcy_day8_diagnostics_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  git status --short
  git log --oneline --decorate -6
  git diff --stat
  docker inspect --format '{{.State.Status}}' erc_sim || true
  if [ "$(docker inspect --format '{{.State.Running}}' erc_sim 2>/dev/null || true)" = true ]; then
    docker exec -e PYTHONDONTWRITEBYTECODE=1 erc_sim bash -lc '
      source /opt/ros/humble/setup.bash
      source /opt/erc_ws/install/setup.bash
      pgrep -af "[g]z sim.*erc_world" || true
      find /opt/erc_ws/src/ku_sparcy_erc/day8_results -name "*.json" -type f -print
      python3 - <<"PY"
import json
from pathlib import Path
files=sorted(Path("/opt/erc_ws/src/ku_sparcy_erc/day8_results").rglob("*.json"),key=lambda p:p.stat().st_mtime)[-8:]
keys=("passed","state","reason","capture_diagnostics","bin_geometry","plan", "contact_evidence", "release_trace", "state_history")
for p in files:
    print("\nFILE",p)
    try:
        d=json.loads(p.read_text())
        for k in keys:
            if k=="plan" and d.get(k):
                print(k, {x:d[k].get(x) for x in ("passed","pre_place_xyz_m","lower_place_xyz_m","retract_xyz_m","rejected_placement_candidates")})
            elif k in d:print(k,json.dumps(d[k],sort_keys=True))
    except Exception as e:print(e)
PY
    ' || true
  fi
} > "$OUT" 2>&1
printf '[DAY8 DIAGNOSTICS][PASS] %s\n' "$OUT"

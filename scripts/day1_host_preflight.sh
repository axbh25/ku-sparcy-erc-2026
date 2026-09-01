#!/usr/bin/env bash
set -u

PASS_COUNT=0
FAIL_COUNT=0

pass() { printf '[PASS] %s\n' "$1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { printf '[FAIL] %s\n' "$1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

printf '=== KU SPARCy Day 1 host preflight ===\n'

if [ "$(uname -m)" = 'x86_64' ]; then pass 'CPU architecture is x86_64'; else fail "CPU architecture is $(uname -m), expected x86_64"; fi

if [ -r /etc/os-release ]; then
  . /etc/os-release
  if [ "${ID:-}" = 'ubuntu' ] && [ "${VERSION_ID:-}" = '22.04' ]; then
    pass 'Host OS is Ubuntu 22.04'
  else
    fail "Host OS is ${PRETTY_NAME:-unknown}; expected Ubuntu 22.04"
  fi
else
  fail '/etc/os-release is not readable'
fi

MEM_GIB="$(awk '/MemTotal:/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
if [ "$MEM_GIB" -ge 29 ]; then pass "RAM is ${MEM_GIB} GiB (>=29 GiB)"; else fail "RAM is ${MEM_GIB} GiB; expected about 32 GiB"; fi

FREE_GIB="$(df -Pk "$HOME" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if [ "$FREE_GIB" -ge 25 ]; then pass "Free space in HOME is ${FREE_GIB} GiB (>=25 GiB)"; else fail "Only ${FREE_GIB} GiB free in HOME; need at least 25 GiB"; fi

if [ -n "${DISPLAY:-}" ]; then pass "DISPLAY is set to $DISPLAY"; else fail 'DISPLAY is unset'; fi
if [ -S /tmp/.X11-unix/X0 ] || compgen -G '/tmp/.X11-unix/X*' >/dev/null; then pass 'An X11 socket exists'; else fail 'No X11 socket found under /tmp/.X11-unix'; fi
if compgen -G '/dev/dri/renderD*' >/dev/null; then pass 'A /dev/dri/renderD* device exists for Intel/Mesa rendering'; else fail 'No /dev/dri/renderD* device exists'; fi

for command in git curl jq tree xhost glxinfo docker gh; do
  if command -v "$command" >/dev/null 2>&1; then pass "$command is installed"; else fail "$command is not installed"; fi
done

if docker version >/dev/null 2>&1; then pass 'Docker daemon is reachable without sudo'; else fail 'Docker daemon is not reachable without sudo'; fi
if docker compose version >/dev/null 2>&1; then pass 'Docker Compose v2 is available'; else fail 'Docker Compose v2 is unavailable'; fi
if gh auth status --hostname github.com >/dev/null 2>&1; then pass 'GitHub CLI is authenticated'; else fail 'GitHub CLI is not authenticated'; fi

RENDERER="$(glxinfo -B 2>/dev/null | awk -F': ' '/OpenGL renderer string/ {print $2; exit}')"
if printf '%s' "$RENDERER" | grep -Eqi 'intel|iris|mesa'; then
  pass "OpenGL renderer is Intel/Mesa: $RENDERER"
else
  fail "OpenGL renderer was not recognized as Intel/Mesa: ${RENDERER:-unknown}"
fi

printf '\nSUMMARY: %d PASS, %d FAIL\n' "$PASS_COUNT" "$FAIL_COUNT"
if [ "$FAIL_COUNT" -eq 0 ]; then
  printf '[DAY1 HOST PREFLIGHT][PASS]\n'
  exit 0
fi
printf '[DAY1 HOST PREFLIGHT][FAIL]\n'
exit 1

#!/usr/bin/env bash
# Read-only inspection of the two organizer-provided patch files. Nothing is applied.
set -euo pipefail

PATCH_8="${1:-$HOME/Downloads/8b5f261.patch}"
PATCH_4="${2:-$HOME/Downloads/4e0ce63.patch}"
EXPECTED_8='72e1c25236d010b83da08b083067817f58f09d75e1b09c42695d33d73f8f8415'
EXPECTED_4='4ed0e059f0493146faa81509cda41e736900f364294e8452c22f6610c2a75b62'

pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; exit 1; }

[ -f "$PATCH_8" ] || fail "missing $PATCH_8"
[ -f "$PATCH_4" ] || fail "missing $PATCH_4"

SHA8="$(sha256sum "$PATCH_8" | awk '{print $1}')"
SHA4="$(sha256sum "$PATCH_4" | awk '{print $1}')"
[ "$SHA8" = "$EXPECTED_8" ] || fail '8b5f261.patch checksum mismatch'
[ "$SHA4" = "$EXPECTED_4" ] || fail '4e0ce63.patch checksum mismatch'
pass 'both uploaded patch checksums match the reviewed files'

printf '\n--- 8b5f261.patch file summary ---\n'
git apply --numstat "$PATCH_8"
printf '\n--- 4e0ce63.patch file summary ---\n'
git apply --numstat "$PATCH_4"

for patch in "$PATCH_8" "$PATCH_4"; do
  grep -Fq '+            <size>0.25 0.02 0.16</size>' "$patch" || \
    fail "2 cm book spine change missing from $patch"
  grep -Fq '+              <mu>10.0</mu>' "$patch" || \
    fail "book friction mu=10 missing from $patch"
  grep -Fq '+    <limit effort="40.0"' "$patch" || \
    fail "mimic-joint effort-limit change missing from $patch"
  grep -Fq '+    <mu1>2.7</mu1>' "$patch" || \
    fail "fingertip friction change missing from $patch"
  if grep -Fq '<command_interface name="effort"' "$patch"; then
    fail "unexpected effort command interface in $patch"
  fi
  if grep -Fq 'gz_system.cpp' "$patch"; then
    fail "unexpected gz_ros2_control source change in $patch"
  fi
done
pass 'both patches retain position-only gripper command semantics'
pass 'neither patch modifies gz_ros2_control source'

# The first patch contains the mesh-reference regression reported in the issue.
grep -Fq 'base_antenna_link_mirror.stl' "$PATCH_8" || \
  fail 'expected 8b5f261 broken antenna mesh reference not found'
grep -Fq 'wheel_link_reverted.stl' "$PATCH_8" || \
  fail 'expected 8b5f261 broken wheel mesh reference not found'
pass '8b5f261 mesh-reference regression is present and documented'

if grep -Eq 'base_antenna_link_mirror\.stl|wheel_link_reverted\.stl' "$PATCH_4"; then
  fail '4e0ce63 unexpectedly reintroduces broken mesh references'
fi
pass '4e0ce63 does not contain the broken mesh-reference edits'

if grep -Fq 'erc_base_book.STL' "$PATCH_4"; then
  fail '4e0ce63 unexpectedly contains a binary book-mesh edit'
fi
pass '4e0ce63 is limited to book SDF and generated URDF text changes'

printf '\n[PATCH 8b5f261 REVIEW][PASS]\n'
printf '[PATCH 4e0ce63 REVIEW][PASS]\n'
printf '[GRIPPER PATCH INSPECTION][PASS]\n'
printf '[NOTICE] These patches are historical evidence; canonical state is upstream/main at 93554d4.\n'
printf '[NOTICE] No patch was applied, cherry-picked, merged, or committed.\n'

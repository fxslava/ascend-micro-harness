#!/usr/bin/env bash
# Behaviour gate for scripts/lib/work_root.sh.
#
# It is eight assertions over a path-returning function, which is not much -- but the
# function decides where a 2500-second BiSheng build and an hours-long CAModel ladder
# land, and the bug it exists to prevent already happened once: a /tmp root, a WSL
# restart, and a finished build plus nine passed ladder rungs gone with no trace that
# anything had ever run. The two cases worth having a test for are the two silent ones:
# falling back to /tmp (correct, but it has to SAY so) and refusing a $HOME that is
# really the Windows drive (where CANN's build and the simulator do not belong).
#
#   bash scripts/test_work_root.sh
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
status=0

check() { # check <label> <expected> <actual>
    printf '%-48s ' "$1"
    if [[ "$2" == "$3" ]]; then echo "OK ($3)"; else echo "FAIL want=$2 got=$3"; status=1; fi
}

. scripts/lib/work_root.sh

out=$(ASCEND_MICRO_WORK_ROOT=/var/tmp/explicit ascend_micro_work_root)
check "ASCEND_MICRO_WORK_ROOT wins" "/var/tmp/explicit" "$out"

out=$(ascend_micro_work_root)
check "the default is persistent, under \$HOME" "${XDG_CACHE_HOME:-$HOME/.cache}/ascend-micro" "$out"

out=$(XDG_CACHE_HOME="$HOME/xdg-selftest" ascend_micro_work_root)
check "XDG_CACHE_HOME is honoured" "$HOME/xdg-selftest/ascend-micro" "$out"

# No usable $HOME is the container case: /tmp is right, but it must be announced,
# because a caller that does not know its tree is volatile cannot plan around it.
out=$(env -u HOME -u XDG_CACHE_HOME -u ASCEND_MICRO_WORK_ROOT bash -c \
        '. scripts/lib/work_root.sh; ascend_micro_work_root' 2>/dev/null)
check "no \$HOME falls back to /tmp" "/tmp" "$out"
warning=$(env -u HOME -u XDG_CACHE_HOME -u ASCEND_MICRO_WORK_ROOT bash -c \
        '. scripts/lib/work_root.sh; ascend_micro_work_root' 2>&1 >/dev/null | head -1)
printf '%-48s ' "and says so on stderr"
if [[ "$warning" == *"restart"* ]]; then echo "OK"; else echo "FAIL: $warning"; status=1; fi

# A $HOME on the Windows drive is the trap this repository already works around by
# building outside the checkout; resolving the work root onto it would undo that.
out=$(HOME=/mnt/c/Users ascend_micro_work_root 2>/dev/null)
check "a \$HOME on the Windows drive is refused" "/tmp" "$out"

first=$(ascend_micro_work_dir work-root-selftest)
again=$(ascend_micro_work_dir work-root-selftest)
check "a named dir is stable across calls" "$first" "$again"
printf '%-48s ' "the named dir is created"
if [[ -d "$first" ]]; then echo "OK ($first)"; else echo FAIL; status=1; fi
unique=$(ascend_micro_work_dir work-root-selftest --unique)
printf '%-48s ' "--unique gives a fresh dir"
if [[ -d "$unique" && "$unique" != "$first" ]]; then echo OK; else echo "FAIL: $unique"; status=1; fi
rmdir "$unique" "$first" 2>/dev/null || true

echo
if ((status)); then echo "FAIL scripts/lib/work_root.sh"; else echo "PASS work root: override, persistent default, XDG, announced /tmp fallback, DrvFs refusal, stable and unique dirs"; fi
exit "$status"

#!/bin/bash
# All members of the current phase can proceed. Admission waits for the opposite phase to drain.
set -euo pipefail
phase=$1
shift
root=$HOME/ascend-sync-schedule-ablation/queue
mkdir -p "$root"
exec {admission}>"$root/phase-admission.lock"
exec {builds}>"$root/phase-compile.lock"
exec {runs}>"$root/phase-camodel.lock"
case "$phase" in
    compile)
        flock -x "$admission"
        flock -x "$runs"
        flock -s "$builds"
        flock -u "$runs"
        flock -u "$admission"
        if pgrep -f '(^|/)test_fused_decode_e2e_sim --(reproduce|wave-v11-tiles)([[:space:]]|$)' >/dev/null; then
            printf 'CAModel is already running outside the phase guard.\n' >&2
            exit 2
        fi
        ;;
    camodel)
        # Also wait for any compiler still holding the previous guard's lock.
        exec {previous}>"$root/bisheng-vs-camodel.lock"
        flock -s "$previous"
        flock -x "$admission"
        flock -x "$builds"
        flock -s "$runs"
        flock -u "$builds"
        flock -u "$admission"
        if pgrep -x bisheng >/dev/null; then
            printf 'BiSheng is already running outside the phase guard.\n' >&2
            exit 2
        fi
        ;;
    *) exit 2 ;;
esac
"$@"


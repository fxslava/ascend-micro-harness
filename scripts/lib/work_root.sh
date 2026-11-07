# Where the long-running build and evidence trees live. Source this, then call
# `ascend_micro_work_dir <name>`.
#
# WHY THIS IS NOT /tmp
#
# These trees are expensive: a clean CAModel build of the fused decode operator is about
# 2500 s of BiSheng, and a full canary ladder is hours of simulator time on top of it.
# They were rooted in /tmp because the repository is checked out on /mnt/c, a DrvFs
# mount, and CANN's build and the simulator both want a real local filesystem -- that
# part was right, and it is why these scripts do not simply build in the checkout.
#
# But /tmp is not merely local, it is VOLATILE. systemd-tmpfiles clears it, and in WSL a
# `wsl --shutdown` or a distro restart takes it with no warning: one restart during this
# work discarded a finished 41-minute build and nine passed ladder rungs at once, and
# nothing in the scripts could tell the difference between that and a run that had never
# happened. A work root has to be on ext4 AND survive a restart, and $HOME in WSL is
# both.
#
# Resolution order:
#   1. $ASCEND_MICRO_WORK_ROOT, if set -- the escape hatch, and what CI should set.
#   2. $XDG_CACHE_HOME/ascend-micro, or $HOME/.cache/ascend-micro, when $HOME is on a
#      local filesystem (not a DrvFs/9p mount of the Windows drive, which would be slow
#      and, for the simulator, wrong).
#   3. /tmp, the old behaviour, with a warning on stderr saying what it costs. This is
#      the path a container with no usable $HOME still takes.
#
# Each caller passes a NAME rather than a full path, so every tree lands under one root
# that can be inspected, archived or deleted in one place.

ascend_micro_work_root() {
    if [[ -n "${ASCEND_MICRO_WORK_ROOT:-}" ]]; then
        printf '%s\n' "$ASCEND_MICRO_WORK_ROOT"
        return 0
    fi
    local base="${XDG_CACHE_HOME:-${HOME:-}/.cache}"
    if [[ -n "${HOME:-}" && -d "$HOME" && -w "$HOME" ]]; then
        # Refuse a $HOME that is really the Windows drive: DrvFs cannot host a CANN
        # build and the simulator misbehaves on it, which is the whole reason these
        # scripts leave the checkout in the first place.
        local fstype
        fstype="$(stat -f -c %T "$HOME" 2>/dev/null || echo unknown)"
        case "$fstype" in
            9p|drvfs|v9fs|cifs|smb2|fuseblk)
                printf 'ascend-micro: $HOME is on %s, not a local filesystem; falling back to /tmp.\n' \
                    "$fstype" >&2
                printf 'ascend-micro: set ASCEND_MICRO_WORK_ROOT to a local, persistent directory to keep builds across restarts.\n' >&2
                printf '%s\n' /tmp
                return 0
                ;;
        esac
        printf '%s/ascend-micro\n' "$base"
        return 0
    fi
    printf 'ascend-micro: no usable $HOME; rooting work trees in /tmp, which a WSL restart or systemd-tmpfiles will delete.\n' >&2
    printf 'ascend-micro: set ASCEND_MICRO_WORK_ROOT to keep an expensive build and its evidence across a restart.\n' >&2
    printf '%s\n' /tmp
}

# ascend_micro_work_dir <name> [--unique]
# Prints a directory path and creates it. Without --unique the same name resolves to the
# same directory every time, which is what makes a rerun a resume. With --unique a fresh
# mktemp directory is made under the root, for runs that must not share state.
ascend_micro_work_dir() {
    local name="${1:?ascend_micro_work_dir needs a name}"
    local unique="${2:-}"
    local root
    root="$(ascend_micro_work_root)"
    mkdir -p "$root"
    if [[ "$unique" == --unique ]]; then
        mktemp -d "$root/${name}.XXXXXX"
    else
        local dir="$root/$name"
        mkdir -p "$dir"
        printf '%s\n' "$dir"
    fi
}

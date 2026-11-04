#!/usr/bin/env bash
# Container plumbing for the Cube GEMM Split-K reduction probe.
#
# The CANN 9.2.0-beta.2 toolkit and the Ascend950PR_9599 simulator live in the
# `ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12` image on this host, not in
# the WSL instance -- the WSL instance's own CANN is 8.5.0 and knows neither
# dav-3510 nor Ascend950PR_9599. So every device gate runs in the container, and
# the host gates run wherever a g++ and a python3 are.
#
#   bash scripts/gemm_reduce_container.sh sync     # refresh the ext4 work copy
#   bash scripts/gemm_reduce_container.sh up       # (re)start the container
#   bash scripts/gemm_reduce_container.sh exec ... # run a command inside it
#   bash scripts/gemm_reduce_container.sh pull DIR # copy evidence back out
#
# Why a work copy rather than a bind mount of the Windows checkout: the drvfs
# page cache can serve BiSheng a stale copy of a source a Windows-side editor has
# just rewritten, and a 9p round trip per include is slow enough to dominate the
# build. The copy lives on the WSL instance's ext4 filesystem and is mounted into
# the container at /work.
set -euo pipefail

image="${GEMM_IMAGE:-ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12}"
name="${GEMM_CONTAINER:-gemm_dev}"
work="${GEMM_WORK:-$HOME/gemm-work}"
checkout="${GEMM_CHECKOUT:-/mnt/d/Projects/ascend-micro-harness}"

case "${1:-}" in
sync)
    mkdir -p "$work/src"
    rsync -a --delete --exclude '.git' --exclude 'reports' --exclude 'build' \
        --exclude 'build-*' --exclude 'bin' --exclude '__pycache__' \
        "$checkout/" "$work/src/"
    printf 'synced %s files into %s/src\n' "$(find "$work/src" -type f | wc -l)" "$work"
    ;;
up)
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" -v "$work:/work" -w /work/src "$image" \
        sleep infinity >/dev/null
    printf 'container %s up on %s\n' "$name" "$image"
    ;;
ensure)
    # Idempotent: the daemon in this WSL instance stops containers across restarts, and a long
    # ladder is not worth losing to that.
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
        docker start "$name" >/dev/null 2>&1 ||
            docker run -d --name "$name" -v "$work:/work" -w /work/src "$image" \
                sleep infinity >/dev/null
    fi
    printf 'container %s running\n' "$name"
    ;;
exec)
    # Note: this joins its arguments, so quoting inside the command does not
    # survive. Use `run` for anything with quotes or shell metacharacters.
    shift
    docker exec "$name" bash -lc "$*"
    ;;
run)
    # Run a script from the work copy inside the container, arguments intact.
    # This is the form to use for anything the layered shells would re-split.
    # Self-healing: this WSL instance's daemon stops containers between invocations often
    # enough that a long ladder is not worth losing to it.
    shift
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
        docker start "$name" >/dev/null 2>&1 ||
            docker run -d --name "$name" -v "$work:/work" -w /work/src "$image" \
                sleep infinity >/dev/null
    fi
    script="$1"
    shift
    docker exec "$name" bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true; exec bash "$@"' \
        bash "/work/src/$script" "$@"
    ;;
pull)
    # Copy an evidence directory out of the work copy into the checkout's reports/.
    shift
    src="$work/$1"
    dst="$checkout/reports/$(basename "$1")"
    mkdir -p "$dst"
    cp -r "$src/." "$dst/"
    printf 'pulled %s -> %s\n' "$src" "$dst"
    ;;
*)
    printf 'usage: %s {sync|up|exec CMD...|pull DIR}\n' "$0" >&2
    exit 2
    ;;
esac

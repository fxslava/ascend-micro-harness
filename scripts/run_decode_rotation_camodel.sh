#!/usr/bin/env bash
# CAModel gates for the ingress/egress rotation controls and the 64-block AIV
# wave. Runs INSIDE ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12, on the
# Ascend950PR_9599 simulator. There is no 950PR on this host, so these are the
# gates that actually EXECUTE; the physical-device legs of the same tests are
# compile-only here and the report says so.
#
#   bash scripts/run_decode_rotation_camodel.sh            # everything below
#   bash scripts/run_decode_rotation_camodel.sh boundary   # section 3 and 4 only
#   bash scripts/run_decode_rotation_camodel.sh grid       # the 64-block waves
#   bash scripts/run_decode_rotation_camodel.sh cube       # the three Cube legs
#   bash scripts/run_decode_rotation_camodel.sh aic        # the prerotated AIC leg
#   bash scripts/run_decode_rotation_camodel.sh census     # the section 5 census
#
# `census` reads the traces the `boundary` and `cube` phases leave, so run
# those first (or run with no argument, which runs everything in order).
#
# What each phase is:
#
#   boundary  REFINEMENT_OF_COMBAT_KERNELS.md section 3 and 4 on BOTH egress
#             legs of the Vector Split-K operator: isolated logical-page
#             head/middle/tail canaries at one tile of the real 256x16
#             geometry, then depth = 2 and wrap = 4. Every GM payload is
#             flooded with 0x7FC0BEEF first, the context is checked BYTE-exact
#             against the leg's own oracle line, and every unassigned byte of
#             every buffer plus both exterior guards must come back canary.
#             Each run is audited with its own traces.
#
#   grid      The raised wave bound, on the launches it exists for: K = 16 at
#             H = 4 B = 1 and K = 8 at H = 4 B = 2, both 64-block grids, both
#             egress legs. A barrier that did not span all 64 AIV subcores
#             hangs here rather than returning a wrong answer, so the gate is
#             that the launch RETURNS and that every emitted context matches
#             its FP64 oracle. The 32-block rungs run too, as the control.
#
#   cube      The three rotation specializations of the MIX_AIC_1_2 operator,
#             each in the single-tile isolation the canary protocol asks for
#             and each against its own oracle. `prerotated` and
#             `rotated-basis` run with `pi_table` -- and, on the second, the
#             three Gather tables and the sign diagonal -- flooded with
#             0x7FC0BEEF, so a prologue or a butterfly that still consumed
#             them could not produce a bit-exact score panel.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, ROT_BUILD_DIR, ROT_WORK_DIR,
# ROT_TIMEOUT.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
build_dir="${ROT_BUILD_DIR:-/work/tq-sim}"
run_timeout="${ROT_TIMEOUT:-1800s}"
phase="${1:-all}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN environment not found: ${toolkit}/set_env.sh" >&2
    exit 2
fi
if [[ ! -x "${build_dir}/lib/test_camodel_decode_splitk_boundary" ]]; then
    echo "build the simulator targets first:" >&2
    echo "  ROT_BUILD_DIR=${build_dir} bash scripts/run_decode_rotation_container.sh sim" >&2
    exit 2
fi
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
export LD_LIBRARY_PATH="${build_dir}/lib:${toolkit}/tools/simulator/${soc}/lib:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"

work_dir="${ROT_WORK_DIR:-$(mktemp -d /tmp/ascend-decode-rotation.XXXXXX)}"
mkdir -p "${work_dir}"
echo "artifact directory: ${work_dir}"
: > "${work_dir}/summary.txt"

# Every case appends one PASS line, and the tally at the end must match. A run
# that stops in the middle -- which is what a `set -e` swallowed inside a
# subshell looks like from outside -- then fails loudly instead of printing a
# pass over a tenth of the work.
expected_cases=0

# A run is a failure if the model reported an error, if any exception dump is
# nonempty, or if the probe itself did not print PASS. A clean exit status is
# not the gate: the model prints its diagnostics and carries on.
screen_run() {
    local dir="$1" log="$2"
    if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' "${log}"; then
        echo "model error in ${log}" >&2
        grep -Eim5 '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' "${log}" >&2
        return 1
    fi
    shopt -s nullglob
    local dump
    for dump in "${dir}"/core*.excp_log.dump "${dir}"/excp_log*.dump; do
        if [[ -s "${dump}" ]]; then
            echo "nonempty exception dump: ${dump}" >&2
            return 1
        fi
    done
    return 0
}

boundary_case() {
    local basis="$1" tag="$2" page="$3" strips="$4" strip_begin="$5"
    local dir="${work_dir}/boundary_${basis}_${tag}"
    mkdir -p "${dir}"
    local flag=()
    if [[ "${basis}" == rotated ]]; then flag=(--rotated); fi
    # `if ! ( ... )` rather than a bare subshell: `set -e` does not reliably
    # carry a failing subshell out of a function called at top level, and a
    # case that failed inside one came back as a clean exit once already.
    if ! (
        cd "${dir}"
        if ! timeout "${run_timeout}" \
            "${build_dir}/lib/test_camodel_decode_splitk_boundary" \
            --head 2 --rows 1 --page-begin "${page}" --strips "${strips}" \
            --strip-begin "${strip_begin}" "${flag[@]}" > run.log 2>&1; then
            tail -40 run.log >&2
            exit 1
        fi
        screen_run . run.log
        grep -q '^PASS ' run.log || { echo "probe did not pass: ${dir}" >&2; exit 1; }
        # The ring census, keyed by operand route and id, over this run's own
        # traces -- the section 5 audit, not a re-read of the probe's verdict.
        local audit=()
        [[ "${tag}" == wrap ]] && audit+=(--require-overlap)
        python3 "${repo_dir}/scripts/analyze_decode_ring_trace.py" . \
            --strips "${strips}" "${audit[@]}" > audit.json
        # NOT scripts/analyze_splitk_decode_trace.py. That auditor keys off the
        # three atomic instruction PCs of the test-and-set protocol this
        # operator used to have, and the workspace-partition rewrite removed
        # every atomic from it, so the auditor now raises "no completed lock
        # intervals" on a perfectly good run. scripts/run_decode_pipeline_camodel.sh
        # still calls it on its splitk legs and therefore cannot complete on the
        # current kernel; that is an open finding recorded in the report and not
        # something this change introduced.
        #
        # What replaces it here is the claim the rewrite actually makes: there
        # is no critical section left at all. Zero SIMT_ATOM instructions on
        # every core of the run, read out of the traces rather than asserted.
        # `|| true` because no match is the PASSING case here, and under
        # `pipefail` a grep that matches nothing fails the pipeline.
        local atomics
        atomics=$({ grep -ho 'SIMT_ATOM' core*.veccore*.instr_log.dump 2>/dev/null \
            || true; } | wc -l)
        if [[ "${atomics}" != 0 ]]; then
            echo "${atomics} atomic instructions in a lock-free operator: ${dir}" >&2
            exit 1
        fi
        echo "{\"atomic_instructions\": ${atomics}}" > protocol.json
    ); then
        echo "FAILED case: ${dir}" >&2
        exit 1
    fi
    grep -h '^PASS ' "${dir}/run.log" | tee -a "${work_dir}/summary.txt"
}

grid_case() {
    local tag="$1" rows="$2" splitk="$3" basis="$4"
    local dir="${work_dir}/grid_${tag}"
    mkdir -p "${dir}"
    # `if ! ( ... )` rather than a bare subshell: `set -e` does not reliably
    # carry a failing subshell out of a function called at top level, and a
    # case that failed inside one came back as a clean exit once already.
    if ! (
        cd "${dir}"
        if ! timeout "${run_timeout}" "${build_dir}/lib/test_decode_splitk_aiv_sim" \
            --rows "${rows}" --splitk "${splitk}" --basis "${basis}" > run.log 2>&1; then
            tail -60 run.log >&2
            exit 1
        fi
        screen_run . run.log
        grep -q '^PASS ' run.log || { echo "ladder did not pass: ${dir}" >&2; exit 1; }
    ); then
        echo "FAILED case: ${dir}" >&2
        exit 1
    fi
    grep -hE '^  ok|^PASS ' "${dir}/run.log" | tee -a "${work_dir}/summary.txt"
}

if [[ "${phase}" == all || "${phase}" == boundary ]]; then
    echo
    echo "=== sections 3 and 4: isolated boundaries, then depth and wrap ==="
    expected_cases=$((expected_cases + 10))
    for basis in model rotated; do
        # Nonzero logical pages and a non-identity block table throughout:
        # rebasing the selected page to zero does not test the offsets.
        boundary_case "${basis}" head 0 1 0
        boundary_case "${basis}" middle 4 1 4
        boundary_case "${basis}" tail 7 1 7
        # Two iterations exercise the depth boundary; four traverse both ring
        # slots twice. One-tile canaries cannot prove steady-state reuse.
        boundary_case "${basis}" depth 0 2 0
        boundary_case "${basis}" wrap 4 4 4
    done
fi

cube_case() {
    local leg="$1" tag="$2" strip="$3" head="$4"
    local dir="${work_dir}/cube_${leg}_${tag}"
    mkdir -p "${dir}"
    # `if ! ( ... )` rather than a bare subshell: `set -e` does not reliably
    # carry a failing subshell out of a function called at top level, and a
    # case that failed inside one came back as a clean exit once already.
    if ! (
        cd "${dir}"
        if ! timeout "${run_timeout}" \
            "${build_dir}/lib/test_camodel_paged_decode_attention" \
            --leg "${leg}" --canary "${strip}" "${head}" \
            --no-fixture-check > run.log 2>&1; then
            tail -60 run.log >&2
            exit 1
        fi
        screen_run . run.log
        grep -q '^PASS ' run.log || { echo "probe did not pass: ${dir}" >&2; exit 1; }
        # The section 5 audit over this run's own traces. The probe isolates
        # block `head * 8`, and block N runs on core N, so the audited core is
        # the one the launch actually drove. `--no-ingress-rotation` does not
        # relax the Cube's "did it run" check on a bypass leg; it changes what
        # that check asks for, and makes the prologue's absence exact.
        local core="core$((head * 8))"
        local rotation=()
        if [[ "${leg}" != contract ]]; then rotation=(--no-ingress-rotation); fi
        python3 "${repo_dir}/scripts/analyze_paged_decode_trace.py" \
            --strips 1 "${rotation[@]}" \
            "${core}.veccore0.instr_popped_log.dump" \
            "${core}.veccore0.instr_log.dump" \
            "${core}.veccore1.instr_popped_log.dump" \
            "${core}.veccore1.instr_log.dump" \
            "${core}.cubecore0.instr_popped_log.dump" \
            "${core}.cubecore0.instr_log.dump" > audit.txt
    ); then
        echo "FAILED case: ${dir}" >&2
        exit 1
    fi
    grep -h '^PASS ' "${dir}/run.log" | tee -a "${work_dir}/summary.txt"
}

if [[ "${phase}" == all || "${phase}" == cube ]]; then
    echo
    echo "=== the Cube operator's three rotation legs, one tile each ==="
    expected_cases=$((expected_cases + 6))
    for leg in contract prerotated rotated-basis; do
        # One strip of one head, with the block table's non-identity mapping
        # still deciding which physical page is read. Strip 0 and strip 7 are
        # the first and last of the page.
        cube_case "${leg}" head 0 2
        cube_case "${leg}" tail 7 2
    done
fi

if [[ "${phase}" == all || "${phase}" == aic ]]; then
    echo
    echo "=== the Cube Split-K operator's prerotated ingress ==="
    expected_cases=$((expected_cases + 1))
    dir="${work_dir}/aic_prerotated"
    mkdir -p "${dir}"
    if ! (
        cd "${dir}"
        # Narrowed to one and four pages at K = 1 and 4: a single-split launch,
        # a multi-split one, a nonzero page_begin and the empty-range neutral
        # element. Every rung floods `pi_table` with the canary and requires
        # `qrot` back untouched, so what it establishes is that the prologue is
        # gone and that nothing downstream noticed.
        if ! timeout "${run_timeout}" "${build_dir}/lib/test_decode_splitk_aic_sim" \
            --section prerotated --pages 1,4 --splitk 1,4 > run.log 2>&1; then
            tail -60 run.log >&2
            exit 1
        fi
        screen_run . run.log
        # The binary prints one PASS line per SECTION it ran, and FAIL plus a
        # nonzero status when a rung fails. The status is what decides: a
        # section selected away still printed its own PASS line in builds from
        # before that was guarded, so a PASS grep alone is not the gate.
        grep -q '^PASS AIC split-K prerotated ingress' run.log \
            || { echo "the prerotated section did not pass: ${dir}" >&2; exit 1; }
    ); then
        echo "FAILED case: ${dir}" >&2
        exit 1
    fi
    grep -hE '^  ok|^PASS ' "${dir}/run.log" | tee -a "${work_dir}/summary.txt"
fi

if [[ "${phase}" == all || "${phase}" == grid ]]; then
    echo
    echo "=== the 64-block wave, on both egress legs ==="
    expected_cases=$((expected_cases + 3))
    # 2 x 4 x 8 = 64 and 1 x 4 x 16 = 64, the two grids the raised bound
    # unlocks. M = 1 keeps each launch short; the row axis is swept by the
    # full ladder rather than here.
    grid_case full_wave_k8 1 8 both
    grid_case full_wave_k16 1 16 both
    # The control: a 32-block grid, which the old bound also admitted. If the
    # full-wave rungs failed and this passed, the wave is the difference.
    grid_case half_wave_k4 1 4 both
fi

if [[ "${phase}" == all || "${phase}" == census ]]; then
    echo
    echo "=== section 5: the instruction census across the legs ==="
    # A bypass that agrees numerically has not been shown to BE a bypass: an
    # `if constexpr` compiled the wrong way round, or a leaf that still runs and
    # happens to be the identity on the fixture, produces the same numbers.
    # These are the two claims stated as facts about the instruction stream:
    #
    #   the Vector egress leg issues strictly fewer instructions, because the
    #   butterfly gather is not in its stream at all;
    #
    #   the Cube ingress leg issues EXACTLY 64 fewer Mmads -- the rotation
    #   prologue's eight source groups times eight output groups -- leaving
    #   only the one strip's eight accumulating score Mmads behind.
    #
    # It reads the traces the `boundary` and `cube` phases left, so it runs
    # after them.
    census_dir="${work_dir}/census"
    mkdir -p "${census_dir}"
    python3 "${repo_dir}/scripts/audit_decode_rotation_census.py" \
        --leg model="${work_dir}/boundary_model_head" \
        --leg rotated="${work_dir}/boundary_rotated_head" \
        --core core0.veccore0 --expect-fewer rotated:model \
        > "${census_dir}/aiv_egress.json"
    echo "  ok  AIV egress: the rotated leg issues fewer Vector instructions"
    python3 "${repo_dir}/scripts/audit_decode_rotation_census.py" \
        --leg prerotated="${work_dir}/cube_prerotated_head" \
        --leg contract="${work_dir}/cube_contract_head" \
        --core core16.cubecore0 --expect-fewer prerotated:contract \
        --expect-delta MMAD:prerotated:contract:64 \
        > "${census_dir}/cube_ingress.json"
    echo "  ok  Cube ingress: exactly 64 Mmads absent from the prerotated leg"
    expected_cases=$((expected_cases + 1))
    echo "PASS rotation instruction census" | tee -a "${work_dir}/summary.txt"
fi

# `boundary_case` and `cube_case` append one PASS line per case; `grid_case`
# and the AIC phase append the ladder's own PASS line plus one `ok` line per
# launch, so PASS lines count cases in every phase.
passes=$(grep -c '^PASS ' "${work_dir}/summary.txt" || true)
echo
if [[ "${passes}" != "${expected_cases}" ]]; then
    echo "incomplete run: ${passes} of ${expected_cases} cases passed" >&2
    exit 1
fi
echo "PASS decode rotation CAModel gates: ${passes}/${expected_cases} cases; evidence: ${work_dir}"

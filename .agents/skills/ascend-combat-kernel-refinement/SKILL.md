---
name: ascend-combat-kernel-refinement
description: Refactor real Ascend operators in ascend-micro-harness into software pipelines with capacity-one HardEvent models, UB lifetime checks, isolated logical-page boundary canaries, ring-wrap tests and CAModel trace audits. Use for production operator pipeline or buffer-lifetime changes in this repository.
---

# Combat kernel refinement

Use `REFINEMENT_OF_COMBAT_KERNELS.md` at the repository root as the primary
verification procedure. Keep actual operator arithmetic in shared processing
functions so diagnostic entry points exercise the code being refined.

Before pushing a pipeline refactor or tagging it for hardware:

1. Prove capacity-one flags and primed feedback cycles; run the engine queue
   model in `tests/cpu/`, including negative controls and empty/short workers.
2. Compile with strict warnings and assert UB alignment, distinct live ranges,
   bank skew, capacity and the 256-byte final read pad.
3. Run one real tile at logical pages 0, K/2 and K-1, with a non-identity block
   table and only assigned slices real. Check exact boundary oracle results,
   every input byte, inactive output regions and both guards.
4. Run depth, at least twice-depth, odd-tail and real multi-worker cases.
5. Audit each run's dispatch/completion flags, empty drain and instruction
   census. Require measured overlapping engine intervals for an overlap claim.

For the two AIV decode kernels, start with
`bash scripts/run_decode_pipeline_camodel.sh`; continue with the regular varied
score and production Split-K runners. Report unavailable gates as pending and
distinguish simulator observations from hardware performance.

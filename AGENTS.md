# Kernel pipeline changes

For pipeline or buffer-lifetime refactors of real operators, follow
`REFINEMENT_OF_COMBAT_KERNELS.md` as the primary verification protocol:
capacity-one token/queue model → compile-time UB checks → isolated logical-page
head/middle/tail canaries on CAModel → depth and wrap tests → trace audit/census.
Complete the baseline before pushing a pipeline refactor or tagging
it for physical hardware. Record unavailable or failing gates explicitly.

Preserve unrelated working-tree changes. Use the existing real operator geometry
and arithmetic in probes; call shared production processing code. A host model
is not CAModel evidence, and a one-tile canary is not ring-wrap or multi-worker
evidence. Keep reports and commands under `reports/` and `scripts/`.

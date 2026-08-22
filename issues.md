# QuantitativeSusceptibilityMappingTGV.jl - issue list

Working notes from the architecture review of 2026-08-19..21. Uncommitted on
purpose. Cross-repository items are in `issues-stack.md` (X0..X7). Full
evidence: https://claude.ai/code/artifact/1dcb7a4a-4523-46fc-af23-7c5b0ecc3688

Measured on this machine unless marked *unverified*. State refers to branch
`claude/julia-repos-architecture-review-tj1zzz`.

Architecturally the best of the Julia packages: KernelAbstractions kernels give
CPU, CUDA, AMDGPU, oneAPI and Metal from one source, and it is the only package
running Aqua. Suite: 19/19 in 2m07 here, CUDA item skipped (no GPU).

---

## Done on this branch

- **F14** CI matrix moved from a hardcoded `1.7` / `1.10` to `min` / `lts` / `1`,
  matching the rest of the family.
- Registers `:tgv` and `:tgv_original` citations so any tool that runs this
  backend records the right references.

## Open

### T1. F7 - no numerical accuracy test, for the stack's only QSM engine
The suite asserts output shape on `randn` input, checks CPU against GPU, and
exercises the Laplacian variants. There is no test that the reconstruction
recovers a *known* susceptibility distribution. An analytic sphere or cylinder
phantom with a closed-form dipole field would give one, and it is a few hours of
work. Without it a regularisation or kernel regression passes CI cleanly.

The new MriResearchTools assertions bound the output to a few ppm and require
the estimators to agree with each other, which catches a scaling regression but
not a systematically wrong answer. Only a phantom catches that.

### T2. Backend selection by passing a `Module`
The ecosystem convention is now dispatch on array type. This also interacts with
MriResearchTools M1 (`qsm_B0` colliding between the two backend extensions):
solving selection properly here is most of what M1 needs.

### T3. Five near-identical standalone scripts, each doing a runtime `Pkg.add`
`tgv_qsm.jl`, `tgv_qsm_cuda.jl`, `tgv_qsm_amdgpu.jl`, `tgv_qsm_metal.jl`,
`tgv_qsm_oneapi.jl` differ essentially in one `using` line, and each performs a
`Pkg.add` into whatever environment the user happens to be in. Part of F13; the
shared CLI layer (X1) should absorb them, with the backend chosen by a flag
rather than by which file you run.

### T4. Comonicon, where everything else uses ArgParse (F13)
Not wrong on its own, but it is the fifth CLI style in the family. Fold into X1.

### T5. The standalone scripts write no provenance record
Additive, blocks nothing. Every other entry point in the family now writes
settings and citations.

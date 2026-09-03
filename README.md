# ExecuTorch interval first-fit memory planner benchmarks

Reproducibility artifacts for [pytorch/executorch#22508](https://github.com/pytorch/executorch/pull/22508). This is an external research harness, not a replacement for ExecuTorch's tests or model/export infrastructure. The upstream PR contains only the planner change and focused tests.

## Current result

The exact benchmark lock is:

- Upstream base: `457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc`
- PR head: `4e5a3906f456120dffd2ec31ea097b50dde17303`
- Patch SHA-256: `7ad25503ce5d048defd193be39ca08ce38d477f100c3b1d53ec3700d9a91241b`

| Frozen holdout | Upstream greedy | Conditional portfolio | Saved | Wins / ties / regressions |
|---|---:|---:|---:|---:|
| 10 model cases | 156,064,272 B | 136,608,768 B | 19,455,504 B (12.466%) | 8 / 2 / 0 |

All ten layouts were valid and deterministic, and all ten upstream-versus-candidate portable-runtime replays were exact. Eight cases reached the sound per-arena peak-live lower bound; the aggregate selected-layout gap was 1,806,336 bytes. See [RESULTS.md](RESULTS.md) for every model and all four planner timings.

## What is new

Current upstream `greedy()` already sorts by decreasing allocated size. Size ordering is not the contribution.

The candidate places each size-sorted lifetime interval at the first aligned legal offset. The opt-in conditional policy computes upstream greedy first, applies sound lower-bound and work-budget exits, runs interval first-fit only when allowed, and returns the smaller layout. Unsupported storage-backed aliases fall back to upstream greedy for the entire planning unit. Default greedy behavior is unchanged.

The conditional baseline uses allocation-key ordering and graph order only for exact key collisions, making equal-size early returns deterministic without scanning graph order on ordinary cases.

## Evidence boundary

- The holdout was frozen before execution and uses ExecuTorch export paths plus TorchVision model definitions.
- Planned bytes compare the same exported graph, inputs, alignment, arenas, and seed.
- Timing is the median of seven direct planner calls on each captured planning unit.
- The four timing probes run in fixed order: upstream, candidate-only, always-both, conditional.
- Timing is machine-sensitive. Byte counts, valid placement, deterministic replay, and exact portable-runtime equality are the primary gates.
- ResNet152 and Wide-ResNet101-2 have exact upstream-versus-candidate runtime equality but nonconfirmatory eager-output diagnostics; both remain visible.
- No model was skipped. The candidate was soundly skipped on two lower-bound ties.
- The 56-row historical corpus is a policy projection, not a fresh current-head runtime replay.

## Reproduce

Use a clean checkout and ExecuTorch's normal source-build prerequisites:

```bash
git clone https://github.com/seanyang0813/executorch-interval-first-fit-benchmarks.git
git clone https://github.com/pytorch/executorch.git executorch-benchmark
cd executorch-benchmark
git checkout 457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc
git apply ../executorch-interval-first-fit-benchmarks/patches/interval_first_fit.patch
./install_executorch.sh

export EXECUTORCH_ROOT="$PWD"
BENCH=../executorch-interval-first-fit-benchmarks
for case in \
  wb2_tv_convnext_large wb2_tv_densenet161 wb2_tv_efficientnet_v2_m \
  wb2_tv_mnasnet0_5 wb2_tv_regnet_y_3_2gf wb2_tv_resnet152 \
  wb2_tv_resnet34 wb2_tv_swin_s_b2 wb2_tv_swin_v2_s_256 \
  wb2_tv_wide_resnet101_2
do
  python "$BENCH/harness/benchmark_final_policy_case.py" \
    --case "$case" --out "/tmp/$case.json"
done

cd "$BENCH"
./verify_evidence.sh
```

## Repository layout

- `patches/interval_first_fit.patch`: canonical cumulative PR patch.
- `harness/benchmark_final_policy_case.py`: exact current-head model, replay, and four-mode timing runner.
- `results/frozen_holdout/`: frozen manifest, ten raw current-head rows, CSV, and summary.
- `results/controls/`: no-gain exits and retained historical failed-timing control.
- `results/randomized_correctness.json`: 10,000 differential cases, including 1,000 alias cases.
- `results/work_budget_boundary.json`: six work-policy boundary scenarios.
- `results/historical/`: source-derived historical corpus and projection.
- `PROVENANCE.md`: hashes and claim boundaries.

## AI disclosure

The patch, evaluation, and this evidence repository were produced with OpenAI Codex assistance and require human maintainer review.

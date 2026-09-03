# ExecuTorch interval first-fit memory planner benchmarks

> **Current head notice:** PR head 3e3016c410aa2456cb507d5c7e9c3a453111dc5b includes a post-benchmark determinism fix. Existing model measurements are bound to a66bd3b1dd465b1806cd682610fb94d7f7a7459e. See [CURRENT_HEAD.md](CURRENT_HEAD.md).

Reproducibility artifacts for [pytorch/executorch#22508](https://github.com/pytorch/executorch/pull/22508).

This is an external research artifact, not a replacement benchmark suite for ExecuTorch. The upstream PR contains only the planner change and focused tests.

## Claim boundary

Current ExecuTorch greedy() already sorts TensorSpecs by decreasing allocated size. Size ordering is not the contribution. The candidate places each size-sorted interval at the first aligned legal offset, then a bounded portfolio returns the smaller of upstream greedy and the candidate.

The benchmark pin is ExecuTorch commit 457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc. The benchmarked patch commit is a66bd3b1dd465b1806cd682610fb94d7f7a7459e, and patches/interval_first_fit.patch has SHA-256 af8cd429bb0209ed0682d0d0903fd7ca8697ede8daec25a8167d123637e51efb.

## Results

See [the full model-by-model index](RESULTS.md) for every frozen holdout row, controls, and exact-final provenance.

### Exact benchmarked-commit confirmations

| Model | Upstream greedy | Portfolio result | Gain | Added planner time | Runtime replay |
|---|---:|---:|---:|---:|---|
| RegNet-Y-3.2GF | 9,660,672 B | 8,128,512 B | 15.860% | 46.923 ms | exact |
| SwinV2-S 256 | 18,087,952 B | 14,155,776 B | 21.739% | 814.086 ms | exact |

Both placements were deterministic and layout-valid. The candidate result includes storage-object ID assignment used by backends.

### Exit-path controls

| Model | Exit | Upstream | Result | Added planner time |
|---|---|---:|---:|---:|
| ResNet34 | sound lower bound reached | 6,422,528 B | 6,422,528 B | -0.028 ms |
| Emformer predictor | hard work cap | 5,513,216 B | 5,513,216 B | 25.543 ms |
| MobileBERT | insufficient lower-bound potential | 819,328 B | 819,328 B | 0.337 ms |

The first clean Emformer replay was also an exact byte tie but added 233.742 ms and failed the frozen 100 ms gate. That result is retained in results/controls/registry_emformer_predict_first.json. Moving the unconditional hard-cap return before lower-bound computation produced the final 25.543 ms path without changing selected bytes.

### Broader evidence

| Evidence set | Upstream bytes | Portfolio bytes | Saved | Wins / ties / regressions | Weighted gain |
|---|---:|---:|---:|---:|---:|
| Frozen 10-case holdout | 156,064,272 | 136,608,768 | 19,455,504 | 8 / 2 / 0 | 12.466% |
| Historical 51 byte-valid rows | 3,397,206,576 | 3,353,515,712 | 43,690,864 | 21 / 30 / 0 | 1.286% |

The ten-case holdout is additional frozen evidence, not a prevalence estimate. The 51-row result is a source-derived projection of the final work-budget policy, not a fresh runtime replay of every row.

Randomized validation executed 10,000 cases, including 1,000 alias cases, with zero failures. Six boundary scenarios made the expected conditional decision and never selected a larger plan.

## Methodology

- Baseline: the pinned upstream greedy() implementation, which is already size-sorted.
- Candidate: exact first-aligned legal interval placement using ExecuTorch's closed lifetime-overlap semantics and per-memory-ID arenas.
- Portfolio: run upstream first; skip the candidate at a sound lower bound, above the hard pair-scan cap, or above the soft cap without at least two percent attainable lower-bound savings; otherwise return min(upstream, candidate).
- Alias policy: any unsupported storage-backed alias relationship falls back to upstream greedy for the entire planning unit.
- Models: existing ExecuTorch model registry/export paths and TorchVision model definitions, not hand-built interval fixtures.
- Comparison: identical exported graph, inputs, alignment, arena configuration, and seed for both planners.
- Planned bytes: activation-memory bytes reported by the same ExecuTorch planning/export path.
- Timing: median of seven direct in-planner repeats on the exported graph.
- Runtime: serialized portable programs were loaded and replayed; direct upstream/candidate outputs had to be exact.
- Accounting: failures, skips, nonconfirmatory eager comparisons, ties, and the first failed Emformer timing gate remain visible.

## Reproduce the benchmarked confirmation cases

The commands below use a clean ExecuTorch checkout and the portable copy of the original final-policy runner. Follow ExecuTorch's normal source-build prerequisites for your platform.

~~~bash
git clone https://github.com/seanyang0813/executorch-interval-first-fit-benchmarks.git
git clone https://github.com/pytorch/executorch.git executorch-benchmark
cd executorch-benchmark
git checkout 457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc
git apply ../executorch-interval-first-fit-benchmarks/patches/interval_first_fit.patch
./install_executorch.sh

export EXECUTORCH_ROOT="$PWD"
BENCH=../executorch-interval-first-fit-benchmarks
python "$BENCH/harness/benchmark_final_policy_case.py" --case wb2_tv_regnet_y_3_2gf --out /tmp/regnet.json
python "$BENCH/harness/benchmark_final_policy_case.py" --case wb2_tv_swin_v2_s_256 --out /tmp/swin.json
python "$BENCH/harness/benchmark_final_policy_case.py" --case wb2_tv_resnet34 --out /tmp/resnet34.json
python "$BENCH/harness/benchmark_final_policy_case.py" --case registry_emformer_predict --out /tmp/emformer.json
python "$BENCH/harness/benchmark_final_policy_case.py" --case registry_mobilebert --out /tmp/mobilebert.json
~~~

Inspect the normalized evidence and verify all published files:

~~~bash
cd ../executorch-interval-first-fit-benchmarks
./verify_evidence.sh
jq '.historical_projection, .frozen_holdout, .final_commit_measurements' results/benchmark_confirmation_summary.json
~~~

Timing is machine-sensitive. Planned-byte equality, policy decisions, placement validity, and deterministic runtime replay are the primary reproducibility checks.

## Repository layout

- patches/: benchmarked submitted diff.
- harness/: portable final-policy runner, its shared ExecuTorch model/export helper, randomized and boundary checkers, and the original discovery harness.
- results/final/: exact submitted-commit RegNet and SwinV2 confirmations.
- results/controls/: lower-bound, hard-cap, soft-benefit, and retained failed-timing controls.
- results/frozen_holdout/: frozen ten-case manifest, raw rows, and aggregate.
- results/historical/: sanitized 56-row source snapshot and final-policy aggregate.
- results/benchmark_confirmation_summary.json: normalized provenance and all headline measurements.
- PROVENANCE.md: hashes, adaptation boundary, and stale raw-metadata explanation.

## AI disclosure

This patch, evaluation, and evidence repository were produced with OpenAI Codex assistance and require human maintainer review.


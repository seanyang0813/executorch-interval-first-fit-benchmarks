# Provenance

## Source lock

- Pinned upstream: 457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc
- Submitted patch commit: a66bd3b1dd465b1806cd682610fb94d7f7a7459e
- Patch SHA-256: af8cd429bb0209ed0682d0d0903fd7ca8697ede8daec25a8167d123637e51efb
- Planner source SHA-256 after patch: 80aea49caeb56d2c8da4854a3d350f8276ce3500f5626a577e3e17f1eefdf90d

Upstream main advanced by nine commits after submission, but neither exir/memory_planning.py nor exir/tests/test_memory_planning.py changed in that interval.

## Harness adaptation boundary

The final case runner used during measurement had SHA-256 0cd083d71af52627f6ea955c311aa91f862a59a9a2c4a8b0a5e1754a5c8fa0. The published portable copy changes only:

- the historical BASE_COMMIT label to the actual benchmark pin;
- the historical PATCH_SHA256 label to the exact submitted patch hash;
- CORE_PATH from a machine-local absolute path to the adjacent benchmark_memory_planning.py file.

The shared benchmark helper used during measurement had SHA-256 912416723d05f7ac9adda5e84635c13ca84aa37b3eecf09d97985746d9bcdbde. Its published copy changes only the import bootstrap so EXECUTORCH_ROOT can identify a source checkout. Planner, export, timing, serialization, runtime replay, and comparison logic are unchanged.

Other original source hashes:

- Discovery harness: 21159dca0dc71a3149af60313ce560d4a64d5d6cd10d3f72fd7503a99163f0e5
- Boundary checker: d78ba901e00cdda94e8bd01b06aecb7a4b83db2a7d34ec46770e7cdd9e0c2774
- Randomized checker: 56099168d0befaa5ab69c94e11bf29d1a5a1f0ddfb88a3e96f77225c00a9f5eb

## Raw metadata boundary

Several raw model and holdout JSON files retain base_commit and candidate_patch_sha256 values from the phase in which their runner was frozen. Those embedded labels are historical and are not used as final rerun provenance. results/benchmark_confirmation_summary.json binds the final RegNet and SwinV2 measurements to the actual clean submitted worktree, patch hash, and planner source hash.

The storage-object-ID correctness change was made after the holdout was frozen. RegNet and SwinV2, the two exact confirmation wins, were rerun on the final submitted commit. Exit-path controls are path-equivalent because interval placement is not executed on those paths.

results/historical/rows.json removes machine-local stack-trace text while retaining every row, status, byte count, timing field, and failure classification. results/randomized_correctness.json normalizes one machine-local planner_source path to exir/memory_planning.py. No measurement values were changed.

## Evidence levels

- Exact confirmation: final submitted code, exact planned-byte match, deterministic placement, valid layout, serialized-program replay.
- Frozen holdout: independently selected before execution; additional evidence, not a prevalence estimate.
- Historical projection: source-derived final-policy projection, not a fresh replay of every row.
- Eager comparison: diagnostic only. Direct upstream/candidate portable-runtime equality is the correctness gate.

## Authorship

The patch, benchmark work, and this publication were produced with OpenAI Codex assistance. Human maintainers should independently review the algorithm and evidence.


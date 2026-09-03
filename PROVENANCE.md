# Provenance

## Source lock

- Pinned upstream: `457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc`
- Submitted and measured PR head: `4e5a3906f456120dffd2ec31ea097b50dde17303`
- Cumulative patch SHA-256: `7ad25503ce5d048defd193be39ca08ce38d477f100c3b1d53ec3700d9a91241b`
- Planner source SHA-256: `b7d9f1d7117cd147bd296346222a00d82e068f9ad94eb71fdb1eff9fa47d2923`
- Final case runner SHA-256: `293f8b8ac901e6563ebbdd565d1948851609068123691b8af3cd222d18ae41b9`
- Shared benchmark helper SHA-256: `145978c58b575ab624fcb74c8aa2005067017f37a8373bcc0a3fb4fcac573577`

`patches/interval_first_fit.patch` is canonical. `patches/current_pr_after_determinism_fix.patch` is retained as an identical compatibility path.

## Measurement protocol

The exact current-head holdout uses the pre-existing frozen ten-case list. Models run sequentially. Each planner timing is the median of seven direct calls against the same captured planning unit, in fixed upstream, candidate-only, always-both, conditional order. The added four-mode probes do not affect policy selection, model inputs, planned-byte accounting, serialization, or runtime comparison.

The final case runner uses ExecuTorch's export, lowering, serialization, and portable runtime paths. TorchVision cases use public model definitions with untrained weights; registry cases use ExecuTorch model factories. Inputs, seed, graph, alignment, memory IDs, and lifetime semantics are held fixed between planners.

## Evidence levels

- Current-head holdout: exact planned bytes, layout checks, deterministic placement replay, and serialized-program upstream-versus-candidate replay.
- Eager comparison: diagnostic only. Two rows are retained as nonconfirmatory while their direct portable-runtime comparison is exact.
- Randomized check: bounded differential evidence, not a proof for every graph.
- Historical corpus: source-derived policy projection, not a fresh current-head runtime replay.
- Planner timing: local machine measurement, not production latency evidence.

No current holdout model was skipped or failed. The full test-file custom-op failure and the earlier Emformer timing-gate failure are retained in [RESULTS.md](RESULTS.md).

## Authorship

The patch, benchmark work, and this publication were produced with OpenAI Codex assistance. Human maintainers should independently review the algorithm and evidence.

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
sha256sum -c CHECKSUMS.sha256 >/dev/null

jq -e '
  .pinned_upstream == "457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc" and
  .commit == "4e5a3906f456120dffd2ec31ea097b50dde17303" and
  .patch_sha256 == "7ad25503ce5d048defd193be39ca08ce38d477f100c3b1d53ec3700d9a91241b" and
  .frozen_holdout.cases == 10 and
  .frozen_holdout.wins == 8 and
  .frozen_holdout.ties == 2 and
  .frozen_holdout.regressions == 0 and
  .frozen_holdout.exact_direct_replays == 10 and
  .frozen_holdout.deterministic_layouts == 10 and
  .randomized.executed_cases == 10000 and
  .randomized.alias_cases == 1000 and
  .randomized.failure_count == 0 and
  .randomized.passed and
  .boundary.all_decisions_correct and
  .boundary.all_candidate_no_larger
' results/benchmark_confirmation_summary.json >/dev/null

jq -e '
  .head_commit == "4e5a3906f456120dffd2ec31ea097b50dde17303" and
  .aggregate.upstream_bytes == 156064272 and
  .aggregate.candidate_bytes == 136608768 and
  .aggregate.saved_bytes == 19455504 and
  .aggregate.regressions == 0 and
  (.rows | length) == 10 and
  ([.rows[].planner_median_ms | keys == ["always_both","candidate_only","conditional","upstream_greedy"]] | all)
' results/frozen_holdout/summary.json >/dev/null

for row in results/frozen_holdout/cases/*.json; do
  jq -e '.candidate_patch_sha256 == "7ad25503ce5d048defd193be39ca08ce38d477f100c3b1d53ec3700d9a91241b"' "$row" >/dev/null
  jq -e '.candidate_no_larger and .runtime_direct_exact and .deterministic_placements and .conditional_two_strategy.layout_valid' "$row" >/dev/null
done

[[ $(find results/frozen_holdout/cases -maxdepth 1 -name '*.json' | wc -l) -eq 10 ]]
jq -e '.passed and .executed_cases == 10000 and .failure_count == 0' results/randomized_correctness.json >/dev/null
jq -e '.all_decisions_correct and .all_candidate_no_larger' results/work_budget_boundary.json >/dev/null

printf '%s\n' 'Evidence checks passed.'

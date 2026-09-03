#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
sha256sum -c CHECKSUMS.sha256 >/dev/null

jq -e '
  .pinned_upstream == "457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc" and
  .commit == "a66bd3b1dd465b1806cd682610fb94d7f7a7459e" and
  .patch_sha256 == "af8cd429bb0209ed0682d0d0903fd7ca8697ede8daec25a8167d123637e51efb" and
  .historical_projection.byte_valid_rows == 51 and
  .historical_projection.wins == 21 and
  .historical_projection.ties == 30 and
  .historical_projection.regressions == 0 and
  .frozen_holdout.rows == 10 and
  .frozen_holdout.wins == 8 and
  .frozen_holdout.ties == 2 and
  .frozen_holdout.regressions == 0 and
  .randomized.executed_cases == 10000 and
  .randomized.failure_count == 0 and
  .boundary.all_decisions_correct and
  .boundary.all_candidate_no_larger
' results/benchmark_confirmation_summary.json >/dev/null

jq -e '.baseline_bytes == 9660672 and .candidate_bytes == 8128512 and .runtime_direct_exact and .deterministic_placements' results/final/wb2_tv_regnet_y_3_2gf.json >/dev/null
jq -e '.baseline_bytes == 18087952 and .candidate_bytes == 14155776 and .runtime_direct_exact and .deterministic_placements' results/final/wb2_tv_swin_v2_s_256.json >/dev/null
jq -e '.rows == 10 and .wins == 8 and .ties == 2 and .regressions == 0' results/frozen_holdout/summary.json >/dev/null
jq -e '.rows | length == 56' results/historical/rows.json >/dev/null

printf '%s\n' 'Evidence checks passed.'


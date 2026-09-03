# Current PR head

- PR: [pytorch/executorch#22508](https://github.com/pytorch/executorch/pull/22508)
- Pinned upstream: `457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc`
- Current and benchmarked head: `4e5a3906f456120dffd2ec31ea097b50dde17303`
- Patch SHA-256: `7ad25503ce5d048defd193be39ca08ce38d477f100c3b1d53ec3700d9a91241b`
- Planner source SHA-256: `b7d9f1d7117cd147bd296346222a00d82e068f9ad94eb71fdb1eff9fa47d2923`

A reviewer identified nondeterministic equal-size tie ordering on conditional early-return paths. The final code orders the opt-in conditional baseline by allocation-relevant keys and consults graph order only for exact key collisions. Default upstream greedy remains unchanged.

The exact current-head ten-case holdout, two controls, 10,000-case randomized check, six boundary scenarios, focused lint, and planner test file have been rerun. See [RESULTS.md](RESULTS.md) for outcomes and retained limitations.

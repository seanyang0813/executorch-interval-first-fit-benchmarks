# Current PR head

PR #22508 advanced from benchmarked commit a66bd3b1dd465b1806cd682610fb94d7f7a7459e to 3e3016c410aa2456cb507d5c7e9c3a453111dc5b after a reviewer identified nondeterministic equal-size tie ordering on conditional early-return paths.

The follow-up feeds graph-ordered specs to the opt-in upstream-greedy baseline. The default greedy planner is not reordered. The current patch is patches/current_pr_after_determinism_fix.patch with SHA-256 3cd008967cc32905a077ba3f19363c14b46ce239070ca4f853993b82ffc8c41e.

Existing model measurements remain bound to a66bd3b1dd465b1806cd682610fb94d7f7a7459e. Focused tests and model benchmarks have not yet been rerun on 3e3016c410aa2456cb507d5c7e9c3a453111dc5b, so this repository does not currently claim exact model evidence for the new head.

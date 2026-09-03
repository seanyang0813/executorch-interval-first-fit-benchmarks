# Current-head model results

All rows use ExecuTorch base `457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc`, PR head `4e5a3906f456120dffd2ec31ea097b50dde17303`, and patch SHA-256 `7ad25503ce5d048defd193be39ca08ce38d477f100c3b1d53ec3700d9a91241b`.

Planner columns are median milliseconds over seven direct calls: `G` is upstream greedy, `I` is candidate-only interval first-fit, `B` is always-both with minimum selection, and `C` is the proposed conditional portfolio.

| Case | Status | Upstream B | Selected B | Gain | Lower bound B | Gap B | G ms | I ms | B ms | C ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Large | ok | 41,060,352 | 40,759,296 | 0.733% | 39,555,072 | 1,204,224 | 8.981 | 24.057 | 33.205 | 34.917 |
| DenseNet161 | ok | 15,455,232 | 12,644,352 | 18.187% | 12,644,352 | 0 | 27.530 | 82.961 | 111.301 | 115.989 |
| EfficientNet-V2-M | ok | 8,732,672 | 8,429,568 | 3.471% | 7,827,456 | 602,112 | 40.310 | 129.199 | 170.266 | 172.988 |
| MNASNet-0.5 | lower-bound tie | 5,125,120 | 5,125,120 | 0.000% | 5,125,120 | 0 | 4.241 | 8.151 | 12.482 | 5.133 |
| RegNet-Y-3.2GF | ok | 9,660,672 | 8,128,512 | 15.860% | 8,128,512 | 0 | 10.719 | 24.230 | 35.051 | 36.637 |
| ResNet152 | eager nonconfirmatory | 14,614,528 | 9,633,792 | 34.081% | 9,633,792 | 0 | 22.381 | 61.163 | 83.520 | 86.366 |
| ResNet34 | lower-bound tie | 6,422,528 | 6,422,528 | 0.000% | 6,422,528 | 0 | 2.552 | 5.443 | 7.987 | 3.225 |
| Swin-S batch 2 | ok | 22,290,688 | 21,676,032 | 2.757% | 21,676,032 | 0 | 180.758 | 498.894 | 673.322 | 677.456 |
| SwinV2-S 256 | ok | 18,087,952 | 14,155,776 | 21.739% | 14,155,776 | 0 | 276.418 | 797.838 | 1,070.106 | 1,077.919 |
| Wide-ResNet101-2 | eager nonconfirmatory | 14,614,528 | 9,633,792 | 34.081% | 9,633,792 | 0 | 11.827 | 30.257 | 42.137 | 44.139 |

Aggregate: 156,064,272 upstream bytes versus 136,608,768 selected bytes, saving 19,455,504 bytes or 12.466%. There were eight wins, two ties, zero regressions, ten valid deterministic layouts, and ten exact direct portable-runtime comparisons. Median-across-case planner times were 17.104 ms (`G`), 45.710 ms (`I`), 62.828 ms (`B`), and 65.253 ms (`C`).

The two eager-nonconfirmatory rows still have exact upstream-versus-candidate portable-runtime equality. They are not presented as eager-equivalence evidence.

## Controls and failures

| Case | Current-head observation | Upstream B | Selected B | Added planner ms | Direct exact |
|---|---|---:|---:|---:|---|
| Emformer predictor | no-gain conditional exit | 5,513,216 | 5,513,216 | -31.219 | yes |
| MobileBERT | no-gain conditional exit; eager nonconfirmatory | 819,328 | 819,328 | -3.767 | yes |

The retained `registry_emformer_predict_first.json` is a historical failed 100 ms timing gate from an earlier implementation (+233.742 ms), not the current head. It remains published to preserve failure history.

No current-head model process failed or was skipped, no selected layout regressed, and no holdout case used alias fallback. The randomized checker executed 10,000 cases with zero failures, including 1,000 alias cases. All six work-boundary scenarios made the expected run/skip decision and selected no larger layout.

The full planner test file produced 51 passes and one environment failure because `torch.ops.llama.sdpa_with_kv_cache` was unavailable; the same failure was reproduced on the pinned baseline and is not attributed to this patch. Changed-file lint passed.

Raw rows are in `results/frozen_holdout/cases/`; the normalized row table is `results/frozen_holdout/results.csv`.

## Historical corpus

The historical snapshot contains 56 rows, 51 byte-valid. Its projected final policy has 21 wins, 30 ties, zero regressions, 3,397,206,576 upstream bytes, and 3,353,515,712 selected bytes, for 1.286% weighted savings. This remains a source-derived projection, not a current-head runtime replay.

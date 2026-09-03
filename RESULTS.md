# Model-by-model results

## Why results/final contains two models

The ten-case holdout was frozen and run before the final storage-object-ID correctness correction. After that correction, the predeclared exact-confirmation subset reran RegNet-Y-3.2GF and SwinV2-S 256 on benchmarked commit a66bd3b1dd465b1806cd682610fb94d7f7a7459e. Those are therefore the only model measurements labeled exact-final.

The other holdout rows remain published as earlier frozen evidence. They were not silently relabeled as final-commit reruns. ResNet34, Emformer, and MobileBERT are also retained as final-code path-equivalent controls because their conditional exits do not execute interval placement.

## Exact benchmarked-commit confirmations

| Case | Upstream bytes | Candidate bytes | Gain | Added planner time | Direct exact |
|---|---:|---:|---:|---:|---|
| RegNet-Y-3.2GF | 9,660,672 | 8,128,512 | 15.860% | 46.923 ms | yes |
| SwinV2-S 256 | 18,087,952 | 14,155,776 | 21.739% | 814.086 ms | yes |

## Frozen ten-case holdout

| Case | Status | Upstream bytes | Candidate bytes | Gain | Added planner time | Direct exact | Eager close |
|---|---|---:|---:|---:|---:|---|---|
| ConvNeXt-Large | ok | 41,060,352 | 40,759,296 | 0.733% | 29.737 ms | yes | yes |
| DenseNet161 | ok | 15,455,232 | 12,644,352 | 18.187% | 182.772 ms | yes | yes |
| EfficientNet-V2-M | ok | 8,732,672 | 8,429,568 | 3.471% | 247.442 ms | yes | yes |
| MNASNet-0.5 | ok | 5,125,120 | 5,125,120 | 0.000% | 0.608 ms | yes | yes |
| RegNet-Y-3.2GF | ok | 9,660,672 | 8,128,512 | 15.860% | 44.419 ms | yes | yes |
| ResNet152 | nonconfirmatory eager | 14,614,528 | 9,633,792 | 34.081% | 150.552 ms | yes | no |
| ResNet34 | ok | 6,422,528 | 6,422,528 | 0.000% | 0.479 ms | yes | yes |
| Swin-S batch 2 | ok | 22,290,688 | 21,676,032 | 2.757% | 475.836 ms | yes | yes |
| SwinV2-S 256 | ok | 18,087,952 | 14,155,776 | 21.739% | 854.306 ms | yes | yes |
| Wide-ResNet101-2 | nonconfirmatory eager | 14,614,528 | 9,633,792 | 34.081% | 67.076 ms | yes | no |

Aggregate: 156,064,272 upstream bytes versus 136,608,768 candidate bytes, saving 19,455,504 bytes or 12.466%. There were eight wins, two ties, zero byte regressions, and ten direct-exact portable-runtime comparisons.

Nonconfirmatory eager means the upstream and candidate portable programs matched each other exactly, but comparison against eager output did not satisfy the diagnostic closeness check. These rows remain visible and are not presented as eager-equivalence evidence.

Machine-readable rows are in results/frozen_holdout/results.csv and results/frozen_holdout/cases/.

## Exit-path and failed-timing controls

| Case | Observation | Upstream bytes | Result bytes | Added planner time | Direct exact |
|---|---|---:|---:|---:|---|
| ResNet34 | lower-bound exit | 6,422,528 | 6,422,528 | -0.028 ms | yes |
| Emformer predictor | first clean replay; failed 100 ms gate | 5,513,216 | 5,513,216 | 233.742 ms | yes |
| Emformer predictor | hard-cap-before-lower-bound final path | 5,513,216 | 5,513,216 | 25.543 ms | yes |
| MobileBERT | soft-benefit exit; eager nonconfirmatory | 819,328 | 819,328 | 0.337 ms | yes |

## Historical corpus

The historical source snapshot contains 56 rows: 18 development rows and 38 untouched-holdout rows. Fifty-one rows are byte-valid. The projected final policy has 21 wins, 30 ties, zero regressions, 3,397,206,576 upstream bytes, and 3,353,515,712 selected bytes, for 1.286% weighted savings.

This is a source-derived policy projection, not a fresh final-commit runtime replay of all 56 rows. Every row and status is available in results/historical/rows.csv and results/historical/rows.json.

## Remaining evidence gap

Current PR head 3e3016c410aa2456cb507d5c7e9c3a453111dc5b has not yet been rerun. The exact model claim remains limited to benchmarked commit a66bd3b1dd465b1806cd682610fb94d7f7a7459e. Rerunning all ten frozen cases on the current head would close that provenance gap without changing benchmark selection.


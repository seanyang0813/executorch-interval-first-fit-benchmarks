#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import platform
import resource
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils import _pytree

import scripts.benchmark_memory_planning as core
from examples.models import MODEL_NAME_TO_MODEL
from examples.models.model_factory import EagerModelFactory
from extension.llm.export.config.llm_config import (
    BaseConfig,
    ExportConfig,
    LlmConfig,
    ModelConfig,
    ModelType,
)

@dataclass(frozen=True)
class CaseSpec:
    name: str
    tier: str
    family: str
    model_name: str
    weights: str
    params: Optional[str] = None
    model_type: Optional[ModelType] = None
    sequence_length: Optional[int] = None
    context_length: Optional[int] = None
    use_kv_cache: bool = False
    use_custom_sdpa: bool = False
    dynamic_shapes: bool = False
    high_memory: bool = False
    scope_note: Optional[str] = None
    unavailable_reason: Optional[str] = None


CASES = {
    spec.name: spec
    for spec in [
        CaseSpec(
            "qwen3_0_6b_prefill_s128",
            "A",
            "Qwen3-0.6B",
            "qwen3",
            "deterministic_synthetic",
            "examples/models/qwen3/config/0_6b_config.json",
            ModelType.qwen3_0_6b,
            128,
            128,
            high_memory=True,
        ),
        CaseSpec(
            "smollm2_135m_prefill_s128",
            "A",
            "SmolLM2-135M",
            "smollm2",
            "deterministic_synthetic",
            "examples/models/smollm2/135M_config.json",
            ModelType.smollm2,
            128,
            128,
        ),
        CaseSpec(
            "smollm2_135m_kv_c128",
            "A",
            "SmolLM2-135M",
            "smollm2",
            "deterministic_synthetic",
            "examples/models/smollm2/135M_config.json",
            ModelType.smollm2,
            1,
            128,
            use_kv_cache=True,
        ),
        CaseSpec(
            "smollm2_135m_kv_custom_sdpa_c128",
            "A",
            "SmolLM2-135M",
            "smollm2",
            "deterministic_synthetic",
            "examples/models/smollm2/135M_config.json",
            ModelType.smollm2,
            1,
            128,
            use_kv_cache=True,
            use_custom_sdpa=True,
        ),
        CaseSpec(
            "phi4_mini_prefill_s128",
            "A",
            "Phi-4 Mini",
            "phi_4_mini",
            "deterministic_synthetic",
            "examples/models/phi_4_mini/config/config.json",
            ModelType.phi_4_mini,
            128,
            128,
            high_memory=True,
        ),
        CaseSpec(
            "llama3_2_1b_prefill_s128",
            "A",
            "Llama-3.2-1B",
            "llama2",
            "deterministic_synthetic",
            model_type=ModelType.llama3_2,
            sequence_length=128,
            context_length=128,
            high_memory=True,
            unavailable_reason="A local Llama-3.2-1B params JSON is required.",
        ),
        CaseSpec("vit_b_16", "B", "ViT-B/16", "vit", "pretrained"),
        CaseSpec("deit_tiny", "B", "DeiT-Tiny", "deit_tiny", "pretrained"),
        CaseSpec("whisper_base", "B", "Whisper-base", "whisper", "pretrained"),
        CaseSpec("yolo26n", "B", "YOLO26n", "yolo26", "pretrained"),
        CaseSpec(
            "smolvlm_500m_text_decoder_s128",
            "B",
            "SmolVLM-500M text decoder",
            "smolvlm",
            "deterministic_synthetic",
            "examples/models/smolvlm/500M_config.json",
            ModelType.smollm2,
            128,
            128,
            high_memory=True,
            scope_note="The registered SmolVLM wrapper covers only the LLM text decoder.",
        ),
        CaseSpec(
            "efficient_sam_vitt",
            "B",
            "EfficientSAM ViT-T",
            "efficient_sam",
            "downloaded_checkpoint",
        ),
        CaseSpec("mobilenet_v3_small", "B", "MobileNetV3-Small", "mv3", "pretrained"),
        CaseSpec("resnet50", "B", "ResNet50", "resnet50", "pretrained"),
    ]
}


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialize_synthetic_model(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 1:
                if name.endswith("bias"):
                    parameter.zero_()
                else:
                    parameter.fill_(1.0)
            else:
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
        _reset_kv_cache(model)


def _reset_kv_cache(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if ".kv_cache." in name:
                buffer.zero_()


def _build_model(spec: CaseSpec, seed: int):
    torch.manual_seed(seed)
    module_name, class_name = MODEL_NAME_TO_MODEL[spec.model_name]
    if spec.params is None:
        model, inputs, kwargs, dynamic_shapes = EagerModelFactory.create_model(
            module_name, class_name
        )
    else:
        params_path = _REPO_ROOT / spec.params
        llm_config = LlmConfig(
            base=BaseConfig(model_class=spec.model_type, params=str(params_path)),
            model=ModelConfig(
                enable_dynamic_shape=spec.dynamic_shapes,
                use_kv_cache=spec.use_kv_cache,
                use_sdpa_with_kv_cache=spec.use_custom_sdpa,
            ),
            export=ExportConfig(
                max_seq_length=spec.sequence_length,
                max_context_length=spec.context_length,
            ),
        )
        model, inputs, kwargs, dynamic_shapes = EagerModelFactory.create_model(
            module_name, class_name, llm_config=llm_config
        )
        _initialize_synthetic_model(model)
    model.eval()
    return model, inputs, kwargs, dynamic_shapes


def _set_batch(inputs: Tuple[Any, ...], batch_size: int) -> Tuple[Any, ...]:
    if batch_size == 1:
        return inputs
    if len(inputs) != 1 or not torch.is_tensor(inputs[0]) or inputs[0].shape[0] != 1:
        raise ValueError("Batch sensitivity requires one tensor input with batch size 1.")
    repeats = [batch_size] + [1] * (inputs[0].dim() - 1)
    return (inputs[0].repeat(*repeats),)


def _tensor_stats(output: Any) -> Dict[str, Any]:
    leaves, _ = _pytree.tree_flatten(output)
    tensors = []
    for value in leaves:
        if not torch.is_tensor(value):
            continue
        floating = value.is_floating_point() or value.is_complex()
        finite_count = int(torch.isfinite(value).sum()) if floating else value.numel()
        tensors.append(
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
                "finite_count": finite_count,
                "nan_count": int(torch.isnan(value).sum()) if floating else 0,
                "inf_count": int(torch.isinf(value).sum()) if floating else 0,
            }
        )
    return {
        "tensor_count": len(tensors),
        "all_finite": all(item["finite_count"] == item["numel"] for item in tensors),
        "tensors": tensors,
    }


def _result_dict(result: core.PlannerResult) -> Dict[str, Any]:
    return {key: value for key, value in asdict(result).items() if key != "executed_output"}


def _run_case(
    spec: CaseSpec,
    case_name: str,
    seed: int,
    batch_size: int,
    config_args: Dict[str, Any],
    check_correctness: bool,
    baseline_algorithm: str,
    candidate_algorithm: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        model, inputs, kwargs, dynamic_shapes = _build_model(spec, seed)
        inputs = _set_batch(inputs, batch_size)
        with torch.no_grad():
            eager_output = model(*inputs, **(kwargs or {}))
        eager_stats = _tensor_stats(eager_output)
        if spec.use_kv_cache:
            _reset_kv_cache(model)
        baseline = core._run_config(
            model,
            inputs,
            kwargs,
            dynamic_shapes,
            baseline_algorithm,
            config_args,
            check_correctness,
            eager_output,
        )
        if spec.use_kv_cache:
            _reset_kv_cache(model)
        candidate = core._run_config(
            model,
            inputs,
            kwargs,
            dynamic_shapes,
            candidate_algorithm,
            config_args,
            check_correctness,
            eager_output,
        )
    except Exception as error:
        return {
            "case": case_name,
            "tier": spec.tier,
            "family": spec.family,
            "status": "model_or_harness_failed",
            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            "wall_time_s": time.perf_counter() - started,
        }

    comparison = None
    candidate_output_stats = None
    if baseline.executed_output is not None and candidate.executed_output is not None:
        comparison = core._compare_outputs(
            baseline.executed_output, candidate.executed_output
        )
        candidate_output_stats = _tensor_stats(candidate.executed_output)

    baseline_bytes = baseline.max_activation_bytes
    candidate_bytes = candidate.max_activation_bytes
    gain_bytes = None
    gain_fraction = None
    if baseline_bytes is not None and candidate_bytes is not None:
        gain_bytes = baseline_bytes - candidate_bytes
        gain_fraction = gain_bytes / baseline_bytes if baseline_bytes else 0.0

    if baseline.status != "ok" or candidate.status != "ok":
        status = "planner_or_runtime_failed"
    elif check_correctness and comparison is None:
        status = "runtime_not_compared"
    elif check_correctness and not comparison["close"]:
        status = "planner_output_mismatch"
    elif check_correctness and not candidate_output_stats["all_finite"]:
        status = "nonfinite_runtime_output"
    else:
        status = "ok"

    params_path = _REPO_ROOT / spec.params if spec.params else None
    return {
        "case": case_name,
        "tier": spec.tier,
        "family": spec.family,
        "model_name": spec.model_name,
        "weights": spec.weights,
        "parameter_config": spec.params,
        "parameter_config_sha256": _sha256(params_path) if params_path else None,
        "sequence_length": spec.sequence_length,
        "context_length": spec.context_length,
        "batch_size": batch_size,
        "use_kv_cache": spec.use_kv_cache,
        "use_custom_sdpa": spec.use_custom_sdpa,
        "dynamic_shapes": spec.dynamic_shapes,
        "scope_note": spec.scope_note,
        "allocation_config": config_args,
        "baseline_algorithm": baseline_algorithm,
        "candidate_algorithm": candidate_algorithm,
        "status": status,
        "baseline_bytes": baseline_bytes,
        "candidate_bytes": candidate_bytes,
        "gain_bytes": gain_bytes,
        "gain_fraction": gain_fraction,
        "direct_runtime_comparison": comparison,
        "eager_output": eager_stats,
        "candidate_runtime_output": candidate_output_stats,
        "baseline": _result_dict(baseline),
        "candidate": _result_dict(candidate),
        "wall_time_s": time.perf_counter() - started,
    }


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    gains = [row["gain_fraction"] for row in valid if row.get("gain_fraction") is not None]
    baseline_total = sum(row["baseline_bytes"] for row in valid)
    candidate_total = sum(row["candidate_bytes"] for row in valid)
    planning_ratios = []
    for row in valid:
        baseline_time = row["baseline"].get("planning_time_s")
        candidate_time = row["candidate"].get("planning_time_s")
        if baseline_time and candidate_time is not None:
            planning_ratios.append(candidate_time / baseline_time)
    family = defaultdict(lambda: {"valid": 0, "wins": 0, "baseline_bytes": 0, "candidate_bytes": 0})
    for row in valid:
        item = family[row["family"]]
        item["valid"] += 1
        item["wins"] += int(row["gain_bytes"] > 0)
        item["baseline_bytes"] += row["baseline_bytes"]
        item["candidate_bytes"] += row["candidate_bytes"]
    for item in family.values():
        item["weighted_gain_fraction"] = (
            (item["baseline_bytes"] - item["candidate_bytes"]) / item["baseline_bytes"]
            if item["baseline_bytes"]
            else 0.0
        )
    return {
        "row_count": len(rows),
        "status_counts": dict(Counter(row.get("status") for row in rows)),
        "valid_count": len(valid),
        "win_count": sum(row["gain_bytes"] > 0 for row in valid),
        "tie_count": sum(row["gain_bytes"] == 0 for row in valid),
        "regression_count": sum(row["gain_bytes"] < 0 for row in valid),
        "at_least_two_percent_count": sum(gain >= 0.02 for gain in gains),
        "mean_gain_fraction": sum(gains) / len(gains) if gains else None,
        "weighted_gain_fraction": (
            (baseline_total - candidate_total) / baseline_total if baseline_total else None
        ),
        "baseline_bytes_total": baseline_total,
        "candidate_bytes_total": candidate_total,
        "mean_planning_time_ratio": (
            sum(planning_ratios) / len(planning_ratios) if planning_ratios else None
        ),
        "families": dict(family),
    }


def _csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    baseline = row.get("baseline", {})
    candidate = row.get("candidate", {})
    comparison = row.get("direct_runtime_comparison") or {}
    return {
        "case": row.get("case"),
        "tier": row.get("tier"),
        "family": row.get("family"),
        "status": row.get("status"),
        "baseline_algorithm": row.get("baseline_algorithm"),
        "candidate_algorithm": row.get("candidate_algorithm"),
        "weights": row.get("weights"),
        "sequence_length": row.get("sequence_length"),
        "context_length": row.get("context_length"),
        "batch_size": row.get("batch_size"),
        "use_kv_cache": row.get("use_kv_cache"),
        "use_custom_sdpa": row.get("use_custom_sdpa"),
        "dynamic_shapes": row.get("dynamic_shapes"),
        "baseline_bytes": row.get("baseline_bytes"),
        "candidate_bytes": row.get("candidate_bytes"),
        "gain_bytes": row.get("gain_bytes"),
        "gain_fraction": row.get("gain_fraction"),
        "baseline_planning_time_s": baseline.get("planning_time_s"),
        "candidate_planning_time_s": candidate.get("planning_time_s"),
        "baseline_export_time_s": baseline.get("export_time_s"),
        "candidate_export_time_s": candidate.get("export_time_s"),
        "candidate_selected_planners": json.dumps(candidate.get("selected_planners")),
        "allocation_count": candidate.get("allocation_count"),
        "alias_count": candidate.get("alias_count"),
        "overlap_verification_ok": candidate.get("overlap_verification_ok"),
        "baseline_program_load_ok": baseline.get("program_load_ok"),
        "candidate_program_load_ok": candidate.get("program_load_ok"),
        "baseline_runtime_replay_ok": baseline.get("runtime_replay_ok"),
        "candidate_runtime_replay_ok": candidate.get("runtime_replay_ok"),
        "planner_outputs_exact": comparison.get("exact"),
        "planner_outputs_close": comparison.get("close"),
        "planner_max_abs_error": comparison.get("max_abs_error"),
        "planner_max_rel_error": comparison.get("max_rel_error"),
        "scope_note": row.get("scope_note"),
        "error": row.get("error"),
        "wall_time_s": row.get("wall_time_s"),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    flat_rows = [_csv_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--include-high-memory", action="store_true")
    parser.add_argument("--llama3-2-params", type=Path)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--allocation-matrix", action="store_true")
    parser.add_argument("--check-correctness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alloc-graph-input", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alloc-graph-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--baseline", choices=sorted(core.ALGO_REGISTRY), default="greedy"
    )
    parser.add_argument(
        "--candidate",
        choices=sorted(core.ALGO_REGISTRY),
        default="greedy_size_first",
    )
    parser.add_argument("--out-json", type=Path, default=Path("memory_planning_modern_results.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("memory_planning_modern_results.csv"))
    args = parser.parse_args()

    if args.list:
        for name in CASES:
            print(name)
        return

    selected = list(CASES) if args.cases == ["all"] else args.cases
    unknown = sorted(set(selected) - set(CASES))
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")

    allocation_configs = [(args.alloc_graph_input, args.alloc_graph_output)]
    if args.allocation_matrix:
        allocation_configs = [(False, False), (True, False), (False, True), (True, True)]

    rows: List[Dict[str, Any]] = []
    for selected_name in selected:
        spec = CASES[selected_name]
        if selected_name == "llama3_2_1b_prefill_s128" and args.llama3_2_params:
            params = args.llama3_2_params.resolve()
            spec = replace(spec, params=str(params), unavailable_reason=None)
        if spec.unavailable_reason:
            rows.append(
                {
                    "case": spec.name,
                    "tier": spec.tier,
                    "family": spec.family,
                    "status": "skipped_unavailable",
                    "error": spec.unavailable_reason,
                }
            )
            continue
        if spec.high_memory and not args.include_high_memory:
            rows.append(
                {
                    "case": spec.name,
                    "tier": spec.tier,
                    "family": spec.family,
                    "status": "skipped_resource",
                    "error": "Pass --include-high-memory to run on a host with sufficient RAM.",
                }
            )
            continue
        for batch_size in args.batch_sizes:
            for alloc_input, alloc_output in allocation_configs:
                suffix = ""
                if len(args.batch_sizes) > 1:
                    suffix += f"__b{batch_size}"
                if args.allocation_matrix:
                    suffix += f"__input{int(alloc_input)}_output{int(alloc_output)}"
                config_args = {
                    "alloc_graph_input": alloc_input,
                    "alloc_graph_output": alloc_output,
                    "alloc_mutable_buffers": True,
                    "share_mutable_buffers": False,
                    "enable_non_cpu_memory_planning": True,
                }
                row = _run_case(
                    spec,
                    spec.name + suffix,
                    args.seed,
                    batch_size,
                    config_args,
                    args.check_correctness,
                    args.baseline,
                    args.candidate,
                )
                rows.append(row)
                print(json.dumps(_csv_row(row), sort_keys=True), flush=True)

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "seed": args.seed,
            "portable_runtime_available": core.HAS_EXEC_RUNTIME,
            "memory_planning_sha256": _sha256(_REPO_ROOT / "exir/memory_planning.py"),
            "benchmark_sha256": _sha256(_REPO_ROOT / "scripts/benchmark_memory_planning.py"),
        },
        "evidence_boundary": {
            "baseline": args.baseline,
            "candidate": args.candidate,
            "runtime": "portable CPU pybinding",
            "synthetic_weights": "Architecture-valid but not pretrained accuracy evidence.",
            "dynamic_shapes": "Static unless the row explicitly says otherwise.",
        },
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    _write_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()

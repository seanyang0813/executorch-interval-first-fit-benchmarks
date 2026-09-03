#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare memory-planning algorithms across example models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
import sys
import types
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils._pytree import tree_flatten


_REPO_ROOT = Path(os.environ.get("EXECUTORCH_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_local_executorch_namespace() -> None:
    """Allow importing `executorch.*` when running directly from a source checkout."""

    if "executorch" in sys.modules:
        return

    # This repository keeps package roots as sibling directories (exir/, extension/, examples/),
    # so exposing the repo root on the package path is sufficient.
    namespace = types.ModuleType("executorch")
    namespace.__path__ = [str(_REPO_ROOT)]
    namespace.__file__ = os.fspath(_REPO_ROOT / "__init__.py")
    sys.modules["executorch"] = namespace


try:
    import executorch  # noqa: F401
except ModuleNotFoundError:
    _ensure_local_executorch_namespace()
    import executorch  # noqa: F401


from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.extension.export_util.utils import export_to_edge
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.memory_planning import (
    _arena_base_offsets,
    _peak_live_lower_bound_bufsizes,
    _stable_spec_order,
    greedy,
    greedy_interval_first_fit_conditional,
    MemoryAlgoResult,
    MemoryPlanningAlgorithmSuite,
    naive,
    SpecAllocResult,
    interval_first_fit,
)
from executorch.exir.passes import MemoryPlanningPass

try:
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    HAS_EXEC_RUNTIME = True
except Exception:
    _load_for_executorch_from_buffer = None
    HAS_EXEC_RUNTIME = False


@dataclass
class PlannerResult:
    status: str
    export_time_s: Optional[float] = None
    max_activation_bytes: Optional[int] = None
    method_bytes: Optional[Dict[str, int]] = None
    method_buffer_sizes: Optional[Dict[str, List[int]]] = None
    error: Optional[str] = None
    correctness_ok: Optional[bool] = None
    correctness_error: Optional[str] = None
    correctness_time_s: Optional[float] = None
    planning_time_s: Optional[float] = None
    allocation_count: Optional[int] = None
    alias_count: Optional[int] = None
    selected_planners: Optional[List[str]] = None
    overlap_verification_ok: Optional[bool] = None
    program_load_ok: Optional[bool] = None
    runtime_replay_ok: Optional[bool] = None
    eager_output_exact: Optional[bool] = None
    eager_output_close: Optional[bool] = None
    eager_max_abs_error: Optional[float] = None
    eager_max_rel_error: Optional[float] = None
    planning_structure: Optional[List[Dict[str, Any]]] = None
    executed_output: Any = None


def greedy_interval_first_fit(
    alignment, specs, graph_module, graph_signature, extra_padding=0
):
    greedy_result = greedy(
        alignment, specs, graph_module, graph_signature, extra_padding
    )
    if any(spec.storage_base is not None for spec in specs):
        return greedy_result
    candidate_result = interval_first_fit(
        alignment, specs, graph_module, graph_signature, extra_padding
    )
    return min(
        (greedy_result, candidate_result),
        key=lambda result: sum(result.bufsizes),
    )


def chronological_interval_first_fit(
    alignment, specs, graph_module, graph_signature, extra_padding=0
):
    if any(spec.storage_base is not None for spec in specs):
        return greedy(
            alignment, specs, graph_module, graph_signature, extra_padding
        )
    if not specs:
        return MemoryAlgoResult({}, [0, 0])

    for spec in specs:
        spec.realign(alignment)
    order = _stable_spec_order(specs, graph_module, graph_signature)
    sorted_specs = sorted(
        specs,
        key=lambda spec: (
            spec.lifetime[0],
            -spec.allocated_memory,
            spec.lifetime[1],
            order[spec],
        ),
    )
    mem_ids = {
        spec.mem_id if spec.mem_id is not None else 1 for spec in specs
    }
    arena_bases = _arena_base_offsets(graph_module, mem_ids, alignment)
    result = MemoryAlgoResult({}, [0] * (max(mem_ids) + 1))

    for spec in sorted_specs:
        mem_id = spec.mem_id if spec.mem_id is not None else 1
        occupied_ranges = sorted(
            (
                allocation.mem_offset,
                allocation.mem_offset + other_spec.allocated_memory,
            )
            for other_spec, allocation in result.spec_dict.items()
            if allocation.mem_id == mem_id
            and not (
                other_spec.lifetime[1] < spec.lifetime[0]
                or spec.lifetime[1] < other_spec.lifetime[0]
            )
        )
        offset = arena_bases[mem_id]
        for occupied_start, occupied_end in occupied_ranges:
            if occupied_end <= offset:
                continue
            if offset + spec.allocated_memory <= occupied_start:
                break
            offset = max(offset, occupied_end)
        result.spec_dict[spec] = SpecAllocResult(mem_id, 0, offset)
        result.bufsizes[mem_id] = max(
            result.bufsizes[mem_id], offset + spec.allocated_memory
        )

    for mem_id in mem_ids:
        result.bufsizes[mem_id] += extra_padding
    return result


def greedy_interval_full_portfolio(
    alignment, specs, graph_module, graph_signature, extra_padding=0
):
    results = [
        algorithm(
            alignment, specs, graph_module, graph_signature, extra_padding
        )
        for algorithm in (
            greedy,
            interval_first_fit,
            chronological_interval_first_fit,
        )
    ]
    return min(results, key=lambda result: sum(result.bufsizes))


ALGO_REGISTRY = {
    "greedy": greedy,
    "greedy_interval_first_fit": greedy_interval_first_fit,
    "greedy_interval_first_fit_conditional": greedy_interval_first_fit_conditional,
    "naive": naive,
    "interval_first_fit": interval_first_fit,
}


def _format_plan_metrics(executorch_program) -> Tuple[int, Dict[str, int], Dict[str, List[int]]]:
    max_activation_bytes = 0
    method_bytes: Dict[str, int] = {}
    method_buffer_sizes: Dict[str, List[int]] = {}

    for execution_plan in executorch_program.executorch_program.execution_plan:
        bufsizes = list(execution_plan.non_const_buffer_sizes)
        method_name = execution_plan.name
        activation_bytes = sum(bufsizes[1:])
        method_bytes[method_name] = activation_bytes
        method_buffer_sizes[method_name] = bufsizes
        max_activation_bytes = max(max_activation_bytes, activation_bytes)

    return max_activation_bytes, method_bytes, method_buffer_sizes


def _compare_outputs(expected_output: Any, actual_output: Any) -> Dict[str, Any]:
    expected_leaves, _ = tree_flatten(expected_output)
    actual_leaves, _ = tree_flatten(actual_output)
    result: Dict[str, Any] = {
        "exact": True,
        "close": True,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    if len(expected_leaves) != len(actual_leaves):
        result["exact"] = False
        result["close"] = False
        return result

    for expected_leaf, actual_leaf in zip(expected_leaves, actual_leaves):
        if torch.is_tensor(expected_leaf):
            if (
                not torch.is_tensor(actual_leaf)
                or expected_leaf.shape != actual_leaf.shape
                or expected_leaf.dtype != actual_leaf.dtype
            ):
                result["exact"] = False
                result["close"] = False
                continue

            exact = torch.equal(expected_leaf, actual_leaf)
            result["exact"] = result["exact"] and exact
            if expected_leaf.is_floating_point() or expected_leaf.is_complex():
                if expected_leaf.dtype in (torch.float16, torch.bfloat16):
                    rtol, atol = 1e-2, 4e-3
                else:
                    rtol, atol = 1e-4, 1e-4
                result["close"] = result["close"] and torch.allclose(
                    expected_leaf, actual_leaf, rtol=rtol, atol=atol
                )
                difference = (expected_leaf - actual_leaf).abs().float()
                denominator = actual_leaf.abs().float().clamp_min(1e-30)
                result["max_abs_error"] = max(
                    result["max_abs_error"], difference.max().item()
                )
                result["max_rel_error"] = max(
                    result["max_rel_error"],
                    (difference / denominator).max().item(),
                )
            else:
                result["close"] = result["close"] and exact
            continue

        exact = type(expected_leaf) is type(actual_leaf) and expected_leaf == actual_leaf
        result["exact"] = result["exact"] and exact
        result["close"] = result["close"] and exact

    return result


def _run_config(
    model: torch.nn.Module,
    example_inputs: Tuple[Any, ...],
    example_kwarg_inputs: Optional[Dict[str, Any]],
    dynamic_shapes: Any,
    algo_name: str,
    config_args: Dict[str, Any],
    check_correctness: bool,
    eager_output: Any,
) -> PlannerResult:
    planning_observations: List[Dict[str, Any]] = []
    algorithm: Callable[..., Any] = ALGO_REGISTRY[algo_name]

    def summarize_structure(specs, algo_result) -> Dict[str, Any]:
        allocations = []
        events: Dict[int, List[int]] = {}
        lifetime_lengths = []
        lifetime_starts = []
        for spec in specs:
            if spec.storage_base is not None:
                continue
            try:
                size = int(spec.allocated_memory)
                start, end = (int(value) for value in spec.lifetime)
            except (AttributeError, TypeError, ValueError):
                continue
            if size <= 0 or end < start:
                continue
            allocations.append(size)
            lifetime_lengths.append(end - start + 1)
            lifetime_starts.append(start)
            events.setdefault(start, [0, 0])
            events[start][0] += size
            events[start][1] += 1
            events.setdefault(end + 1, [0, 0])
            events[end + 1][0] -= size
            events[end + 1][1] -= 1

        live_bytes = 0
        live_allocations = 0
        peak_live_bytes = 0
        peak_live_allocations = 0
        for _, (byte_delta, allocation_delta) in sorted(events.items()):
            live_bytes += byte_delta
            live_allocations += allocation_delta
            peak_live_bytes = max(peak_live_bytes, live_bytes)
            peak_live_allocations = max(peak_live_allocations, live_allocations)

        sizes_descending = sorted(allocations, reverse=True)
        sizes_ascending = list(reversed(sizes_descending))
        requested_bytes = sum(sizes_descending)
        plan_bytes = sum(int(size) for size in algo_result.bufsizes[1:])

        def percentile(values, fraction):
            if not values:
                return None
            index = round((len(values) - 1) * fraction)
            return values[index]

        def correlation(left, right):
            if len(left) < 2 or len(left) != len(right):
                return None
            left_mean = sum(left) / len(left)
            right_mean = sum(right) / len(right)
            numerator = sum(
                (left_value - left_mean) * (right_value - right_mean)
                for left_value, right_value in zip(left, right)
            )
            left_variance = sum((value - left_mean) ** 2 for value in left)
            right_variance = sum((value - right_mean) ** 2 for value in right)
            denominator = (left_variance * right_variance) ** 0.5
            return numerator / denominator if denominator else None

        mean_size = requested_bytes / len(allocations) if allocations else None
        size_variance = (
            sum((size - mean_size) ** 2 for size in allocations) / len(allocations)
            if allocations
            else None
        )
        timeline_steps = max(events) if events else 0
        long_lived_bytes = sum(
            size
            for size, lifetime in zip(allocations, lifetime_lengths)
            if timeline_steps and lifetime * 2 >= timeline_steps
        )
        return {
            "plan_bytes": plan_bytes,
            "primary_allocation_count": len(allocations),
            "requested_bytes": requested_bytes,
            "peak_live_bytes_lower_bound": peak_live_bytes,
            "excess_over_live_bytes": max(0, plan_bytes - peak_live_bytes),
            "packing_efficiency_lower_bound": (
                peak_live_bytes / plan_bytes if plan_bytes else None
            ),
            "peak_live_allocation_count": peak_live_allocations,
            "largest_allocation_bytes": (
                sizes_descending[0] if sizes_descending else 0
            ),
            "median_allocation_bytes": percentile(sizes_ascending, 0.5),
            "p90_allocation_bytes": percentile(sizes_ascending, 0.9),
            "distinct_allocation_size_count": len(set(allocations)),
            "allocation_size_coefficient_of_variation": (
                size_variance**0.5 / mean_size
                if mean_size and size_variance is not None
                else None
            ),
            "top_five_allocation_share": (
                sum(sizes_descending[:5]) / requested_bytes
                if requested_bytes
                else None
            ),
            "mean_lifetime_steps": (
                sum(lifetime_lengths) / len(lifetime_lengths)
                if lifetime_lengths
                else None
            ),
            "timeline_steps": timeline_steps,
            "long_lived_requested_byte_share": (
                long_lived_bytes / requested_bytes if requested_bytes else None
            ),
            "size_start_correlation": correlation(
                allocations, lifetime_starts
            ),
            "size_lifetime_correlation": correlation(
                allocations, lifetime_lengths
            ),
        }

    @wraps(algorithm)
    def measured_algorithm(
        alignment, specs, graph_module, graph_signature, extra_padding=0
    ):
        planning_start = time.perf_counter()
        algo_result = algorithm(
            alignment,
            specs,
            graph_module,
            graph_signature,
            extra_padding,
        )
        planning_time_s = time.perf_counter() - planning_start
        lower_bound_bufsizes = _peak_live_lower_bound_bufsizes(
            alignment,
            specs,
            graph_module,
            extra_padding,
        )
        structure = summarize_structure(specs, algo_result)
        structure.update(
            {
                "sound_peak_live_lower_bound_bufsizes": lower_bound_bufsizes,
                "sound_peak_live_lower_bound_bytes": (
                    sum(lower_bound_bufsizes)
                    if lower_bound_bufsizes is not None
                    else None
                ),
                "gap_to_sound_lower_bound_bytes": (
                    sum(algo_result.bufsizes) - sum(lower_bound_bufsizes)
                    if lower_bound_bufsizes is not None
                    else None
                ),
            }
        )
        if algo_name == "interval_first_fit":
            planner_probes = {
                "upstream_greedy": greedy,
                "interval_first_fit": interval_first_fit,
                "always_both": greedy_interval_first_fit,
                "conditional": greedy_interval_first_fit_conditional,
                "chronological_first_fit": chronological_interval_first_fit,
                "full_portfolio": greedy_interval_full_portfolio,
            }
            probe_times = {}
            probe_bufsizes = {}
            for probe_name, probe in planner_probes.items():
                samples = []
                probe_result = None
                for _ in range(3):
                    probe_start = time.perf_counter()
                    probe_result = probe(
                        alignment,
                        specs,
                        graph_module,
                        graph_signature,
                        extra_padding,
                    )
                    samples.append(time.perf_counter() - probe_start)
                probe_times[probe_name] = sorted(samples)[1]
                probe_bufsizes[probe_name] = probe_result.bufsizes
            structure["planner_time_median_s"] = probe_times
            structure["planner_probe_bufsizes"] = probe_bufsizes
            structure["larger_portfolio_extra_gain_bytes"] = (
                sum(probe_bufsizes["always_both"])
                - sum(probe_bufsizes["full_portfolio"])
            )
            structure["conditional_skipped_candidate"] = (
                lower_bound_bufsizes is not None
                and probe_bufsizes["upstream_greedy"] == lower_bound_bufsizes
            )
        planning_observations.append(
            {
                "time_s": planning_time_s,
                "allocation_count": len(specs),
                "alias_count": sum(
                    spec.storage_base is not None for spec in specs
                ),
                "selected_planner": getattr(
                    algo_result, "algorithm_name", None
                )
                or algo_name,
                "structure": structure,
            }
        )
        return algo_result

    planner = MemoryPlanningPass(
        memory_planning_algo=MemoryPlanningAlgorithmSuite(
            algo_list=[measured_algorithm]
        ),
        alloc_graph_input=config_args["alloc_graph_input"],
        alloc_graph_output=config_args["alloc_graph_output"],
        alloc_mutable_buffers=config_args["alloc_mutable_buffers"],
        share_mutable_buffers=config_args["share_mutable_buffers"],
    )
    backend_passes = []
    initialized_mutable_buffer_names = config_args.get(
        "initialized_mutable_buffer_names", []
    )
    if initialized_mutable_buffer_names:
        from executorch.exir.passes.init_mutable_pass import (
            InitializedMutableBufferPass,
        )

        backend_passes.append(
            InitializedMutableBufferPass(initialized_mutable_buffer_names)
        )

    backend_config = ExecutorchBackendConfig(
        passes=backend_passes,
        memory_planning_pass=planner,
        enable_non_cpu_memory_planning=config_args["enable_non_cpu_memory_planning"],
    )

    try:
        start = time.perf_counter()
        edge_program = export_to_edge(
            model,
            example_inputs,
            example_kwarg_inputs=example_kwarg_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        executorch_program = edge_program.to_executorch(config=backend_config)
        export_time_s = time.perf_counter() - start
    except Exception as e:
        return PlannerResult(
            status="export_failed",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            planning_time_s=sum(
                observation["time_s"] for observation in planning_observations
            ),
            allocation_count=sum(
                observation["allocation_count"]
                for observation in planning_observations
            ),
            alias_count=sum(
                observation["alias_count"] for observation in planning_observations
            ),
        )

    (
        max_activation_bytes,
        method_bytes,
        method_buffer_sizes,
    ) = _format_plan_metrics(executorch_program)

    result = PlannerResult(
        status="ok",
        export_time_s=export_time_s,
        max_activation_bytes=max_activation_bytes,
        method_bytes=method_bytes,
        method_buffer_sizes=method_buffer_sizes,
        planning_time_s=sum(
            observation["time_s"] for observation in planning_observations
        ),
        allocation_count=sum(
            observation["allocation_count"] for observation in planning_observations
        ),
        alias_count=sum(
            observation["alias_count"] for observation in planning_observations
        ),
        selected_planners=list(
            dict.fromkeys(
                observation["selected_planner"]
                for observation in planning_observations
            )
        ),
        planning_structure=[
            observation["structure"] for observation in planning_observations
        ],
        overlap_verification_ok=(algo_name in ALGO_REGISTRY),
    )

    if not check_correctness:
        return result

    runtime_inputs, _ = torch.utils._pytree.tree_flatten(
        (example_inputs, example_kwarg_inputs or {})
    )
    if any(not torch.is_tensor(inp) for inp in runtime_inputs):
        result.correctness_ok = False
        result.correctness_error = (
            "Skipping correctness check because flattened example inputs are not all tensors."
        )
        return result

    if not HAS_EXEC_RUNTIME or _load_for_executorch_from_buffer is None:
        result.correctness_ok = False
        result.correctness_error = (
            "Skipping correctness check because portable runtime is unavailable."
        )
        return result

    try:
        correct_start = time.perf_counter()
        et_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        result.program_load_ok = True
        cloned_runtime_inputs = [input_value.clone() for input_value in runtime_inputs]
        executed = et_module.run_method("forward", cloned_runtime_inputs)
        result.runtime_replay_ok = True
        comparison = _compare_outputs(eager_output, executed)
        result.correctness_ok = comparison["close"]
        result.eager_output_exact = comparison["exact"]
        result.eager_output_close = comparison["close"]
        result.eager_max_abs_error = comparison["max_abs_error"]
        result.eager_max_rel_error = comparison["max_rel_error"]
        result.executed_output = executed
        result.correctness_time_s = time.perf_counter() - correct_start
    except Exception as e:
        result.correctness_ok = False
        result.correctness_error = f"{type(e).__name__}: {e}"
        result.status = "correctness_failed"

    return result


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    header = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _to_output_rows(
    model_name: str,
    baseline: PlannerResult,
    candidate: PlannerResult,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _record(label: str, result: PlannerResult) -> Dict[str, Any]:
        return {
            "model_name": model_name,
            "planner": label,
            "status": result.status,
            "export_time_s": result.export_time_s,
            "max_activation_bytes": result.max_activation_bytes,
            "method_bytes": json.dumps(result.method_bytes),
            "method_buffer_sizes": json.dumps(result.method_buffer_sizes),
            "error": result.error,
            "correctness_ok": result.correctness_ok,
            "correctness_error": result.correctness_error,
            "correctness_time_s": result.correctness_time_s,
            "planning_time_s": result.planning_time_s,
            "allocation_count": result.allocation_count,
            "alias_count": result.alias_count,
            "selected_planners": json.dumps(result.selected_planners),
            "overlap_verification_ok": result.overlap_verification_ok,
            "program_load_ok": result.program_load_ok,
            "runtime_replay_ok": result.runtime_replay_ok,
            "eager_output_exact": result.eager_output_exact,
            "eager_output_close": result.eager_output_close,
            "eager_max_abs_error": result.eager_max_abs_error,
            "eager_max_rel_error": result.eager_max_rel_error,
        }

    rows.append(_record("baseline", baseline))
    rows.append(_record("candidate", candidate))

    gain_bytes = None
    gain_ratio = None
    if baseline.max_activation_bytes is not None and candidate.max_activation_bytes is not None:
        gain_bytes = candidate.max_activation_bytes - baseline.max_activation_bytes
        if baseline.max_activation_bytes > 0:
            gain_ratio = gain_bytes / baseline.max_activation_bytes

    rows.append(
        {
            "model_name": model_name,
            "planner": "gain(candidate-baseline)",
            "status": "ok" if gain_bytes is not None else "skipped",
            "export_time_s": None,
            "max_activation_bytes": gain_bytes,
            "method_bytes": None,
            "method_buffer_sizes": None,
            "error": None,
            "correctness_ok": None,
            "correctness_error": None,
            "correctness_time_s": gain_ratio,
        }
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark memory-planning algorithms for example models."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Model names to run. Defaults to all entries in MODEL_NAME_TO_MODEL.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Run only this many models from the chosen model list.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed used before constructing each model.",
    )
    parser.add_argument(
        "--baseline",
        choices=sorted(ALGO_REGISTRY.keys()),
        default="greedy",
        help="Baseline memory-planning algorithm.",
    )
    parser.add_argument(
        "--candidate",
        choices=sorted(ALGO_REGISTRY.keys()),
        default="greedy_interval_first_fit",
        help="Candidate memory-planning algorithm.",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Run forward on loaded executorch module and compare to eager.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("memory_planning_benchmark.json"),
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to write CSV results.",
    )
    parser.add_argument(
        "--alloc-graph-input",
        action="store_true",
        help="Enable planning of graph inputs.",
    )
    parser.set_defaults(alloc_graph_input=True)
    parser.add_argument(
        "--no-alloc-graph-input",
        action="store_false",
        dest="alloc_graph_input",
        help="Disable planning of graph inputs.",
    )
    parser.add_argument(
        "--alloc-graph-output",
        action="store_true",
        help="Enable planning of graph outputs.",
    )
    parser.set_defaults(alloc_graph_output=True)
    parser.add_argument(
        "--no-alloc-graph-output",
        action="store_false",
        dest="alloc_graph_output",
        help="Disable planning of graph outputs.",
    )
    parser.add_argument(
        "--alloc-mutable-buffers",
        action="store_true",
        help="Enable planning of mutable buffers.",
    )
    parser.set_defaults(alloc_mutable_buffers=True)
    parser.add_argument(
        "--no-alloc-mutable-buffers",
        action="store_false",
        dest="alloc_mutable_buffers",
        help="Disable planning of mutable buffers.",
    )
    parser.add_argument(
        "--share-mutable-buffers",
        action="store_true",
        help="Whether to share mutable buffers across entry points.",
    )
    parser.set_defaults(share_mutable_buffers=False)
    parser.add_argument(
        "--device-aware-planning",
        action="store_true",
        help="Enable device-aware memory planning.",
    )
    parser.set_defaults(device_aware_planning=True)
    parser.add_argument(
        "--no-device-aware-planning",
        dest="device_aware_planning",
        action="store_false",
        help="Disable device-aware memory planning.",
    )

    args = parser.parse_args()

    model_names = args.models or list(MODEL_NAME_TO_MODEL.keys())
    if args.max_models is not None:
        model_names = model_names[: args.max_models]

    config_args = {
        "alloc_graph_input": args.alloc_graph_input,
        "alloc_graph_output": args.alloc_graph_output,
        "alloc_mutable_buffers": args.alloc_mutable_buffers,
        "share_mutable_buffers": args.share_mutable_buffers,
        "enable_non_cpu_memory_planning": args.device_aware_planning,
    }

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for model_name in model_names:
        try:
            torch.manual_seed(args.seed)
            eager_model, example_inputs, example_kwarg_inputs, dynamic_shapes = (
                EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[model_name])
            )
            eager_model.eval()
            example_inputs = tuple(example_inputs)
            with torch.no_grad():
                eager_output = eager_model(
                    *example_inputs, **(example_kwarg_inputs or {})
                )
        except Exception as e:
            summary_rows.append(
                {
                    "model_name": model_name,
                    "planner": "summary",
                    "status": "model_init_failed",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue

        baseline_result = _run_config(
            eager_model,
            example_inputs,
            example_kwarg_inputs,
            dynamic_shapes,
            args.baseline,
            config_args,
            args.check_correctness,
            eager_output,
        )
        candidate_result = _run_config(
            eager_model,
            example_inputs,
            example_kwarg_inputs,
            dynamic_shapes,
            args.candidate,
            config_args,
            args.check_correctness,
            eager_output,
        )

        all_rows.extend(_to_output_rows(model_name, baseline_result, candidate_result))
        baseline_bytes = baseline_result.max_activation_bytes
        candidate_bytes = candidate_result.max_activation_bytes
        gain = None
        if baseline_bytes is not None and candidate_bytes is not None and baseline_bytes > 0:
            gain = (baseline_bytes - candidate_bytes) / baseline_bytes

        runtime_equivalence = None
        if (
            baseline_result.executed_output is not None
            and candidate_result.executed_output is not None
        ):
            runtime_equivalence = _compare_outputs(
                baseline_result.executed_output,
                candidate_result.executed_output,
            )

        if baseline_result.status != "ok" or candidate_result.status != "ok":
            summary_status = "partial_or_failed"
        elif runtime_equivalence is not None and not runtime_equivalence["close"]:
            summary_status = "planner_output_mismatch"
        elif args.check_correctness and runtime_equivalence is None:
            summary_status = "runtime_unverified"
        else:
            summary_status = "ok"

        summary_rows.append(
            {
                "model_name": model_name,
                "baseline": args.baseline,
                "candidate": args.candidate,
                "status": summary_status,
                "baseline_status": baseline_result.status,
                "candidate_status": candidate_result.status,
                "baseline_bytes": baseline_bytes,
                "candidate_bytes": candidate_bytes,
                "gain_fraction": gain,
                "gain_bytes": (
                    baseline_bytes - candidate_bytes
                    if baseline_bytes is not None and candidate_bytes is not None
                    else None
                ),
                "baseline_planning_time_s": baseline_result.planning_time_s,
                "candidate_planning_time_s": candidate_result.planning_time_s,
                "baseline_selected_planners": baseline_result.selected_planners,
                "candidate_selected_planners": candidate_result.selected_planners,
                "allocation_count": candidate_result.allocation_count,
                "alias_count": candidate_result.alias_count,
                "overlap_verification_ok": candidate_result.overlap_verification_ok,
                "baseline_program_load_ok": baseline_result.program_load_ok,
                "candidate_program_load_ok": candidate_result.program_load_ok,
                "baseline_runtime_replay_ok": baseline_result.runtime_replay_ok,
                "candidate_runtime_replay_ok": candidate_result.runtime_replay_ok,
                "planner_outputs_exact": (
                    runtime_equivalence["exact"]
                    if runtime_equivalence is not None
                    else None
                ),
                "planner_outputs_close": (
                    runtime_equivalence["close"]
                    if runtime_equivalence is not None
                    else None
                ),
                "planner_max_abs_error": (
                    runtime_equivalence["max_abs_error"]
                    if runtime_equivalence is not None
                    else None
                ),
                "planner_max_rel_error": (
                    runtime_equivalence["max_rel_error"]
                    if runtime_equivalence is not None
                    else None
                ),
                "baseline_eager_output_close": baseline_result.eager_output_close,
                "candidate_eager_output_close": candidate_result.eager_output_close,
                "baseline_eager_max_abs_error": baseline_result.eager_max_abs_error,
                "candidate_eager_max_abs_error": candidate_result.eager_max_abs_error,
            }
        )

    # Write outputs.
    args.out_json.write_text(
        json.dumps(
            {
                "metadata": {
                    "seed": args.seed,
                    "check_correctness": args.check_correctness,
                    **config_args,
                },
                "runs": all_rows,
                "summary": summary_rows,
            },
            indent=2,
        )
    )
    if args.out_csv is not None:
        _write_summary_csv(args.out_csv, all_rows + summary_rows)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()

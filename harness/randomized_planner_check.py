#!/usr/bin/env python3

import argparse
import hashlib
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import torch
from torch.export.graph_signature import ExportGraphSignature
from torch.fx import Graph, GraphModule

from executorch.exir.memory_planning import (
    _peak_live_lower_bound_bufsizes,
    greedy,
    greedy_interval_first_fit_conditional,
)
from executorch.exir.tensor import TensorSpec


SCHEMA_VERSION = 1
MASTER_SEED = 0xE7C0_2026
PATTERNS = (
    "equal_lifetimes",
    "nested_lifetimes",
    "disjoint_lifetimes",
    "high_overlap",
    "sparse_overlap",
    "boundary_inclusive",
)
ALIGNMENTS = (1, 2, 4, 8, 16, 32, 64, 128)


def make_spec(size: int, lifetime: Sequence[int], mem_id: int) -> TensorSpec:
    spec = TensorSpec.from_tensor(torch.empty(size, dtype=torch.uint8))
    spec.lifetime = list(lifetime)
    spec.mem_id = mem_id
    return spec


def graph_module(specs: Sequence[TensorSpec], bases: Sequence[int]) -> GraphModule:
    graph = Graph()
    nodes = []
    for index, spec in enumerate(specs):
        node = graph.placeholder(f"input_{index}")
        node.meta["spec"] = spec
        nodes.append(node)
    graph.output(tuple(nodes))
    module = GraphModule({}, graph)
    module.input_mem_buffer_sizes = list(bases)
    return module


def canonical(result, specs: Sequence[TensorSpec]) -> Tuple[Any, ...]:
    return (
        tuple(result.bufsizes),
        tuple(
            (
                allocation.mem_id,
                allocation.mem_obj_id,
                allocation.mem_offset,
            )
            for allocation in (result.spec_dict[spec] for spec in specs)
        ),
    )


def lifetime_pattern(
    rng: random.Random, pattern: str, count: int
) -> List[Tuple[int, int]]:
    if pattern == "equal_lifetimes":
        end = rng.randint(0, 30)
        return [(0, end)] * count
    if pattern == "nested_lifetimes":
        return [(index, 2 * count - index) for index in range(count)]
    if pattern == "disjoint_lifetimes":
        return [(2 * index, 2 * index) for index in range(count)]
    if pattern == "high_overlap":
        middle = rng.randint(10, 30)
        return [
            (rng.randint(0, middle), rng.randint(middle, middle + 30))
            for _ in range(count)
        ]
    if pattern == "sparse_overlap":
        return [
            (start := rng.randint(0, count * 5), start + rng.randint(0, 2))
            for _ in range(count)
        ]
    if pattern == "boundary_inclusive":
        return [
            (index // 2, index // 2 + (index % 2)) for index in range(count)
        ]
    raise KeyError(pattern)


def sizes(rng: random.Random, count: int, skewed: bool) -> List[int]:
    if skewed:
        values = [1, 2, 3, 7, 15, 31, 127, 1024, 65536, 1048576]
        return [rng.choice(values) for _ in range(count)]
    return [rng.randint(1, 65536) for _ in range(count)]


def validate_layout(
    alignment: int,
    specs: Sequence[TensorSpec],
    result,
) -> List[str]:
    failures = []
    primary = []
    for spec in specs:
        if spec not in result.spec_dict:
            failures.append("missing_allocation")
            continue
        allocation = result.spec_dict[spec]
        if allocation.mem_id >= len(result.bufsizes):
            failures.append("mem_id_outside_bufsizes")
            continue
        if allocation.mem_offset + int(spec.allocated_memory) > result.bufsizes[allocation.mem_id]:
            failures.append("allocation_outside_arena")
        if spec.storage_base is not None:
            base = result.spec_dict[spec.storage_base]
            if allocation.mem_id != base.mem_id:
                failures.append("alias_mem_id_mismatch")
            if allocation.mem_obj_id != base.mem_obj_id:
                failures.append("alias_mem_obj_id_mismatch")
            if allocation.mem_offset != base.mem_offset + spec.storage_base_offset:
                failures.append("alias_offset_mismatch")
            continue
        if allocation.mem_offset % alignment:
            failures.append("unaligned_primary")
        primary.append(
            (
                allocation.mem_id,
                allocation.mem_offset,
                allocation.mem_offset + int(spec.allocated_memory),
                int(spec.lifetime[0]),
                int(spec.lifetime[1]),
                allocation.mem_obj_id,
            )
        )
    primary.sort()
    for index, left in enumerate(primary):
        for right in primary[index + 1 :]:
            if right[0] != left[0]:
                if right[0] > left[0]:
                    break
                continue
            if right[1] >= left[2]:
                break
            if left[5] != right[5]:
                failures.append("storage_overlap_mem_obj_id_mismatch")
            if not (left[4] < right[3] or right[4] < left[3]):
                failures.append("live_spatial_overlap")
                break
    return sorted(set(failures))


def make_instance(rng: random.Random, index: int) -> Dict[str, Any]:
    pattern = PATTERNS[index % len(PATTERNS)]
    count = rng.randint(1, 64)
    alignment = ALIGNMENTS[index % len(ALIGNMENTS)]
    lifetimes = lifetime_pattern(rng, pattern, count)
    allocation_sizes = sizes(rng, count, skewed=index % 3 == 0)
    mem_ids = [1 + rng.randrange(3) for _ in range(count)]
    specs = [
        make_spec(size, lifetime, mem_id)
        for size, lifetime, mem_id in zip(allocation_sizes, lifetimes, mem_ids)
    ]
    aliases = []
    if index % 10 == 0 and specs:
        base = specs[rng.randrange(len(specs))]
        base_size = int(base.nbytes())
        offset = alignment * rng.randint(0, (base_size - 1) // alignment)
        child_size = rng.randint(1, base_size - offset)
        child = make_spec(child_size, base.lifetime, int(base.mem_id))
        child.storage_base = base
        child.storage_base_offset = offset
        specs.append(child)
        aliases.append(len(specs) - 1)
    max_mem_id = max(int(spec.mem_id) for spec in specs)
    bases = [0] * (max_mem_id + 1)
    for mem_id in range(1, max_mem_id + 1):
        bases[mem_id] = alignment * rng.randint(0, 4)
    return {
        "pattern": pattern,
        "alignment": alignment,
        "extra_padding": alignment * (index % 3),
        "specs": specs,
        "bases": bases,
        "alias_indices": aliases,
        "sizes": allocation_sizes + [int(specs[i].allocated_memory) for i in aliases],
        "lifetimes": [list(spec.lifetime) for spec in specs],
        "mem_ids": [int(spec.mem_id) for spec in specs],
    }


def check_instance(instance: Dict[str, Any], rng: random.Random) -> List[str]:
    specs = instance["specs"]
    module = graph_module(specs, instance["bases"])
    alignment = instance["alignment"]
    padding = instance["extra_padding"]
    signature = cast(ExportGraphSignature, None)
    spec_set = set(specs)
    for spec in specs:
        spec.realign(alignment)
    baseline = greedy(alignment, spec_set, module, signature, padding)
    candidate = greedy_interval_first_fit_conditional(
        alignment, spec_set, module, signature, padding
    )
    failures = []
    if sum(candidate.bufsizes) > sum(baseline.bufsizes):
        failures.append("candidate_larger_than_upstream")
    failures.extend(f"baseline_{value}" for value in validate_layout(alignment, specs, baseline))
    failures.extend(f"candidate_{value}" for value in validate_layout(alignment, specs, candidate))

    aliases = instance["alias_indices"]
    lower_bound = _peak_live_lower_bound_bufsizes(
        alignment, spec_set, module, padding
    )
    if aliases:
        if lower_bound is not None:
            failures.append("alias_lower_bound_not_disabled")
        if canonical(candidate, specs) != canonical(baseline, specs):
            failures.append("alias_fallback_not_exact")
    else:
        if lower_bound is None:
            failures.append("missing_lower_bound")
        elif sum(lower_bound) > sum(candidate.bufsizes):
            failures.append("lower_bound_exceeds_candidate")

    expected = canonical(candidate, specs)
    for _ in range(3):
        repeated = greedy_interval_first_fit_conditional(
            alignment, spec_set, module, signature, padding
        )
        if canonical(repeated, specs) != expected:
            failures.append("nondeterministic_identical_input")
            break

    orders = [list(specs), list(reversed(specs))]
    shuffled = list(specs)
    rng.shuffle(shuffled)
    orders.append(shuffled)
    for order in orders:
        permutation_set = set(order)
        permutation_baseline = greedy(
            alignment, permutation_set, module, signature, padding
        )
        permutation_candidate = greedy_interval_first_fit_conditional(
            alignment, permutation_set, module, signature, padding
        )
        if sum(permutation_candidate.bufsizes) > sum(permutation_baseline.bufsizes):
            failures.append("permutation_candidate_larger_than_upstream")
        failures.extend(
            f"permutation_candidate_{value}"
            for value in validate_layout(alignment, specs, permutation_candidate)
        )
        permutation_lower_bound = _peak_live_lower_bound_bufsizes(
            alignment, permutation_set, module, padding
        )
        if aliases:
            if canonical(permutation_candidate, specs) != canonical(
                permutation_baseline, specs
            ):
                failures.append("permutation_alias_fallback_not_exact")
        elif (
            permutation_lower_bound is None
            or sum(permutation_lower_bound) > sum(permutation_candidate.bufsizes)
        ):
            failures.append("permutation_lower_bound_invalid")
    return sorted(set(failures))


def public_instance(instance: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "pattern": instance["pattern"],
        "alignment": instance["alignment"],
        "extra_padding": instance["extra_padding"],
        "bases": instance["bases"],
        "sizes": instance["sizes"],
        "lifetimes": instance["lifetimes"],
        "mem_ids": instance["mem_ids"],
        "alias_indices": instance["alias_indices"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(MASTER_SEED)
    started = time.perf_counter()
    failure_records = []
    pattern_counts = {pattern: 0 for pattern in PATTERNS}
    alias_cases = 0
    alignments = {str(value): 0 for value in ALIGNMENTS}
    for index in range(args.cases):
        instance = make_instance(rng, index)
        pattern_counts[instance["pattern"]] += 1
        alignments[str(instance["alignment"])] += 1
        alias_cases += bool(instance["alias_indices"])
        failures = check_instance(instance, rng)
        if failures:
            failure_records.append(
                {
                    "failures": failures,
                    "instance": public_instance(instance, index),
                }
            )
            if len(failure_records) >= 20:
                break
    executed = sum(pattern_counts.values())
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": "457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc",
        "planner_source": inspect.getsourcefile(greedy),
        "planner_source_sha256": hashlib.sha256(
            Path(cast(str, inspect.getsourcefile(greedy))).read_bytes()
        ).hexdigest(),
        "master_seed": MASTER_SEED,
        "requested_cases": args.cases,
        "executed_cases": executed,
        "patterns": pattern_counts,
        "alignments": alignments,
        "alias_cases": alias_cases,
        "failures": failure_records,
        "failure_count": len(failure_records),
        "passed": executed == args.cases and not failure_records,
        "mechanical_checks": [
            "simultaneously live primary allocations do not overlap",
            "primary offsets satisfy target alignment",
            "all allocations fit reported arenas",
            "spatially overlapping allocations share a memory object ID",
            "identical input produces identical placement",
            "candidate plan is no larger than upstream",
            "sound per-arena lower bound does not exceed either valid plan",
            "storage-backed aliases exactly fall back to upstream on the same input",
            "allocation-order permutations independently preserve all invariants"
        ],
        "deterministic_regressions_added": [
            {
                "test": "TestIntervalFirstFitMemoryPlanning.test_conditional_returns_upstream_on_lower_bound_tie",
                "defect": "Pinned upstream greedy can violate its decreasing-size invariant when target alignment differs from TensorSpec alignment.",
                "fix": "Normalize specs inside the opt-in wrapper before invoking unchanged upstream greedy."
            }
        ],
        "wall_time_s": time.perf_counter() - started,
        "evidence_boundary": (
            "Randomized bug-finding evidence, not a proof. The master seed and up to "
            "twenty complete counterexamples are retained."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "executed_cases": executed,
        "alias_cases": alias_cases,
        "failure_count": len(failure_records),
        "passed": artifact["passed"],
        "wall_time_s": artifact["wall_time_s"],
    }, sort_keys=True))
    if not artifact["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

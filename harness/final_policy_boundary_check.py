#!/usr/bin/env python3

import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch
from torch.export.graph_signature import ExportGraphSignature
from torch.fx import Graph, GraphModule

import executorch.exir.memory_planning as planning
from executorch.exir.tensor import TensorSpec

ALIGNMENT = 16
REPEATS = 3
SCENARIOS = (
    ("below_soft_material", 2000, False),
    ("above_soft_material", 2001, False),
    ("above_soft_low_potential", 2001, True),
    ("below_hard_material", 3162, False),
    ("above_hard_material", 3163, False),
    ("far_above_hard_material", 8000, False),
)


def make_spec(size: int, start: int, end: int) -> TensorSpec:
    value = TensorSpec.from_tensor(torch.empty(size, dtype=torch.uint8))
    value.lifetime = [start, end]
    value.mem_id = 1
    return value


def make_specs(count: int, dominant: bool):
    values = [make_spec(4096, 0, 100_000)] if dominant else []
    block = 0
    while len(values) < count:
        start = block * 6
        values.extend((
            make_spec(100, start, start),
            make_spec(60, start + 1, start + 2),
            make_spec(40, start + 2, start + 3),
            make_spec(30, start + 3, start + 4),
        ))
        block += 1
    return values[:count]


def make_graph(specs):
    graph = Graph()
    nodes = []
    for index, value in enumerate(specs):
        node = graph.placeholder(f"input_{index}")
        node.meta["spec"] = value
        nodes.append(node)
    graph.output(tuple(nodes))
    return GraphModule({}, graph)


def call(function, specs, graph):
    return function(
        ALIGNMENT,
        set(specs),
        graph,
        cast(ExportGraphSignature, None),
        0,
    )


def main():
    rows = []
    for name, count, dominant in SCENARIOS:
        specs = make_specs(count, dominant)
        graph = make_graph(specs)
        upstream = call(planning.greedy, specs, graph)
        lower_bound = planning._peak_live_lower_bound_bufsizes(
            ALIGNMENT, set(specs), graph
        )
        scans = count * (count - 1) // 2
        upstream_bytes = sum(upstream.bufsizes)
        potential_percent = 100 * (upstream_bytes - sum(lower_bound)) / upstream_bytes
        expected_run = scans <= planning._INTERVAL_FIRST_FIT_HARD_MAX_PAIR_SCANS and (
            scans <= planning._INTERVAL_FIRST_FIT_SOFT_MAX_PAIR_SCANS
            or potential_percent
            >= planning._INTERVAL_FIRST_FIT_MIN_POTENTIAL_SAVINGS_PERCENT
        )
        samples = []
        original = planning.interval_first_fit
        with patch.object(planning, "interval_first_fit", wraps=original) as candidate:
            guarded = None
            for _ in range(REPEATS):
                started = time.perf_counter()
                guarded = call(
                    planning.greedy_interval_first_fit_conditional, specs, graph
                )
                samples.append(time.perf_counter() - started)
        rows.append({
            "scenario": name,
            "interval_count": count,
            "pair_scan_estimate": scans,
            "potential_savings_percent": potential_percent,
            "expected_candidate_calls": REPEATS if expected_run else 0,
            "candidate_calls": candidate.call_count,
            "decision_correct": candidate.call_count == (REPEATS if expected_run else 0),
            "upstream_bytes": upstream_bytes,
            "guarded_bytes": sum(guarded.bufsizes),
            "candidate_no_larger": sum(guarded.bufsizes) <= upstream_bytes,
            "guarded_median_ms": 1000 * statistics.median(samples),
            "guarded_samples_s": samples,
        })
    source = Path(cast(str, inspect.getsourcefile(planning.greedy)))
    artifact = {
        "schema_version": 1,
        "base_commit": "c65ad53ad6861de3d9a76f31ddaf882a7115176f",
        "planner_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "repeats": REPEATS,
        "rows": rows,
        "all_decisions_correct": all(row["decision_correct"] for row in rows),
        "all_candidate_no_larger": all(row["candidate_no_larger"] for row in rows),
    }
    Path("results/memory_planning_work_budget/final_policy_boundary_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "all_decisions_correct": artifact["all_decisions_correct"],
        "all_candidate_no_larger": artifact["all_candidate_no_larger"],
        "rows": [{key: row[key] for key in (
            "scenario", "pair_scan_estimate", "potential_savings_percent",
            "candidate_calls", "expected_candidate_calls", "guarded_median_ms",
        )} for row in rows],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

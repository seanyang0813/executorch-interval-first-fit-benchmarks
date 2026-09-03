#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_map


BASE_COMMIT = "457a2a8b9f7d103765d73752c5d2efc6b2e8c8bc"
PATCH_SHA256 = "af8cd429bb0209ed0682d0d0903fd7ca8697ede8daec25a8167d123637e51efb"
REPEATS = 7
CORE_PATH = Path(__file__).with_name("benchmark_memory_planning.py")


def _load_core():
    spec = importlib.util.spec_from_file_location("pr_ready_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load core benchmark helpers from {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = _load_core()


@dataclass
class Case:
    name: str
    category: str
    family: str
    model: torch.nn.Module
    example_args: Tuple[Any, ...]
    example_kwargs: Dict[str, Any]
    dynamic_shapes: Any = None
    replay_inputs: Optional[List[Tuple[Tuple[Any, ...], Dict[str, Any]]]] = None
    weights: str = "deterministic_synthetic"


class FirstTensorOutput(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args):
        return self.model(*args)["out"]


class BertOutput(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]


class DecoderOutput(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]


class BartOutput(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]


class DynamicConvPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, 1),
        )

    def forward(self, value):
        return self.layers(value)


class DynamicTransformerEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )

    def forward(self, value):
        return self.encoder(value)


def _registry_case(name: str, registry_name: str, category: str, family: str) -> Case:
    from examples.models import MODEL_NAME_TO_MODEL
    from examples.models.model_factory import EagerModelFactory

    module_name, class_name = MODEL_NAME_TO_MODEL[registry_name]
    model, args, kwargs, dynamic_shapes = EagerModelFactory.create_model(
        module_name, class_name
    )
    return Case(
        name=name,
        category=category,
        family=family,
        model=model.eval(),
        example_args=tuple(args),
        example_kwargs=dict(kwargs or {}),
        dynamic_shapes=dynamic_shapes,
        weights="registry_default",
    )


def _classification_case(name: str, constructor_name: str, family: str) -> Case:
    import torchvision.models as models

    kwargs: Dict[str, Any] = {"weights": None}
    if constructor_name in {"googlenet", "inception_v3"}:
        kwargs.update(aux_logits=False, init_weights=False)
    model = getattr(models, constructor_name)(**kwargs).eval()
    side = (
        256
        if name == "wb2_tv_swin_v2_s_256"
        else 299 if constructor_name == "inception_v3" else 224
    )
    batch = 2 if name == "wb2_tv_swin_s_b2" else 1
    return Case(
        name=name,
        category=(
            "vision transformer"
            if constructor_name.startswith(("vit_", "swin_", "maxvit"))
            else "image classification"
        ),
        family=family,
        model=model,
        example_args=(torch.randn(batch, 3, side, side),),
        example_kwargs={},
    )


def _segmentation_case(name: str, constructor_name: str, family: str) -> Case:
    import torchvision.models.segmentation as models

    model = getattr(models, constructor_name)(
        weights=None,
        weights_backbone=None,
    ).eval()
    return Case(
        name=name,
        category="segmentation",
        family=family,
        model=FirstTensorOutput(model),
        example_args=(torch.randn(1, 3, 224, 224),),
        example_kwargs={},
    )


def _detection_case(name: str, constructor_name: str, family: str) -> Case:
    import torchvision.models.detection as models

    kwargs: Dict[str, Any] = {"weights": None, "weights_backbone": None}
    if constructor_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        kwargs.update(min_size=320, max_size=320, box_score_thresh=0.5)
    elif constructor_name == "retinanet_resnet50_fpn_v2":
        kwargs.update(min_size=320, max_size=320, score_thresh=0.5)
    model = getattr(models, constructor_name)(**kwargs).eval()
    return Case(
        name=name,
        category="detection",
        family=family,
        model=model,
        example_args=([torch.randn(3, 320, 320)],),
        example_kwargs={},
    )


def _nlp_case(name: str) -> Case:
    import transformers

    vocab = 2048
    input_ids = torch.randint(3, vocab, (1, 128), dtype=torch.int64)
    mask = torch.ones_like(input_ids)
    common = {
        "example_args": (input_ids, mask),
        "example_kwargs": {},
        "category": "transformer/NLP",
    }
    if name == "hf_bert_tiny_s128":
        config = transformers.BertConfig(
            vocab_size=vocab,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=128,
            use_cache=False,
        )
        return Case(
            name=name,
            family="BERT",
            model=BertOutput(transformers.BertModel(config).eval()),
            **common,
        )
    if name == "hf_distilbert_tiny_s128":
        config = transformers.DistilBertConfig(
            vocab_size=vocab,
            max_position_embeddings=128,
            n_layers=2,
            n_heads=4,
            dim=128,
            hidden_dim=256,
            dropout=0.0,
            attention_dropout=0.0,
        )
        return Case(
            name=name,
            family="DistilBERT",
            model=BertOutput(transformers.DistilBertModel(config).eval()),
            **common,
        )
    if name == "hf_gpt2_tiny_s128":
        config = transformers.GPT2Config(
            vocab_size=vocab,
            n_positions=128,
            n_embd=128,
            n_layer=2,
            n_head=4,
            n_inner=256,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        common["category"] = "LLM-like decoder"
        return Case(
            name=name,
            family="GPT-2",
            model=DecoderOutput(transformers.GPT2Model(config).eval()),
            **common,
        )
    if name == "hf_opt_tiny_s128":
        config = transformers.OPTConfig(
            vocab_size=vocab,
            hidden_size=128,
            num_hidden_layers=2,
            ffn_dim=256,
            max_position_embeddings=128,
            num_attention_heads=4,
            dropout=0.0,
            attention_dropout=0.0,
            use_cache=False,
        )
        common["category"] = "LLM-like decoder"
        return Case(
            name=name,
            family="OPT",
            model=DecoderOutput(transformers.OPTModel(config).eval()),
            **common,
        )
    if name == "hf_bloom_tiny_s128":
        config = transformers.BloomConfig(
            vocab_size=vocab,
            hidden_size=128,
            n_layer=2,
            n_head=4,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cache=False,
        )
        common["category"] = "LLM-like decoder"
        return Case(
            name=name,
            family="BLOOM",
            model=DecoderOutput(transformers.BloomModel(config).eval()),
            **common,
        )
    if name == "hf_bart_tiny_s128":
        decoder_ids = torch.randint(3, vocab, (1, 16), dtype=torch.int64)
        decoder_mask = torch.ones_like(decoder_ids)
        config = transformers.BartConfig(
            vocab_size=vocab,
            max_position_embeddings=128,
            encoder_layers=2,
            encoder_ffn_dim=256,
            encoder_attention_heads=4,
            decoder_layers=2,
            decoder_ffn_dim=256,
            decoder_attention_heads=4,
            d_model=128,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            use_cache=False,
        )
        return Case(
            name=name,
            category="encoder-decoder transformer",
            family="BART",
            model=BartOutput(transformers.BartModel(config).eval()),
            example_args=(input_ids, mask, decoder_ids, decoder_mask),
            example_kwargs={},
        )
    raise KeyError(name)


def build_case(name: str) -> Case:
    registry = {
        "registry_dl3": ("dl3", "segmentation", "DeepLab registry"),
        "registry_ds_cnn": ("ds_cnn", "audio/small mobile", "DS-CNN"),
        "registry_edsr": ("edsr", "image restoration", "EDSR"),
        "registry_emformer_join": ("emformer_join", "audio/speech", "Emformer"),
        "registry_emformer_predict": ("emformer_predict", "audio/speech", "Emformer"),
        "registry_emformer_transcribe": (
            "emformer_transcribe",
            "audio/speech",
            "Emformer",
        ),
        "registry_mobilebert": ("mobilebert", "transformer/NLP", "MobileBERT"),
        "registry_mobilenet_v1_025": (
            "mobilenet_v1_025",
            "small mobile classification",
            "MobileNetV1",
        ),
        "registry_w2l": ("w2l", "audio/speech", "Wav2Letter"),
    }
    if name in registry:
        return _registry_case(name, *registry[name])

    classification = {
        "tv_alexnet": ("alexnet", "AlexNet"),
        "tv_squeezenet1_1": ("squeezenet1_1", "SqueezeNet"),
        "tv_densenet121": ("densenet121", "DenseNet"),
        "tv_shufflenet_v2_x1_0": ("shufflenet_v2_x1_0", "ShuffleNetV2"),
        "tv_mnasnet1_0": ("mnasnet1_0", "MNASNet"),
        "tv_efficientnet_b0": ("efficientnet_b0", "EfficientNet"),
        "tv_regnet_y_400mf": ("regnet_y_400mf", "RegNet"),
        "tv_googlenet": ("googlenet", "GoogLeNet"),
        "tv_inception_v3": ("inception_v3", "InceptionV3"),
        "tv_vgg16": ("vgg16", "VGG"),
        "tv_vit_b_32": ("vit_b_32", "ViT"),
        "tv_swin_v2_t": ("swin_v2_t", "SwinV2"),
        "tv_maxvit_t": ("maxvit_t", "MaxViT"),
        "tv_convnext_small": ("convnext_small", "ConvNeXt"),
        "wb_tv_resnext50_32x4d": ("resnext50_32x4d", "ResNeXt"),
        "wb_tv_shufflenet_v2_x2_0": ("shufflenet_v2_x2_0", "ShuffleNetV2"),
        "wb_tv_densenet201": ("densenet201", "DenseNet"),
        "wb_tv_efficientnet_v2_s": ("efficientnet_v2_s", "EfficientNetV2"),
        "wb_tv_convnext_base": ("convnext_base", "ConvNeXt"),
        "wb_tv_regnet_y_1_6gf": ("regnet_y_1_6gf", "RegNet"),
        "wb_tv_swin_s": ("swin_s", "Swin"),
        "wb_tv_swin_b": ("swin_b", "Swin"),
        "wb_tv_swin_v2_s": ("swin_v2_s", "SwinV2"),
        "wb_tv_swin_v2_b": ("swin_v2_b", "SwinV2"),
        "wb2_tv_resnet34": ("resnet34", "ResNet"),
        "wb2_tv_resnet152": ("resnet152", "ResNet"),
        "wb2_tv_wide_resnet101_2": ("wide_resnet101_2", "WideResNet"),
        "wb2_tv_densenet161": ("densenet161", "DenseNet"),
        "wb2_tv_efficientnet_v2_m": ("efficientnet_v2_m", "EfficientNetV2"),
        "wb2_tv_convnext_large": ("convnext_large", "ConvNeXt"),
        "wb2_tv_regnet_y_3_2gf": ("regnet_y_3_2gf", "RegNet"),
        "wb2_tv_mnasnet0_5": ("mnasnet0_5", "MNASNet"),
        "wb2_tv_swin_s_b2": ("swin_s", "Swin"),
        "wb2_tv_swin_v2_s_256": ("swin_v2_s", "SwinV2"),
    }
    if name in classification:
        return _classification_case(name, *classification[name])

    segmentation = {
        "tv_fcn_resnet50": ("fcn_resnet50", "FCN"),
        "tv_deeplabv3_mobilenet_v3_large": (
            "deeplabv3_mobilenet_v3_large",
            "DeepLabV3",
        ),
        "tv_lraspp_mobilenet_v3_large": (
            "lraspp_mobilenet_v3_large",
            "LR-ASPP",
        ),
    }
    if name in segmentation:
        return _segmentation_case(name, *segmentation[name])

    detection = {
        "tv_ssdlite320_mobilenet_v3_large": (
            "ssdlite320_mobilenet_v3_large",
            "SSD-Lite",
        ),
        "tv_fasterrcnn_mobilenet_v3_large_320_fpn": (
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "Faster R-CNN",
        ),
        "tv_retinanet_resnet50_fpn_v2": (
            "retinanet_resnet50_fpn_v2",
            "RetinaNet",
        ),
    }
    if name in detection:
        return _detection_case(name, *detection[name])

    if name.startswith("hf_"):
        return _nlp_case(name)

    if name == "audio_conformer_small":
        from torchaudio.models import Conformer

        model = Conformer(
            input_dim=80,
            num_heads=4,
            ffn_dim=128,
            num_layers=2,
            depthwise_conv_kernel_size=31,
            dropout=0.0,
        ).eval()
        return Case(
            name=name,
            category="audio/speech",
            family="Conformer",
            model=model,
            example_args=(torch.randn(1, 128, 80), torch.tensor([128])),
            example_kwargs={},
        )

    if name == "dynamic_conv_pipeline":
        height = torch.export.Dim("height", min=32, max=96)
        width = torch.export.Dim("width", min=32, max=96)
        return Case(
            name=name,
            category="dynamic-shape program",
            family="dynamic CNN",
            model=DynamicConvPipeline().eval(),
            example_args=(torch.randn(1, 3, 64, 64),),
            example_kwargs={},
            dynamic_shapes=({2: height, 3: width},),
            replay_inputs=[
                ((torch.randn(1, 3, 32, 48),), {}),
                ((torch.randn(1, 3, 64, 64),), {}),
                ((torch.randn(1, 3, 80, 96),), {}),
            ],
        )

    if name == "dynamic_transformer_encoder":
        tokens = torch.export.Dim("tokens", min=8, max=64)
        return Case(
            name=name,
            category="dynamic-shape program",
            family="dynamic transformer encoder",
            model=DynamicTransformerEncoder().eval(),
            example_args=(torch.randn(1, 32, 64),),
            example_kwargs={},
            dynamic_shapes=({1: tokens},),
            replay_inputs=[
                ((torch.randn(1, 8, 64),), {}),
                ((torch.randn(1, 32, 64),), {}),
                ((torch.randn(1, 64, 64),), {}),
            ],
        )

    raise KeyError(name)


def _clone(value: Any) -> Any:
    return tree_map(lambda item: item.clone() if torch.is_tensor(item) else item, value)


def _finite(value: Any) -> bool:
    leaves, _ = tree_flatten(value)
    return all(
        not torch.is_tensor(leaf)
        or not (leaf.is_floating_point() or leaf.is_complex())
        or bool(torch.isfinite(leaf).all())
        for leaf in leaves
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _canonical(result, specs) -> Tuple[Any, ...]:
    placements = tuple(
        sorted(
            (
                id(spec),
                allocation.mem_id,
                allocation.mem_obj_id,
                allocation.mem_offset,
            )
            for spec, allocation in result.spec_dict.items()
        )
    )
    return tuple(result.bufsizes), placements


def _analyze_plan(alignment, specs, graph_module, result, lower_bound):
    primary = []
    failures = []
    for spec in specs:
        allocation = result.spec_dict.get(spec)
        if allocation is None:
            failures.append("missing allocation")
            continue
        if spec.storage_base is not None:
            continue
        size = int(spec.allocated_memory)
        start, end = (int(value) for value in spec.lifetime)
        if size <= 0 or end < start:
            continue
        primary.append(
            {
                "spec": spec,
                "mem_id": int(allocation.mem_id),
                "offset": int(allocation.mem_offset),
                "size": size,
                "start": start,
                "end": end,
            }
        )
        if allocation.mem_offset % alignment:
            failures.append("unaligned allocation")
        if allocation.mem_id >= len(result.bufsizes):
            failures.append("memory id outside bufsizes")
        elif allocation.mem_offset + size > result.bufsizes[allocation.mem_id]:
            failures.append("allocation exceeds arena")

    by_memory: Dict[int, List[Dict[str, Any]]] = {}
    for item in primary:
        by_memory.setdefault(item["mem_id"], []).append(item)

    overlap_pairs = 0
    possible_pairs = 0
    peak_live_count = 0
    peak_live_bytes = 0
    arena_summaries = []
    bases = core._arena_base_offsets(graph_module, set(by_memory), alignment)
    for mem_id, allocations in sorted(by_memory.items()):
        possible_pairs += len(allocations) * (len(allocations) - 1) // 2
        events = []
        for item in allocations:
            events.append((item["start"], 0, item))
            events.append((item["end"], 1, item))
        active = 0
        active_bytes = 0
        arena_peak_count = 0
        arena_peak_bytes = 0
        peak_time = 0
        for timestamp, kind, item in sorted(events, key=lambda event: (event[0], event[1])):
            if kind == 0:
                overlap_pairs += active
                active += 1
                active_bytes += item["size"]
                if active_bytes > arena_peak_bytes:
                    arena_peak_bytes = active_bytes
                    arena_peak_count = active
                    peak_time = timestamp
            else:
                active -= 1
                active_bytes -= item["size"]
        peak_live_count = max(peak_live_count, arena_peak_count)
        peak_live_bytes += arena_peak_bytes

        spatial = sorted(allocations, key=lambda item: (item["offset"], item["size"]))
        for index, left in enumerate(spatial):
            left_end = left["offset"] + left["size"]
            for right in spatial[index + 1 :]:
                if right["offset"] >= left_end:
                    break
                lifetime_overlap = not (
                    left["end"] < right["start"] or right["end"] < left["start"]
                )
                if lifetime_overlap:
                    failures.append("simultaneously live allocations overlap")
                    break

        live_ranges = sorted(
            (item["offset"], item["offset"] + item["size"])
            for item in allocations
            if item["start"] <= peak_time <= item["end"]
        )
        merged = []
        for start, end in live_ranges:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        base = int(bases.get(mem_id, 0))
        arena_size = int(result.bufsizes[mem_id])
        internal_holes = sum(
            max(0, merged[index + 1][0] - merged[index][1])
            for index in range(len(merged) - 1)
        )
        leading_hole = max(0, merged[0][0] - base) if merged else arena_size - base
        trailing_hole = max(0, arena_size - merged[-1][1]) if merged else 0
        arena_summaries.append(
            {
                "mem_id": mem_id,
                "arena_bytes": arena_size,
                "base_offset": base,
                "allocation_count": len(allocations),
                "peak_live_time": peak_time,
                "peak_live_bytes": arena_peak_bytes,
                "peak_live_count": arena_peak_count,
                "holes_at_peak_count": max(0, len(merged) - 1)
                + int(leading_hole > 0)
                + int(trailing_hole > 0),
                "holes_at_peak_bytes": internal_holes + leading_hole + trailing_hole,
                "internal_holes_at_peak_bytes": internal_holes,
                "leading_hole_at_peak_bytes": leading_hole,
                "trailing_hole_at_peak_bytes": trailing_hole,
                "live_ranges_at_peak_sample": merged[:20],
            }
        )

    plan_bytes = sum(int(value) for value in result.bufsizes[1:])
    lower_bound_bytes = (
        sum(int(value) for value in lower_bound[1:])
        if lower_bound is not None
        else None
    )
    if lower_bound_bytes is not None and lower_bound_bytes > plan_bytes:
        failures.append("sound lower bound exceeds plan")
    return {
        "layout_valid": not failures,
        "layout_failures": sorted(set(failures)),
        "allocation_count": len(specs),
        "primary_interval_count": len(primary),
        "alias_count": sum(spec.storage_base is not None for spec in specs),
        "overlap_pair_count": overlap_pairs,
        "possible_pair_count": possible_pairs,
        "overlap_density": overlap_pairs / possible_pairs if possible_pairs else 0.0,
        "peak_live_allocation_count": peak_live_count,
        "peak_live_bytes": peak_live_bytes,
        "lower_bound_bufsizes": lower_bound,
        "lower_bound_bytes": lower_bound_bytes,
        "plan_bytes": plan_bytes,
        "gap_to_lower_bound_bytes": (
            plan_bytes - lower_bound_bytes if lower_bound_bytes is not None else None
        ),
        "arenas": arena_summaries,
    }


def _build_program(case: Case, policy: str) -> Dict[str, Any]:
    algorithm = (
        core.greedy
        if policy == "upstream_greedy"
        else core.greedy_interval_first_fit_conditional
    )
    calls = []

    @wraps(algorithm)
    def measured(alignment, specs, graph_module, graph_signature, extra_padding=0):
        started = time.perf_counter()
        result = algorithm(
            alignment,
            specs,
            graph_module,
            graph_signature,
            extra_padding,
        )
        calls.append(
            {
                "alignment": alignment,
                "specs": specs,
                "graph_module": graph_module,
                "graph_signature": graph_signature,
                "extra_padding": extra_padding,
                "actual_time_s": time.perf_counter() - started,
                "actual_result": result,
            }
        )
        return result

    planner = core.MemoryPlanningPass(
        memory_planning_algo=core.MemoryPlanningAlgorithmSuite(
            algo_list=[measured]
        ),
        alloc_graph_input=False,
        alloc_graph_output=False,
        alloc_mutable_buffers=True,
        share_mutable_buffers=False,
    )
    edge_started = time.perf_counter()
    edge = core.export_to_edge(
        case.model,
        case.example_args,
        example_kwarg_inputs=case.example_kwargs or None,
        dynamic_shapes=case.dynamic_shapes,
        verbose=False,
    )
    edge_export_time = time.perf_counter() - edge_started
    lowering_started = time.perf_counter()
    program = edge.to_executorch(
        config=core.ExecutorchBackendConfig(
            memory_planning_pass=planner,
            enable_non_cpu_memory_planning=True,
            run_reinplace_pass=False,
        )
    )
    lowering_time = time.perf_counter() - lowering_started
    serialization_started = time.perf_counter()
    buffer = program.buffer
    serialization_time = time.perf_counter() - serialization_started
    max_bytes, method_bytes, method_buffer_sizes = core._format_plan_metrics(program)

    units = []
    for call in calls:
        lower_bound = core._peak_live_lower_bound_bufsizes(
            call["alignment"],
            call["specs"],
            call["graph_module"],
            call["extra_padding"],
        )
        expected = _canonical(call["actual_result"], call["specs"])
        repeat_times = []
        deterministic = True
        repeat_result = call["actual_result"]
        for _ in range(REPEATS):
            started = time.perf_counter()
            repeat_result = algorithm(
                call["alignment"],
                call["specs"],
                call["graph_module"],
                call["graph_signature"],
                call["extra_padding"],
            )
            repeat_times.append(time.perf_counter() - started)
            deterministic = deterministic and (
                _canonical(repeat_result, call["specs"]) == expected
            )
        analysis = _analyze_plan(
            call["alignment"],
            call["specs"],
            call["graph_module"],
            call["actual_result"],
            lower_bound,
        )
        independent_greedy = core.greedy(
            call["alignment"],
            call["specs"],
            call["graph_module"],
            call["graph_signature"],
            call["extra_padding"],
        )
        independent_greedy_bytes = sum(independent_greedy.bufsizes[1:])
        if analysis["alias_count"]:
            selected = "whole_unit_greedy_alias_fallback"
            fallback_exact = _canonical(independent_greedy, call["specs"]) == expected
        elif analysis["lower_bound_bufsizes"] == independent_greedy.bufsizes:
            selected = "greedy_lower_bound_early_exit"
            fallback_exact = None
        elif analysis["plan_bytes"] < independent_greedy_bytes:
            selected = "interval_first_fit"
            fallback_exact = None
        else:
            selected = "greedy_tie_or_better"
            fallback_exact = None
        units.append(
            {
                **analysis,
                "actual_planner_time_s": call["actual_time_s"],
                "repeat_count": REPEATS,
                "repeat_times_s": repeat_times,
                "planner_median_s": statistics.median(repeat_times),
                "planner_p90_s": _percentile(repeat_times, 0.9),
                "planner_min_s": min(repeat_times),
                "planner_max_s": max(repeat_times),
                "deterministic_placements": deterministic,
                "selected_policy": selected,
                "independent_greedy_bytes": independent_greedy_bytes,
                "alias_fallback_exact": fallback_exact,
            }
        )

    load_started = time.perf_counter()
    module = core._load_for_executorch_from_buffer(buffer)
    load_time = time.perf_counter() - load_started
    return {
        "status": "planned",
        "policy": policy,
        "edge_export_time_s": edge_export_time,
        "lowering_time_s": lowering_time,
        "serialization_time_s": serialization_time,
        "total_pipeline_time_s": edge_export_time + lowering_time + serialization_time,
        "program_load_time_s": load_time,
        "program_bytes": len(buffer),
        "max_activation_bytes": max_bytes,
        "method_bytes": method_bytes,
        "method_buffer_sizes": method_buffer_sizes,
        "planning_units": units,
        "planner_median_s": sum(unit["planner_median_s"] for unit in units),
        "planner_p90_s": sum(unit["planner_p90_s"] for unit in units),
        "allocation_count": sum(unit["allocation_count"] for unit in units),
        "primary_interval_count": sum(
            unit["primary_interval_count"] for unit in units
        ),
        "overlap_pair_count": sum(unit["overlap_pair_count"] for unit in units),
        "possible_pair_count": sum(unit["possible_pair_count"] for unit in units),
        "overlap_density": (
            sum(unit["overlap_pair_count"] for unit in units)
            / sum(unit["possible_pair_count"] for unit in units)
            if sum(unit["possible_pair_count"] for unit in units)
            else 0.0
        ),
        "lower_bound_bytes": max(
            (unit["lower_bound_bytes"] for unit in units if unit["lower_bound_bytes"] is not None),
            default=None,
        ),
        "layout_valid": all(unit["layout_valid"] for unit in units),
        "deterministic_placements": all(
            unit["deterministic_placements"] for unit in units
        ),
        "_program": program,
        "_module": module,
    }


def _input_description(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List[Any]:
    leaves, _ = tree_flatten((args, kwargs))
    result = []
    for value in leaves:
        if torch.is_tensor(value):
            result.append(
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            )
        else:
            result.append({"type": type(value).__name__, "value": repr(value)[:200]})
    return result


def _runtime_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List[Any]:
    leaves, _ = tree_flatten((args, kwargs))
    return [value.clone() if torch.is_tensor(value) else value for value in leaves]


def _public(result: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def run_case(name: str) -> Dict[str, Any]:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    started = time.perf_counter()
    artifact: Dict[str, Any] = {
        "schema_version": 1,
        "case": name,
        "seed": seed,
        "base_commit": BASE_COMMIT,
        "candidate_patch_sha256": PATCH_SHA256,
        "planner_repeats": REPEATS,
    }
    try:
        case = build_case(name)
    except Exception as exc:
        artifact.update(
            status="model_construction_failed",
            error=f"{type(exc).__name__}: {exc}",
            wall_time_s=time.perf_counter() - started,
        )
        return artifact

    artifact.update(
        category=case.category,
        family=case.family,
        weights=case.weights,
        dynamic_shapes=case.dynamic_shapes is not None,
        example_inputs=_input_description(case.example_args, case.example_kwargs),
    )
    try:
        with torch.no_grad():
            eager_example = case.model(
                *_clone(case.example_args), **_clone(case.example_kwargs)
            )
        artifact["eager_output_finite"] = _finite(eager_example)
    except Exception as exc:
        artifact.update(
            status="eager_failed",
            error=f"{type(exc).__name__}: {exc}",
            wall_time_s=time.perf_counter() - started,
        )
        return artifact

    built = {}
    for policy in ("upstream_greedy", "conditional_two_strategy"):
        try:
            built[policy] = _build_program(case, policy)
            artifact[policy] = _public(built[policy])
        except Exception as exc:
            artifact[policy] = {
                "status": "export_or_lowering_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    if any(policy not in built for policy in ("upstream_greedy", "conditional_two_strategy")):
        artifact.update(
            status="export_or_lowering_failed",
            wall_time_s=time.perf_counter() - started,
        )
        return artifact

    baseline = built["upstream_greedy"]
    candidate = built["conditional_two_strategy"]
    artifact["baseline_bytes"] = baseline["max_activation_bytes"]
    artifact["candidate_bytes"] = candidate["max_activation_bytes"]
    artifact["change_bytes"] = candidate["max_activation_bytes"] - baseline["max_activation_bytes"]
    artifact["gain_fraction"] = (
        (baseline["max_activation_bytes"] - candidate["max_activation_bytes"])
        / baseline["max_activation_bytes"]
        if baseline["max_activation_bytes"]
        else 0.0
    )
    artifact["candidate_no_larger"] = (
        candidate["max_activation_bytes"] <= baseline["max_activation_bytes"]
    )
    artifact["lower_bound_bytes"] = candidate["lower_bound_bytes"]
    artifact["upstream_excess_bytes"] = (
        baseline["max_activation_bytes"] - candidate["lower_bound_bytes"]
        if candidate["lower_bound_bytes"] is not None
        else None
    )
    artifact["candidate_excess_bytes"] = (
        candidate["max_activation_bytes"] - candidate["lower_bound_bytes"]
        if candidate["lower_bound_bytes"] is not None
        else None
    )
    artifact["added_planner_ms"] = 1000.0 * (
        candidate["planner_median_s"] - baseline["planner_median_s"]
    )
    artifact["planner_ratio"] = (
        candidate["planner_median_s"] / baseline["planner_median_s"]
        if baseline["planner_median_s"]
        else None
    )
    artifact["candidate_planner_fraction_of_pipeline"] = (
        candidate["planner_median_s"] / candidate["total_pipeline_time_s"]
        if candidate["total_pipeline_time_s"]
        else None
    )

    replays = case.replay_inputs or [(case.example_args, case.example_kwargs)]
    replay_results = []
    for args, kwargs in replays:
        record: Dict[str, Any] = {"inputs": _input_description(args, kwargs)}
        try:
            with torch.no_grad():
                eager = case.model(*_clone(args), **_clone(kwargs))
            baseline_output = baseline["_module"].run_method(
                "forward", _runtime_inputs(args, kwargs)
            )
            candidate_output = candidate["_module"].run_method(
                "forward", _runtime_inputs(args, kwargs)
            )
            candidate_repeat = candidate["_module"].run_method(
                "forward", _runtime_inputs(args, kwargs)
            )
            record.update(
                eager_finite=_finite(eager),
                baseline_finite=_finite(baseline_output),
                candidate_finite=_finite(candidate_output),
                baseline_vs_candidate=core._compare_outputs(
                    baseline_output, candidate_output
                ),
                baseline_vs_eager=core._compare_outputs(eager, baseline_output),
                candidate_vs_eager=core._compare_outputs(eager, candidate_output),
                candidate_repeat=core._compare_outputs(
                    candidate_output, candidate_repeat
                ),
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        replay_results.append(record)
    artifact["replays"] = replay_results

    direct_exact = all(
        "error" not in replay
        and replay["baseline_vs_candidate"]["exact"]
        and replay["candidate_repeat"]["exact"]
        for replay in replay_results
    )
    eager_close = all(
        "error" not in replay
        and replay["eager_finite"]
        and replay["baseline_finite"]
        and replay["candidate_finite"]
        and replay["baseline_vs_eager"]["close"]
        and replay["candidate_vs_eager"]["close"]
        for replay in replay_results
    )
    layout_valid = baseline["layout_valid"] and candidate["layout_valid"]
    deterministic = (
        baseline["deterministic_placements"]
        and candidate["deterministic_placements"]
    )
    if not artifact["candidate_no_larger"] or not layout_valid or not deterministic:
        status = "planner_correctness_failure"
    elif not direct_exact:
        status = "runtime_failure_or_mismatch"
    elif not eager_close or not artifact["eager_output_finite"]:
        status = "nonconfirmatory_output"
    else:
        status = "ok"
    artifact.update(
        status=status,
        byte_valid=layout_valid and artifact["candidate_no_larger"],
        runtime_direct_exact=direct_exact,
        runtime_eager_close=eager_close,
        deterministic_placements=deterministic,
        wall_time_s=time.perf_counter() - started,
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_case(args.case)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: artifact.get(key)
        for key in (
            "case", "category", "family", "status", "byte_valid",
            "baseline_bytes", "candidate_bytes", "gain_fraction",
            "added_planner_ms", "planner_ratio", "wall_time_s", "error"
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()

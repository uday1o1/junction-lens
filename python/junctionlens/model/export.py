"""ONNX opset-18 export and structural validation for the M0 graph model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import onnx
import torch

from junctionlens.model.contract import (
    TensorContract,
    contract_sha256,
    input_contract,
    output_contract,
)
from junctionlens.model.profile import M0ModelProfile
from junctionlens.model.spike import INPUT_NAMES, OUTPUT_NAMES, M0GraphModel
from junctionlens.model.synthetic import make_micro_inputs


class ModelExportError(RuntimeError):
    """Raised when a checkpoint or exported graph violates the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(profile: M0ModelProfile, checkpoint_sha256: str) -> dict[str, str]:
    return {
        "junctionlens.checkpoint_sha256": checkpoint_sha256,
        "junctionlens.input_contract_sha256": contract_sha256(input_contract(profile)),
        "junctionlens.opset": str(profile.export.opset),
        "junctionlens.output_contract_sha256": contract_sha256(output_contract(profile)),
        "junctionlens.precision": profile.export.precision,
        "junctionlens.profile_id": profile.profile_id,
        "junctionlens.profile_sha256": profile.canonical_sha256(),
        "junctionlens.schema_version": profile.schema_version,
    }


def _strip_graph_debug_metadata(graph: onnx.GraphProto) -> None:
    graph.doc_string = ""
    if hasattr(graph, "metadata_props"):
        graph.ClearField("metadata_props")
    for value_info in (*graph.input, *graph.output, *graph.value_info):
        value_info.doc_string = ""
        if hasattr(value_info, "metadata_props"):
            value_info.ClearField("metadata_props")
    for initializer in graph.initializer:
        initializer.doc_string = ""
        if hasattr(initializer, "metadata_props"):
            initializer.ClearField("metadata_props")
    for node in graph.node:
        node.doc_string = ""
        if hasattr(node, "metadata_props"):
            node.ClearField("metadata_props")
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                _strip_graph_debug_metadata(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested_graph in attribute.graphs:
                    _strip_graph_debug_metadata(nested_graph)


def _strip_exporter_debug_metadata(model: onnx.ModelProto) -> None:
    """Remove source paths and debug annotations that vary by checkout location."""
    model.doc_string = ""
    _strip_graph_debug_metadata(model.graph)
    for function in model.functions:
        function.doc_string = ""
        if hasattr(function, "metadata_props"):
            function.ClearField("metadata_props")
        for node in function.node:
            node.doc_string = ""
            if hasattr(node, "metadata_props"):
                node.ClearField("metadata_props")


def _load_state(checkpoint_path: Path, profile: M0ModelProfile) -> dict[str, Any]:
    checkpoint = cast(
        dict[str, Any],
        torch.load(checkpoint_path, map_location="cpu", weights_only=True),
    )
    if checkpoint.get("profile_sha256") != profile.canonical_sha256():
        raise ModelExportError("checkpoint profile hash differs from the requested export profile")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ModelExportError("checkpoint does not contain a model_state_dict mapping")
    return cast(dict[str, Any], state)


def _onnx_dimensions(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    dimensions: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("")
    return tuple(dimensions)


def _validate_value_info(
    actual: list[onnx.ValueInfoProto], expected: tuple[TensorContract, ...]
) -> None:
    if len(actual) != len(expected):
        raise ModelExportError(
            f"graph has {len(actual)} tensors where {len(expected)} were required"
        )
    for value_info, contract in zip(actual, expected, strict=True):
        if value_info.name != contract.name:
            raise ModelExportError(
                f"tensor name mismatch: expected {contract.name}, observed {value_info.name}"
            )
        observed_type = value_info.type.tensor_type.elem_type
        required_type = (
            onnx.TensorProto.BOOL if contract.element_type == "bool" else onnx.TensorProto.FLOAT
        )
        if observed_type != required_type:
            raise ModelExportError(f"tensor {contract.name} has the wrong element type")
        dimensions = _onnx_dimensions(value_info)
        if len(dimensions) != len(contract.shape):
            raise ModelExportError(f"tensor {contract.name} has the wrong rank")
        for index, (observed, required) in enumerate(zip(dimensions, contract.shape, strict=True)):
            if required == "batch":
                if not isinstance(observed, str) or not observed:
                    raise ModelExportError(
                        f"tensor {contract.name} dimension {index} is not dynamic"
                    )
            elif observed != required:
                raise ModelExportError(
                    f"tensor {contract.name} dimension {index} is {observed}, expected {required}"
                )


def _restore_dynamic_output_annotations(model: onnx.ModelProto, profile: M0ModelProfile) -> None:
    """Correct a PyTorch 2.8 annotation bug after proving batch-two execution."""
    expected = output_contract(profile)
    for value_info, contract in zip(model.graph.output, expected, strict=True):
        if contract.shape[0] != "batch":
            continue
        dimension = value_info.type.tensor_type.shape.dim[0]
        dimension.ClearField("dim_value")
        dimension.dim_param = "batch"


def validate_exported_model(model_path: Path, profile: M0ModelProfile) -> onnx.ModelProto:
    """Run checker, type/shape inference, metadata, and tensor-contract checks."""
    model = onnx.load_model(model_path, load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, check_type=True, strict_mode=True, data_prop=True
    )
    _strip_exporter_debug_metadata(inferred)
    _restore_dynamic_output_annotations(inferred, profile)
    onnx.checker.check_model(inferred, full_check=True)
    default_opsets = [
        item.version for item in inferred.opset_import if item.domain in {"", "ai.onnx"}
    ]
    if default_opsets != [profile.export.opset]:
        raise ModelExportError(
            f"default ONNX opset is {default_opsets}, expected {[profile.export.opset]}"
        )
    _validate_value_info(list(inferred.graph.input), input_contract(profile))
    _validate_value_info(list(inferred.graph.output), output_contract(profile))
    actual_metadata = _metadata_dict(inferred)
    checkpoint_sha256 = actual_metadata.get("junctionlens.checkpoint_sha256")
    if checkpoint_sha256 is None:
        raise ModelExportError("model metadata junctionlens.checkpoint_sha256 is absent")
    expected_metadata = _metadata(profile, checkpoint_sha256)
    for key, value in expected_metadata.items():
        if actual_metadata.get(key) != value:
            raise ModelExportError(f"model metadata {key} is absent or incorrect")
    return inferred


def _metadata_dict(model: onnx.ModelProto) -> dict[str, str]:
    return {item.key: item.value for item in model.metadata_props}


def export_model(
    profile: M0ModelProfile,
    checkpoint_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Export a checked single-file ONNX model with a dynamic batch dimension."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = _sha256(checkpoint_path)
    model = M0GraphModel(profile)
    model.load_state_dict(_load_state(checkpoint_path, profile), strict=True)
    model.eval()
    inputs = make_micro_inputs(
        profile,
        # PyTorch 2.11 specializes a named dimension observed only at its lower
        # bound.  A batch-two witness keeps the declared batch dimension symbolic.
        torch.tensor([0, 1], dtype=torch.int64),
        spatial_size=(profile.input.height, profile.input.width),
    )
    batch_dimension = torch.export.Dim("batch", min=1)
    dynamic_shapes = {name: {0: batch_dimension} for name in INPUT_NAMES}
    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            model_path,
            dynamo=True,
            external_data=False,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=profile.export.opset,
            dynamic_shapes=dynamic_shapes,
            verify=False,
        )
    exported = onnx.load_model(model_path, load_external_data=True)
    _strip_exporter_debug_metadata(exported)
    onnx.helper.set_model_props(exported, _metadata(profile, checkpoint_sha256))
    onnx.save_model(exported, model_path, save_as_external_data=False)
    inferred = validate_exported_model(model_path, profile)
    onnx.save_model(inferred, model_path, save_as_external_data=False)
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PASSED",
        "profile_id": profile.profile_id,
        "profile_sha256": profile.canonical_sha256(),
        "checkpoint_sha256": checkpoint_sha256,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "opset": profile.export.opset,
        "onnx_checker": "PASSED",
        "onnx_shape_inference": "PASSED",
        "input_contract_sha256": contract_sha256(input_contract(profile)),
        "output_contract_sha256": contract_sha256(output_contract(profile)),
    }
    report_path = model_path.with_suffix(".export.json")
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report

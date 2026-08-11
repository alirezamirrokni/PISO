from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .config import validate_algorithm_params


@dataclass(frozen=True)
class VariantSpec:
    suite: str
    variant_id: str
    label: str
    params: dict[str, Any]
    metadata: dict[str, Any]


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _safe_id(text: str) -> str:
    output: list[str] = []
    for character in text:
        output.append(character.lower() if character.isalnum() else "_")
    return "".join(output).strip("_")


def _parameter_label(parameter: str, value: Any) -> str:
    shown = _format_value(value)
    if parameter == "gamma":
        return rf"$\gamma={shown}$"
    if parameter == "rho":
        return rf"$\rho={shown}$"
    if parameter == "lambda":
        return rf"$\lambda={shown}$"
    if parameter == "cycle_length":
        return f"cycle length={shown}"
    return f"{parameter}={shown}"


def _resolve_params(
    config: dict[str, Any],
    template: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    if template not in config["defaults"]:
        raise ValueError(f"unknown default template: {template}")
    params = deepcopy(config["defaults"][template])
    params.update(deepcopy(overrides))
    validate_algorithm_params(params, f"variant based on {template}")
    return params


def build_suite(config: dict[str, Any], suite_name: str) -> tuple[list[VariantSpec], str]:
    if suite_name not in config["suites"]:
        raise ValueError(f"unknown suite {suite_name!r}")

    suite = config["suites"][suite_name]
    kind = str(suite["kind"])
    variants: list[VariantSpec] = []
    reference_id = ""

    if kind == "grouped_explicit":
        reference_id = str(suite["reference"])
        for panel_index, panel in enumerate(suite["panels"]):
            panel_id = str(panel["id"])
            panel_title = str(panel["title"])
            for variant_index, item in enumerate(panel["variants"]):
                variant_id = str(item["id"])
                template = str(item.get("template", "piso"))
                params = _resolve_params(config, template, item.get("overrides", {}))
                variants.append(
                    VariantSpec(
                        suite=suite_name,
                        variant_id=variant_id,
                        label=str(item.get("label", variant_id)),
                        params=params,
                        metadata={
                            "kind": kind,
                            "template": template,
                            "panel_id": panel_id,
                            "panel_title": panel_title,
                            "panel_index": panel_index,
                            "variant_index": variant_index,
                        },
                    )
                )

    elif kind == "grouped_sweep":
        parameter = str(suite["parameter"])
        values = list(suite["values"])
        reference_value = suite["reference_value"]
        for panel_index, panel in enumerate(suite["panels"]):
            panel_id = str(panel["id"])
            panel_title = str(panel["title"])
            template = str(panel.get("template", "piso"))
            base_overrides = deepcopy(panel.get("overrides", {}))
            for value_index, value in enumerate(values):
                overrides = deepcopy(base_overrides)
                overrides[parameter] = value
                params = _resolve_params(config, template, overrides)
                variant_id = _safe_id(
                    f"{panel_id}__{parameter}_{_format_value(value)}"
                )
                variants.append(
                    VariantSpec(
                        suite=suite_name,
                        variant_id=variant_id,
                        label=_parameter_label(parameter, value),
                        params=params,
                        metadata={
                            "kind": kind,
                            "template": template,
                            "panel_id": panel_id,
                            "panel_title": panel_title,
                            "panel_index": panel_index,
                            "variant_index": value_index,
                            "parameter": parameter,
                            "value": value,
                        },
                    )
                )
                if panel_index == 0 and value == reference_value:
                    reference_id = variant_id
    else:
        raise ValueError(
            f"suite {suite_name}: kind must be grouped_explicit or grouped_sweep"
        )

    ids = [variant.variant_id for variant in variants]
    if len(ids) != len(set(ids)):
        raise ValueError(f"suite {suite_name}: duplicate variant IDs")
    if not variants:
        raise ValueError(f"suite {suite_name}: no variants were generated")
    if reference_id not in set(ids):
        raise ValueError(f"suite {suite_name}: reference variant does not exist")
    return variants, reference_id


def suite_simulations(config: dict[str, Any], suite_name: str) -> int:
    if suite_name not in config["suites"]:
        raise ValueError(f"unknown suite {suite_name!r}")
    return int(config["experiment"]["simulations"])

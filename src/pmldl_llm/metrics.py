from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any, Iterable


METRICS_SCHEMA_VERSION = 1
METRIC_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
METRIC_STATUSES = {"running", "completed", "failed", "aborted"}
PRIMARY_DIRECTIONS = {"minimize", "maximize"}
REQUIRED_METRICS_FIELDS = {
    "schema_version",
    "run_id",
    "experiment_id",
    "status",
    "evaluation_role",
    "seed",
    "primary_metric",
    "summary",
    "history",
}


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_metric_mapping(value: Any, location: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{location} must be an object."]
    for name, metric_value in value.items():
        if not isinstance(name, str) or not METRIC_NAME_PATTERN.fullmatch(name):
            errors.append(
                f"{location} contains invalid metric name {name!r}; "
                "use lowercase snake_case."
            )
        if not is_finite_number(metric_value):
            errors.append(f"{location}.{name} must be a finite number.")
    return errors


def validate_metrics_document(
    document: Any,
    *,
    required_metrics: Iterable[str] = (),
    require_required_metrics: bool = False,
) -> list[str]:
    """Return all violations of the canonical metrics.json contract."""
    if not isinstance(document, dict):
        return ["metrics.json must contain an object."]

    errors: list[str] = []
    missing_fields = sorted(REQUIRED_METRICS_FIELDS.difference(document))
    unknown_fields = sorted(set(document).difference(REQUIRED_METRICS_FIELDS))
    if missing_fields:
        errors.append(
            "metrics.json is missing fields: " + ", ".join(missing_fields) + "."
        )
    if unknown_fields:
        errors.append(
            "metrics.json has unknown fields: " + ", ".join(unknown_fields) + "."
        )
    if missing_fields:
        return errors

    if document["schema_version"] != METRICS_SCHEMA_VERSION:
        errors.append(
            f"metrics.schema_version must equal {METRICS_SCHEMA_VERSION}."
        )
    for name in ("run_id", "experiment_id", "evaluation_role"):
        if not isinstance(document[name], str) or not document[name]:
            errors.append(f"metrics.{name} must be a non-empty string.")
    status = document["status"]
    if not isinstance(status, str) or status not in METRIC_STATUSES:
        errors.append(f"metrics.status has unsupported value {status!r}.")
    seed = document["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        errors.append("metrics.seed must be a non-negative integer.")

    primary = document["primary_metric"]
    if not isinstance(primary, dict):
        errors.append("metrics.primary_metric must be an object.")
    else:
        expected_primary_fields = {"name", "namespace", "direction"}
        if set(primary) != expected_primary_fields:
            errors.append(
                "metrics.primary_metric must contain exactly name, namespace, "
                "and direction."
            )
        else:
            if not isinstance(primary["name"], str) or not METRIC_NAME_PATTERN.fullmatch(
                primary["name"]
            ):
                errors.append("metrics.primary_metric.name must be snake_case.")
            if (
                not isinstance(primary["namespace"], str)
                or not METRIC_NAME_PATTERN.fullmatch(primary["namespace"])
            ):
                errors.append("metrics.primary_metric.namespace must be snake_case.")
            if (
                not isinstance(primary["direction"], str)
                or primary["direction"] not in PRIMARY_DIRECTIONS
            ):
                errors.append(
                    "metrics.primary_metric.direction must be minimize or maximize."
                )

    summary = document["summary"]
    if not isinstance(summary, dict):
        errors.append("metrics.summary must be an object.")
    else:
        for namespace, values in summary.items():
            if (
                not isinstance(namespace, str)
                or not METRIC_NAME_PATTERN.fullmatch(namespace)
            ):
                errors.append(
                    f"metrics.summary has invalid namespace {namespace!r}."
                )
            errors.extend(
                validate_metric_mapping(values, f"metrics.summary.{namespace}")
            )

    history = document["history"]
    if not isinstance(history, list):
        errors.append("metrics.history must be an array.")
    else:
        for index, record in enumerate(history):
            location = f"metrics.history[{index}]"
            if not isinstance(record, dict):
                errors.append(f"{location} must be an object.")
                continue
            if set(record) != {"namespace", "step", "metrics"}:
                errors.append(
                    f"{location} must contain exactly namespace, step, and metrics."
                )
                continue
            namespace = record["namespace"]
            if (
                not isinstance(namespace, str)
                or not METRIC_NAME_PATTERN.fullmatch(namespace)
            ):
                errors.append(f"{location}.namespace must be snake_case.")
            step = record["step"]
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                errors.append(f"{location}.step must be a non-negative integer.")
            errors.extend(
                validate_metric_mapping(record["metrics"], f"{location}.metrics")
            )

    if require_required_metrics and isinstance(summary, dict):
        validation = summary.get("validation")
        if not isinstance(validation, dict):
            errors.append(
                "A completed full run must contain metrics.summary.validation."
            )
        else:
            missing_metrics = [
                name for name in required_metrics if name not in validation
            ]
            if missing_metrics:
                errors.append(
                    "A completed full run is missing validation metrics: "
                    + ", ".join(missing_metrics)
                    + "."
                )

    return errors

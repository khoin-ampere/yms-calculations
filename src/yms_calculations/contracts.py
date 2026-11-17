"""Transport-neutral contracts shared by YMS report calculations.

The objects in this module deliberately contain no FastAPI, Redis, Celery, or
filesystem dependencies.  A report package owns its calculation semantics and
declarative contract; the application owns transport and operations.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import pandas as pd


FilterValueKind = Literal["string", "number", "datetime", "boolean"]


class ReportValidationError(ValueError):
    """Raised when a report request violates its data-owned filter contract."""


@dataclass(frozen=True)
class ArtifactSpec:
    """Logical processed artifact consumed by a report.

    ``logical_name`` is intentionally not a filesystem path.  The backend
    maps it to a mounted Parquet artifact in each deployment environment.
    """

    logical_name: str
    source_kind: str


@dataclass(frozen=True)
class FilterSpec:
    """One supported request filter and its semantic cache relevance."""

    key: str
    value_kind: FilterValueKind
    multiple: bool = False
    calculation_relevant: bool = True
    required_with: tuple[str, ...] = ()
    allowed_values: tuple[object, ...] = ()
    minimum_items: int | None = None
    data_filter: bool = False


@dataclass(frozen=True)
class ColumnSpec:
    """Semantic column metadata rendered by the backend UI adapter."""

    key: str
    header: str
    format: str
    display_by_default: bool = True
    color_type: str | None = None
    color_reference: str | None = None


@dataclass(frozen=True)
class OptionalColumnGroup:
    """One display-only request flag and the fields it projects into a response."""

    key: str
    columns: tuple[ColumnSpec, ...]


@dataclass(frozen=True)
class ReportManifest:
    """Declarative, data-owned contract for one report plugin."""

    name: str
    contract_version: str
    processed_artifact: ArtifactSpec
    commitment_kind: str | None
    filters: tuple[FilterSpec, ...]
    group_by_fields: tuple[str, ...] = ()
    group_by_columns: tuple[ColumnSpec, ...] = ()
    columns: tuple[ColumnSpec, ...] = ()
    optional_columns: tuple[str, ...] = ()
    optional_column_groups: tuple[OptionalColumnGroup, ...] = ()

    @property
    def filter_keys(self) -> frozenset[str]:
        return frozenset(spec.key for spec in self.filters)

    @property
    def calculation_filter_keys(self) -> frozenset[str]:
        return frozenset(spec.key for spec in self.filters if spec.calculation_relevant)

    @property
    def data_filter_keys(self) -> tuple[str, ...]:
        """Return fields that filter individual processed-data rows."""
        return tuple(spec.key for spec in self.filters if spec.data_filter)

    @property
    def group_by_column_map(self) -> dict[str, ColumnSpec]:
        """Return group-by presentation metadata keyed by field name."""
        return {column.key: column for column in self.group_by_columns}

    @property
    def optional_column_group_map(self) -> dict[str, tuple[ColumnSpec, ...]]:
        """Return display-only response fields keyed by their request flag."""
        return {group.key: group.columns for group in self.optional_column_groups}


@dataclass(frozen=True)
class CalculationResult:
    """The semantic result returned by a report calculation.

    The application owns HTTP serialization and UI presentation.  The data
    package returns calculated rows, the effective time window, and canonical
    names of requested dynamic metrics.
    """

    frame: pd.DataFrame
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    dynamic_metric_keys: tuple[str, ...] = ()


def normalize_filters(
    manifest: ReportManifest,
    filters: Mapping[str, object],
) -> dict[str, object]:
    """Validate and normalize generic filter shapes without HTTP dependencies.

    The function intentionally contains no HTTP dependency.  Application
    schemas can use it as their data-owned semantic validation boundary.
    """

    unknown = set(filters).difference(manifest.filter_keys)
    if unknown:
        raise ReportValidationError(
            f"Unsupported filters for {manifest.name}: {sorted(unknown)}"
        )

    normalized: dict[str, object] = {}
    specs_by_key = {spec.key: spec for spec in manifest.filters}
    for key, value in filters.items():
        if value is None:
            normalized[key] = None
            continue
        spec = specs_by_key[key]
        values = _normalize_multiple(spec, value)
        for item in values:
            _validate_value(spec, item)
        if spec.allowed_values:
            unsupported = [item for item in values if item not in spec.allowed_values]
            if unsupported:
                raise ReportValidationError(
                    f"Unsupported {spec.key} for {manifest.name}: {unsupported}"
                )
        if spec.minimum_items is not None and values and len(values) < spec.minimum_items:
            raise ReportValidationError(
                f"{spec.key} requires at least {spec.minimum_items} values for {manifest.name}."
            )
        normalized[key] = values if spec.multiple else values[0]

    for spec in manifest.filters:
        if spec.required_with and normalized.get(spec.key) is not None:
            missing = [key for key in spec.required_with if normalized.get(key) is None]
            if missing:
                raise ReportValidationError(
                    f"{spec.key} requires {sorted(missing)} for {manifest.name}."
                )
    return normalized


def _normalize_multiple(spec: FilterSpec, value: object) -> list[object]:
    if spec.multiple:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ReportValidationError(f"{spec.key} must be a list for {spec.value_kind} values.")
        if not value:
            return []
        return list(value)
    if isinstance(value, (list, tuple)):
        raise ReportValidationError(f"{spec.key} must be a single {spec.value_kind} value.")
    return [value]


def _validate_value(spec: FilterSpec, value: object) -> None:
    if spec.value_kind == "string" and not isinstance(value, str):
        raise ReportValidationError(f"{spec.key} must contain strings.")
    if spec.value_kind == "number" and (
        not isinstance(value, Real) or isinstance(value, bool)
    ):
        raise ReportValidationError(f"{spec.key} must contain numbers.")
    if spec.value_kind == "datetime" and not isinstance(value, (str, pd.Timestamp)):
        raise ReportValidationError(f"{spec.key} must contain ISO datetime strings or pandas timestamps.")
    if spec.value_kind == "boolean" and not isinstance(value, bool):
        raise ReportValidationError(f"{spec.key} must be a boolean.")

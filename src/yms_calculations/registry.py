"""Registry of data-owned report calculations."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType

from . import yms03, yms04, yms05, yms06
from .contracts import ReportManifest, normalize_filters


@dataclass(frozen=True)
class ReportDefinition:
    """Stable, data-owned report plugin consumed by the application adapter.

    The calculation callables remain the current module functions in Phase 1.
    The manifest and normalization methods are additive only; no existing
    backend service calls them until a later migration phase.
    """

    name: str
    module: ModuleType
    manifest: ReportManifest

    def __post_init__(self) -> None:
        if self.manifest.name != self.name:
            raise ValueError(
                f"Report registry name {self.name!r} does not match manifest name "
                f"{self.manifest.name!r}."
            )

    @property
    def transform(self):
        return self.module.process_raw_die_data if hasattr(self.module, "process_raw_die_data") else self.module.process_unit_test_data

    @property
    def calculate(self):
        return self.module.calculate

    @property
    def filter_options(self):
        return getattr(self.module, "filter_options", None)

    def normalize_filters(self, filters: Mapping[str, object]) -> dict[str, object]:
        """Return data-owned normalized filters for future generic services."""
        return normalize_filters(self.manifest, filters)

    def calculation_filter_data(
        self, filters: Mapping[str, object]
    ) -> dict[str, object]:
        """Return only filters that are relevant to a calculation cache key."""
        normalized = self.normalize_filters(filters)
        return {
            key: value
            for key, value in normalized.items()
            if key in self.manifest.calculation_filter_keys
        }


REPORT_REGISTRY = {
    "yms03": ReportDefinition("yms03", yms03, yms03.MANIFEST),
    "yms04": ReportDefinition("yms04", yms04, yms04.MANIFEST),
    "yms05": ReportDefinition("yms05", yms05, yms05.MANIFEST),
    "yms06": ReportDefinition("yms06", yms06, yms06.MANIFEST),
}


def get_report(name: str) -> ReportDefinition:
    """Return a registered report or fail with an actionable configuration error."""
    try:
        return REPORT_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown YMS report: {name}") from error

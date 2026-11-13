"""Registry of data-owned report calculations."""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from . import yms03, yms04, yms05


@dataclass(frozen=True)
class ReportDefinition:
    """Stable entry points a service adapter may use for one report."""

    name: str
    module: ModuleType

    @property
    def transform(self):
        return self.module.process_raw_die_data if hasattr(self.module, "process_raw_die_data") else self.module.process_unit_test_data

    @property
    def calculate(self):
        return self.module.calculate

    @property
    def filter_options(self):
        return getattr(self.module, "filter_options", None)


REPORT_REGISTRY = {
    "yms03": ReportDefinition("yms03", yms03),
    "yms04": ReportDefinition("yms04", yms04),
    "yms05": ReportDefinition("yms05", yms05),
}


def get_report(name: str) -> ReportDefinition:
    """Return a registered report or fail with an actionable configuration error."""
    try:
        return REPORT_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown YMS report: {name}") from error

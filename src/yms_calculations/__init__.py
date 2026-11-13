"""Pure, transport-neutral calculations for YMS reports."""

from .contracts import CalculationResult
from .registry import REPORT_REGISTRY, ReportDefinition, get_report

__all__ = ["CalculationResult", "REPORT_REGISTRY", "ReportDefinition", "get_report"]

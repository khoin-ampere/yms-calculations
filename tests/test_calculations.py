from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from yms_calculations import get_report
from yms_calculations import registry, yms03, yms04, yms05


def test_registry_exposes_supported_reports() -> None:
    assert set(registry.REPORT_REGISTRY) == {"yms03", "yms04", "yms05"}
    assert get_report("yms04").calculate is yms04.calculate
    with pytest.raises(ValueError, match="Unknown YMS report"):
        get_report("unknown")


def test_package_has_no_web_or_runtime_dependencies() -> None:
    forbidden_prefixes = ("fastapi", "redis", "celery", "yms.")
    root = Path(registry.__file__).parent
    for source in root.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = [
            name
            for node in ast.walk(tree)
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
        ]
        assert not any(name.startswith(forbidden_prefixes) for name in imported), source.name


def test_yms04_calculates_semantic_result(die_data, die_commits) -> None:
    processed = yms04.process_raw_die_data(die_data)
    result = yms04.calculate(
        processed,
        die_commits,
        {"start_date": "2026-10-01T00:00:00Z", "end_date": "2026-10-02T00:00:00Z", "hbins": [1]},
    )
    assert result.start_date.strftime("%Y-%m-%dT%H:%M:%SZ") == "2026-10-01T00:00:00Z"
    assert result.dynamic_metric_keys == ("HBIN1",)
    assert result.frame.iloc[0]["HBIN1"] == pytest.approx(0.25)


def test_yms05_assigns_a_lot_to_its_latest_work_week(die_data, die_commits) -> None:
    processed = yms05.process_raw_die_data(die_data)
    assert processed[processed["wafer_lot_id"].eq("LOT1")]["work_week"].nunique() == 1
    result = yms05.calculate(processed, die_commits, {"hbins": [1]})
    assert result.dynamic_metric_keys == ("HBIN1",)


def test_yms03_processes_and_filters_documented_dimension() -> None:
    now = pd.Timestamp.now(tz="UTC")
    raw = pd.DataFrame(
        [
            {"test_program": "PROD", "serial_no": "s1", "lot_end_time": now, "unit_end_time": None, "lot_start_time": None, "unit_start_time": None, "hbin": "1", "sbin": "1", "run_time": "5", "test_stage": "FT", "process_step": "FT1", "rework_index": 1, "lot_id": "LOT-A.1", "product_family": "P1", "facility": "A", "is_latest": 1},
            {"test_program": "PROD", "serial_no": "s2", "lot_end_time": now, "unit_end_time": None, "lot_start_time": None, "unit_start_time": None, "hbin": "4", "sbin": "2", "run_time": "10", "test_stage": "FT", "process_step": "FT2", "rework_index": 2, "lot_id": "LOT-B.1", "product_family": "P1", "facility": "B", "is_latest": 1},
        ]
    )
    commits = pd.DataFrame([{"test_stage": "FT", "product_family": "P1", "process_step": "FT1", "commit_yield": 0.8, "commit_top_bin_yield": 0.8, "commit_test_time": 4.0}])
    result = yms03.calculate(yms03.process_unit_test_data(raw), commits, {"facility": ["A"]})
    assert set(result.frame["process_step"]) == {"FT1"}

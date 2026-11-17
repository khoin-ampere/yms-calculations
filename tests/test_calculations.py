from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from yms_calculations import get_report
from yms_calculations import registry, yms03, yms04, yms05, yms06


def test_registry_exposes_supported_reports() -> None:
    assert set(registry.REPORT_REGISTRY) == {"yms03", "yms04", "yms05", "yms06"}
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

def test_yms03_parses_lot_filters_by_lot_and_uses_latest_week() -> None:
    first = pd.Timestamp("2026-09-28T00:00:00Z")
    latest = pd.Timestamp("2026-10-08T00:00:00Z")
    raw = pd.DataFrame(
        [
            {
                "test_program": "PROD",
                "serial_no": "s1",
                "lot_end_time": first,
                "unit_end_time": None,
                "lot_start_time": None,
                "unit_start_time": None,
                "hbin": "1",
                "sbin": "1",
                "run_time": "5",
                "test_stage": "FT",
                "process_step": "FT1",
                "rework_index": 1,
                "lot_id": "PREFIX_A.1",
                "product_family": "P1",
                "facility_id": "F1",
                "is_latest": 1,
            },
            {
                "test_program": "PROD",
                "serial_no": "s2",
                "lot_end_time": latest,
                "unit_end_time": None,
                "lot_start_time": None,
                "unit_start_time": None,
                "hbin": "1",
                "sbin": "1",
                "run_time": "5",
                "test_stage": "FT",
                "process_step": "FT1",
                "rework_index": 1,
                "lot_id": "PREFIX_A.2",
                "product_family": "P1",
                "facility_id": "F1",
                "is_latest": 1,
            },
            {
                "test_program": "PROD",
                "serial_no": "s3",
                "lot_end_time": latest,
                "unit_end_time": None,
                "lot_start_time": None,
                "unit_start_time": None,
                "hbin": "1",
                "sbin": "1",
                "run_time": "5",
                "test_stage": "FT",
                "process_step": "FT1",
                "rework_index": 1,
                "lot_id": "PREFIX_B.1",
                "product_family": "P1",
                "facility_id": "F1",
                "is_latest": 1,
            },
        ]
    )

    processed = yms03.process_unit_test_data(raw)
    a_rows = processed.loc[processed["lot"].eq("A")]

    assert set(a_rows["facility"]) == {"F1"}
    assert a_rows["work_week"].nunique() == 1

    commits = pd.DataFrame(
        [
            {
                "test_stage": "FT",
                "product_family": "P1",
                "process_step": "FT1",
                "commit_yield": 0.8,
                "commit_top_bin_yield": 0.8,
                "commit_test_time": 4.0,
            }
        ]
    )
    result = yms03.calculate(
        processed,
        commits,
        {"lot": ["A"], "group_by_columns": ["lot"]},
    )

    assert result.frame.loc[0, "lot"] == "A"
    assert result.frame.loc[0, "volume"] == 2


def test_yms06_calculates_test_time_for_filtered_records(die_data, die_commits) -> None:
    processed = yms06.process_raw_die_data(die_data.assign(facility_id="F1", handler_id="H1"))
    result = yms06.calculate(
        processed,
        die_commits,
        {
            "start_date": "2026-10-01T00:00:00Z",
            "end_date": "2026-10-02T00:00:00Z",
            "hbin": [1],
        },
    )
    assert result.start_date.strftime("%Y-%m-%dT%H:%M:%SZ") == "2026-10-01T00:00:00Z"
    assert result.frame.loc[0, "volume"] == 1
    assert result.frame.loc[0, "test_time"] == pytest.approx(10)


def test_yms06_transform_rejects_incomplete_input_and_keeps_latest_lot_week(die_data) -> None:
    raw = die_data.assign(facility_id="F1", handler_id="H1")
    raw.loc[1, "wafer_end_time"] = "2026-10-08T10:00:00Z"
    processed = yms06.process_raw_die_data(raw)

    assert len(processed) == 3
    assert set(processed["facility"]) == {"F1"}
    assert processed["work_week"].nunique() == 1
    with pytest.raises(ValueError, match="missing required columns"):
        yms06.process_raw_die_data(raw.drop(columns="die_id"))


def test_yms06_combines_step_metrics_and_unique_wafer_counts(die_data) -> None:
    data = yms06.process_raw_die_data(die_data.assign(facility_id="F1", handler_id="H1"))
    cp2 = data.iloc[[0]].copy()
    cp2["process_step"] = "CP2"
    combined_input = pd.concat([data, cp2], ignore_index=True)
    commits = pd.DataFrame(
        [
            {"product_family": "P1", "process_step": "CP1", "commit_test_time": 12.0, "good_test_time": 9.0, "trigger_limit_test_time": 18.0},
            {"product_family": "P1", "process_step": "CP2", "commit_test_time": 8.0, "good_test_time": 5.0, "trigger_limit_test_time": 12.0},
        ]
    )

    summary = yms06.create_summary(
        combined_input,
        commits,
        ["product_family", "process_step"],
        process_step_combination=["CP1", "CP2"],
    )
    row = summary.loc[summary["process_step"].eq("CP1, CP2")].iloc[0]

    assert row["volume"] == 4
    assert row["no_wafers"] == 1

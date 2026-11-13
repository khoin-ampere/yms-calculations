"""Pure YMS03 FT/SLT transformations and aggregations."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .contracts import CalculationResult
from .yms04 import add_dynamic_yields as add_generic_dynamic_yields

EXCLUDED_TEST_PROGRAM_PATTERN = r"QK|Screen|ENG|CORR|Calibration|HTOL"
DEFAULT_SUMMARY_KEYS = ("test_stage", "product_family", "process_step")
REQUEST_FILTERS = ("test_program", "facility", "tester_id", "handler_id", "card_id")


def process_unit_test_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    data = dataframe.copy()
    data = data.loc[~data["test_program"].astype("string").str.contains(EXCLUDED_TEST_PROGRAM_PATTERN, case=False, na=False, regex=True)]
    data = data.loc[data["serial_no"].notna() & data["serial_no"].astype("string").str.strip().ne("")].copy()
    for column in ("lot_end_time", "unit_end_time", "lot_start_time", "unit_start_time"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["event_time"] = data["lot_end_time"].fillna(data["unit_end_time"]).fillna(data["lot_start_time"]).fillna(data["unit_start_time"])
    data = data.loc[data["event_time"] >= pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=6)].copy()
    data["hbin"] = pd.to_numeric(data["hbin"], errors="coerce")
    data["sbin"] = pd.to_numeric(data["sbin"], errors="coerce")
    data["run_time"] = pd.to_numeric(data["run_time"], errors="coerce")
    slt_pass = data["test_stage"].eq("SLT") & data["hbin"].le(3)
    ft_pass = data["test_stage"].eq("FT") & (data["hbin"].le(3) | (data["hbin"].eq(4) & data["process_step"].eq("FT1")))
    data["test_result"] = np.where(slt_pass | ft_pass, 1.0, 0.0)
    data["test_result_top_bin"] = np.where(data["test_result"].eq(1) & data["hbin"].eq(1) & data["sbin"].eq(1), 1.0, 0.0)
    data["retest"] = pd.to_numeric(data["rework_index"], errors="coerce").gt(1).astype(int)
    data["lot"] = data["lot_id"].astype("string").str.split(".").str[0]
    data["work_week"] = data["event_time"].dt.strftime("%Y%W")
    return data


def filter_options(dataframe: pd.DataFrame) -> dict[str, object]:
    times = pd.to_datetime(dataframe["event_time"], errors="coerce", utc=True)
    result: dict[str, object] = {
        "start_date": times.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date": times.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hbins": sorted(dataframe["hbin"].dropna().unique().tolist()),
        "sbins": sorted(dataframe["sbin"].dropna().unique().tolist()),
        "group_by_columns": ["work_week", *REQUEST_FILTERS],
    }
    for column in REQUEST_FILTERS:
        if column in dataframe:
            result[column] = sorted(dataframe[column].dropna().unique().tolist())
    return result


def create_summary(filtered_data: pd.DataFrame, commit_data: pd.DataFrame, summary_keys: list[str]) -> pd.DataFrame:
    latest = filtered_data[filtered_data["is_latest"] == 1].copy()
    yield_summary = latest.groupby(summary_keys, dropna=False).agg(
        volume=("serial_no", "nunique"), no_lots=("lot", "nunique"),
        yield_sum=("test_result", "sum"), top_bin_yield_sum=("test_result_top_bin", "sum"),
    ).reset_index()
    time_summary = filtered_data.groupby(summary_keys, dropna=False).agg(
        retest_rate=("retest", "mean"), test_time=("run_time", "mean"),
        ttg=("run_time", lambda values: values[filtered_data.loc[values.index, "test_result"] == 1].mean()),
    ).reset_index()
    summary = yield_summary.merge(time_summary, on=summary_keys, how="left")
    summary = summary.merge(commit_data, on=["test_stage", "product_family", "process_step"], how="left")
    summary = summary.rename(columns={"commit_yield": "yield_commit", "commit_top_bin_yield": "top_bin_yield_commit", "commit_test_time": "test_time_commit"})
    for column in ("yield_commit", "top_bin_yield_commit", "test_time_commit"):
        if column not in summary:
            summary[column] = np.nan
    summary["yield"] = summary["yield_sum"] / summary["volume"]
    summary["top_bin_yield"] = summary["top_bin_yield_sum"] / summary["volume"]
    summary["delta"] = summary["yield"] - summary["yield_commit"]
    summary["top_bin_delta"] = summary["top_bin_yield"] - summary["top_bin_yield_commit"]
    yield_bad = summary["yield"].lt(summary.get("trigger_limit_yield", -np.inf)) | summary["yield"].gt(summary.get("good_yield", np.inf))
    top_bad = summary["top_bin_yield"].lt(summary.get("trigger_limit_top_bin_yield", -np.inf)) | summary["top_bin_yield"].gt(summary.get("good_top_bin_yield", np.inf))
    time_bad = summary["test_time"].lt(summary.get("good_test_time", -np.inf)) | summary["test_time"].gt(summary.get("trigger_limit_test_time", np.inf))
    summary["yield_delta"] = np.where(yield_bad, summary["delta"], 0.0)
    summary["top_bin_yield_delta"] = np.where(top_bad, summary["top_bin_delta"], 0.0)
    summary["test_time_delta"] = np.where(time_bad, summary["test_time_commit"] - summary["test_time"], 0.0)
    return summary.sort_values(summary_keys, na_position="last").reset_index(drop=True)


def calculate(
    dataframe: pd.DataFrame,
    commit_data: pd.DataFrame,
    filters: Mapping[str, object],
    summary_keys: list[str] | None = None,
) -> CalculationResult:
    data = dataframe.copy()
    data["event_time"] = pd.to_datetime(data["event_time"], errors="coerce")
    start_value, end_value = filters.get("start_date"), filters.get("end_date")
    if start_value and end_value:
        start_date, end_date = pd.Timestamp(start_value), pd.Timestamp(end_value)
    else:
        end_date = data["event_time"].max()
        start_date = (end_date - pd.Timedelta(weeks=13)).normalize()
    filtered = data.query("event_time >= @start_date and event_time <= @end_date")
    for column in REQUEST_FILTERS:
        values = filters.get(column)
        if values and column in filtered:
            filtered = filtered[filtered[column].isin(values)]
    keys = list(summary_keys or DEFAULT_SUMMARY_KEYS)
    for column in filters.get("group_by_columns") or []:
        if column in filtered and column not in keys:
            keys.append(column)
    summary = create_summary(filtered, commit_data, keys)
    summary, dynamic_keys = add_generic_dynamic_yields(
        summary, filtered[filtered["is_latest"] == 1], filters, keys,
    )
    return CalculationResult(summary, start_date, end_date, dynamic_keys)

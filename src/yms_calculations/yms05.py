"""Pure YMS05 transformations and work-week aggregations."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .contracts import CalculationResult
from . import yms04

DEFAULT_SUMMARY_KEYS = ("product_family", "process_step", "test_temp", "work_week")


def add_work_week(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Assign each lot the work week of its latest completed wafer."""
    data = dataframe.copy()
    if "work_week" in data.columns:
        return data
    data["wafer_end_time"] = pd.to_datetime(data["wafer_end_time"], errors="coerce")
    data["work_week_raw"] = data["wafer_end_time"].dt.strftime("%Y%W")
    latest_week = data.groupby("wafer_lot_id")["work_week_raw"].max().rename("work_week")
    return data.merge(latest_week, on="wafer_lot_id", how="left")


def process_raw_die_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    return add_work_week(yms04.process_raw_die_data(dataframe))


def create_summary(filtered_data: pd.DataFrame, commit_data: pd.DataFrame) -> pd.DataFrame:
    latest = filtered_data[filtered_data["is_latest"] == 1].copy()
    return yms04.create_final_summary(latest, filtered_data, commit_data, list(DEFAULT_SUMMARY_KEYS))


def calculate(dataframe: pd.DataFrame, commit_data: pd.DataFrame, filters: Mapping[str, object]) -> CalculationResult:
    data = dataframe.copy()
    data["wafer_end_time"] = pd.to_datetime(data["wafer_end_time"], errors="coerce")
    start_value, end_value = filters.get("start_date"), filters.get("end_date")
    if start_value and end_value:
        start_date, end_date = pd.Timestamp(start_value), pd.Timestamp(end_value)
    else:
        end_date = data["wafer_end_time"].max()
        start_date = (end_date - pd.Timedelta(weeks=13)).normalize()
    filtered = data.query("wafer_end_time >= @start_date and wafer_end_time <= @end_date")
    summary = create_summary(filtered, commit_data)
    summary, dynamic_keys = yms04.add_dynamic_yields(
        summary, filtered[filtered["is_latest"] == 1], filters, list(DEFAULT_SUMMARY_KEYS),
    )
    return CalculationResult(summary, start_date, end_date, dynamic_keys)

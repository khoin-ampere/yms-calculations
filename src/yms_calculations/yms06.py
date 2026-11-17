"""Pure YMS06 wafer-sort test-time transformations and aggregations."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .contracts import CalculationResult


EXCLUDED_TEST_PROGRAM_PATTERN = r"QK|Screen|ENG|CORR|Calibration|HTOL"
FILTER_COLUMNS = (
    "test_program",
    "facility",
    "handler_id",
    "test_temp",
    "hbin",
    "lot",
    "process_step",
)
GROUP_BY_COLUMNS = (
    "work_week",
    "test_program",
    "facility",
    "handler_id",
    "test_temp",
    "lot",
)
_REQUIRED_COLUMNS = {
    "die_id",
    "wafer_no",
    "wafer_lot_id",
    "hbin",
    "wafer_end_time",
    "die_run_time",
    "test_program",
    "test_temp",
    "product_family",
    "process_step",
}


def process_raw_die_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean WS test records and derive YMS06 test-time dimensions."""
    missing_columns = _REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {sorted(missing_columns)}")

    excluded_program = dataframe["test_program"].astype("string").str.contains(
        EXCLUDED_TEST_PROGRAM_PATTERN, case=False, na=False, regex=True,
    )
    valid_die_id = dataframe["die_id"].notna() & dataframe["die_id"].astype("string").str.strip().ne("")
    processed = dataframe.loc[~excluded_program & valid_die_id].copy()
    processed.rename(columns={"facility_id": "facility"}, inplace=True)
    processed["lot_wafer"] = (
        processed["wafer_lot_id"].astype("string") + " - " + processed["wafer_no"].astype("string")
    )
    processed["lot"] = processed["wafer_lot_id"].astype("string")
    processed["hbin"] = pd.to_numeric(processed["hbin"], errors="coerce")
    processed["die_run_time"] = pd.to_numeric(processed["die_run_time"], errors="coerce")
    processed["test_temp"] = pd.to_numeric(processed["test_temp"], errors="coerce")
    processed["test_result"] = processed["hbin"].le(6).astype("int64")
    processed["wafer_end_time"] = pd.to_datetime(processed["wafer_end_time"], errors="coerce", utc=True)
    processed["work_week_raw"] = processed["wafer_end_time"].dt.strftime("%Y%W")
    latest_ww_per_lot = processed.groupby("lot")["work_week_raw"].max().rename("work_week")
    return processed.merge(latest_ww_per_lot, on="lot", how="left")


def filter_options(dataframe: pd.DataFrame) -> dict[str, object]:
    """Return the YMS06 filter choices from a processed artifact."""
    timestamps = dataframe["wafer_end_time"].dropna()
    return {
        "start_date": timestamps.min().strftime("%Y-%m-%dT%H:%M:%SZ") if not timestamps.empty else None,
        "end_date": timestamps.max().strftime("%Y-%m-%dT%H:%M:%SZ") if not timestamps.empty else None,
        "hbin": sorted(dataframe["hbin"].dropna().astype(int).unique().tolist()),
        "group_by_columns": list(GROUP_BY_COLUMNS),
        "test_program": sorted(dataframe["test_program"].dropna().unique().tolist()),
        "facility": sorted(dataframe["facility"].dropna().unique().tolist()),
        "handler_id": sorted(dataframe["handler_id"].dropna().unique().tolist()),
        "test_temp": sorted(dataframe["test_temp"].dropna().unique().tolist()),
        "lot": sorted(dataframe["lot"].dropna().unique().tolist()),
        "process_step": sorted(dataframe["process_step"].dropna().unique().tolist()),
        "optional_columns": ["ttg_volume", "ttb_volume", "no_lots"],
    }


def create_summary(
    test_data: pd.DataFrame,
    commit_data: pd.DataFrame,
    summary_keys: list[str],
    process_step_combination: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate WS test records and attach test-time commits and color deltas."""
    data = test_data.copy()
    data["ttg_time"] = data["die_run_time"].where(data["test_result"].eq(1))
    data["ttb_time"] = data["die_run_time"].where(data["test_result"].eq(0))
    summary = (
        data.groupby(summary_keys, dropna=False)
        .agg(
            volume=("die_id", "size"),
            test_time=("die_run_time", "mean"),
            ttg=("ttg_time", "mean"),
            ttg_volume=("test_result", "sum"),
            ttb=("ttb_time", "mean"),
            ttb_volume=("test_result", lambda values: values.eq(0).sum()),
        )
        .reset_index()
    )
    commit_columns = [
        "product_family",
        "process_step",
        "commit_test_time",
        "trigger_limit_test_time",
        "good_test_time",
    ]
    commits = commit_data.loc[:, commit_columns].drop_duplicates(["product_family", "process_step"])
    summary = summary.merge(
        commits,
        on=["product_family", "process_step"],
        how="left",
        validate="many_to_one",
    )
    summary.rename(columns={"commit_test_time": "test_time_commit"}, inplace=True)

    if process_step_combination:
        combined_steps = set(process_step_combination)
        combined_label = ", ".join(process_step_combination)
        summary["group"] = summary["process_step"].where(~summary["process_step"].isin(combined_steps), combined_label)
        combined_keys = ["group" if key == "process_step" else key for key in summary_keys]
        sum_columns = [
            "volume",
            "test_time",
            "ttg",
            "ttg_volume",
            "ttb",
            "ttb_volume",
            "test_time_commit",
            "trigger_limit_test_time",
            "good_test_time",
        ]
        summary = (
            summary.groupby(combined_keys, dropna=False)[sum_columns]
            .agg(lambda values: values.sum(min_count=1))
            .reset_index()
            .rename(columns={"group": "process_step"})
        )
        count_data = data.copy()
        count_data["group"] = count_data["process_step"].where(~count_data["process_step"].isin(combined_steps), combined_label)
        unique_counts = (
            count_data.groupby(combined_keys, dropna=False)
            .agg(no_wafers=("lot_wafer", "nunique"), no_lots=("lot", "nunique"))
            .reset_index()
            .rename(columns={"group": "process_step"})
        )
        summary = summary.merge(unique_counts, on=summary_keys, how="left", validate="one_to_one")
    else:
        unique_counts = (
            data.groupby(summary_keys, dropna=False)
            .agg(no_wafers=("lot_wafer", "nunique"), no_lots=("lot", "nunique"))
            .reset_index()
        )
        summary = summary.merge(unique_counts, on=summary_keys, how="left", validate="one_to_one")

    summary["delta"] = summary["test_time"] - summary["test_time_commit"]
    outside_limits = summary["test_time"].lt(summary["good_test_time"]) | summary["test_time"].gt(summary["trigger_limit_test_time"])
    summary["test_time_delta"] = np.where(outside_limits, summary["test_time_commit"] - summary["test_time"], 0.0)
    summary["test_time_delta"] = summary["test_time_delta"].where(summary["test_time_commit"].notna())
    return summary.sort_values(summary_keys, na_position="last").reset_index(drop=True)


def calculate(
    dataframe: pd.DataFrame,
    commit_data: pd.DataFrame,
    filters: Mapping[str, object],
) -> CalculationResult:
    """Calculate a filtered YMS06 summary without application I/O."""
    data = dataframe.copy()
    data["wafer_end_time"] = pd.to_datetime(data["wafer_end_time"], errors="coerce", utc=True)
    start_value, end_value = filters.get("start_date"), filters.get("end_date")
    if start_value is not None:
        start_date = pd.to_datetime(start_value, utc=True)
        end_date = pd.to_datetime(end_value, utc=True)
    else:
        end_date = data["wafer_end_time"].max()
        start_date = (end_date - pd.Timedelta(weeks=13)).normalize()
    filtered = data.loc[data["wafer_end_time"].between(start_date, end_date, inclusive="both")].copy()
    for column in FILTER_COLUMNS:
        values = filters.get(column)
        if values:
            filtered = filtered.loc[filtered[column].isin(values)].copy()
    summary_keys = ["product_family", "process_step", *(filters.get("group_by_columns") or [])]
    summary = create_summary(
        filtered,
        commit_data,
        summary_keys,
        process_step_combination=filters.get("process_step_combination"),
    )
    return CalculationResult(summary, start_date, end_date)

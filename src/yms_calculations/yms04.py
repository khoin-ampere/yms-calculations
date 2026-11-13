"""Pure YMS04 wafer-sort transformations and aggregations."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .contracts import CalculationResult

EXCLUDED_TEST_PROGRAM_PATTERN = r"QK|Screen|ENG|CORR|Calibration|HTOL"
DEFAULT_SUMMARY_KEYS = ("product_family", "process_step", "test_temp")
SUMMARY_REQUIRED_COLUMNS = {
    "die_id", "lot_wafer", "wafer_lot_id", "test_result", "test_result_full_core",
    "test_result_first_pass", "product_family", "process_step", "test_temp", "wafer_type",
}
ALL_TEST_REQUIRED_COLUMNS = {
    "product_family", "process_step", "test_temp", "retest", "die_run_time", "test_result",
}


def process_raw_die_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean raw die-test data and derive calculation-ready columns."""
    required_columns = {
        "die_id", "wafer_no", "wafer_lot_id", "wafer_type", "hbin",
        "rework_index", "test_program", "product_family", "process_step",
        "test_temp", "die_run_time",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {sorted(missing_columns)}")

    excluded_program = dataframe["test_program"].astype("string").str.contains(
        EXCLUDED_TEST_PROGRAM_PATTERN, case=False, na=False, regex=True,
    )
    valid_die_id = dataframe["die_id"].notna() & dataframe["die_id"].astype("string").str.strip().ne("")
    processed = dataframe.loc[~excluded_program & valid_die_id].copy()
    processed["lot_wafer"] = (
        processed["wafer_lot_id"].astype("string") + " - " + processed["wafer_no"].astype("string")
    )
    hbin = pd.to_numeric(processed["hbin"], errors="coerce")
    processed["test_result"] = hbin.le(6).astype("float64")
    processed["test_result_full_core"] = hbin.eq(1).astype("float64")
    rework_index = pd.to_numeric(processed["rework_index"], errors="coerce")
    processed["test_result_first_pass"] = (rework_index.eq(1) & processed["test_result"].eq(1.0)).astype("float64")
    processed["retest"] = rework_index.gt(1)
    processed["die_run_time"] = pd.to_numeric(processed["die_run_time"], errors="coerce")
    return processed


def filter_options(dataframe: pd.DataFrame) -> dict[str, object]:
    """Return semantic filter choices for a processed YMS04 artifact."""
    times = pd.to_datetime(dataframe["wafer_end_time"], errors="coerce", utc=True)
    return {
        "start_date": times.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date": times.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hbins": sorted(dataframe["hbin"].dropna().unique().tolist()),
        "sbins": sorted(dataframe["sbin"].dropna().unique().tolist()),
    }


def create_latest_test_summary(dataframe: pd.DataFrame, summary_keys: list[str]) -> pd.DataFrame:
    missing_columns = SUMMARY_REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing summary columns: {sorted(missing_columns)}")
    return (
        dataframe.groupby(summary_keys, dropna=False)
        .agg(
            volume_initial=("die_id", "nunique"),
            no_wafers=("lot_wafer", "nunique"),
            no_lots=("wafer_lot_id", "nunique"),
            yield_sum=("test_result", "sum"),
            full_core_yield_sum=("test_result_full_core", "sum"),
            first_pass_yield_sum=("test_result_first_pass", "sum"),
            has_ccl=("wafer_type", lambda values: "CCL" in values.unique()),
        )
        .reset_index()
    )


def create_all_test_metrics(dataframe: pd.DataFrame, summary_keys: list[str]) -> pd.DataFrame:
    missing_columns = ALL_TEST_REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing all-test metric columns: {sorted(missing_columns)}")
    return (
        dataframe.groupby(summary_keys, dropna=False)
        .agg(
            retest_rate=("retest", "mean"),
            test_time=("die_run_time", "mean"),
            ttg=("die_run_time", lambda values: values[dataframe.loc[values.index, "test_result"] == 1].mean()),
        )
        .reset_index()
    )


def create_final_summary(
    latest_test_data: pd.DataFrame,
    all_test_data: pd.DataFrame,
    commit_data: pd.DataFrame,
    summary_keys: list[str],
) -> pd.DataFrame:
    if commit_data.duplicated(["product_family", "process_step"]).any():
        raise ValueError("Commit data has duplicate product_family/process_step keys.")
    yield_summary = create_latest_test_summary(latest_test_data, summary_keys)
    all_test_metrics = create_all_test_metrics(all_test_data, summary_keys)
    summary = yield_summary.merge(all_test_metrics, on=summary_keys, how="left", validate="one_to_one")
    summary = summary.merge(
        commit_data,
        on=["product_family", "process_step"],
        how="left",
        validate="many_to_one",
    )
    is_cp1 = summary["process_step"].astype("string").eq("CP1")
    summary["volume"] = np.where(is_cp1, summary["no_wafers"] * summary["no_dies_per_wafer"], summary["volume_initial"])
    summary["yield"] = summary["yield_sum"] / summary["volume"]
    summary["full_core_yield"] = summary["full_core_yield_sum"] / summary["volume"]
    summary["first_pass_yield"] = summary["first_pass_yield_sum"] / summary["volume"]
    summary["yield_commit"] = summary["commit_yield"]
    summary["test_time_commit"] = summary["commit_test_time"]
    summary["delta"] = summary["yield"] - summary["yield_commit"]
    valid_d0 = summary["full_core_yield"].gt(0) & summary["n"].gt(0) & summary["area"].gt(0)
    summary["d0"] = np.nan
    summary.loc[valid_d0, "d0"] = (
        (1 / summary.loc[valid_d0, "full_core_yield"].pow(1 / summary.loc[valid_d0, "n"]) - 1)
        / summary.loc[valid_d0, "area"]
    )
    summary.loc[~summary["has_ccl"], "full_core_yield"] = np.nan
    yield_outside_limits = summary["yield"].lt(summary["trigger_limit_yield"]) | summary["yield"].gt(summary["good_yield"])
    summary["yield_delta"] = np.where(yield_outside_limits, summary["yield"] - summary["commit_yield"], 0.0)
    full_core_outside_limits = summary["full_core_yield"].lt(summary["trigger_limit_fc_yield"]) | summary["full_core_yield"].gt(summary["good_fc_yield"])
    summary["fc_yield_delta"] = np.where(full_core_outside_limits, summary["full_core_yield"] - summary["commit_fc_yield"], 0.0)
    test_time_outside_limits = summary["test_time"].lt(summary["good_test_time"]) | summary["test_time"].gt(summary["trigger_limit_test_time"])
    summary["test_time_delta"] = np.where(test_time_outside_limits, summary["commit_test_time"] - summary["test_time"], 0.0)
    return summary.sort_values(summary_keys, na_position="last").reset_index(drop=True)


def add_dynamic_yields(
    summary_df: pd.DataFrame,
    latest_test_data: pd.DataFrame,
    filters: Mapping[str, object],
    summary_keys: list[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    final_summary = summary_df.copy()
    dynamic_keys: list[str] = []

    def calculate_yields_for_category(category: str, values: object) -> None:
        nonlocal final_summary
        if not values:
            return
        pivot = pd.pivot_table(
            latest_test_data[latest_test_data[category].isin(values)],
            index=summary_keys,
            columns=category,
            aggfunc="size",
            fill_value=0,
        )
        pivot.columns = [f"{category.upper()}{column}" for column in pivot.columns]
        final_summary = final_summary.merge(pivot, on=summary_keys, how="left")
        for column in pivot.columns:
            final_summary[column] = final_summary[column] / final_summary["volume"]
            dynamic_keys.append(column)

    calculate_yields_for_category("hbin", filters.get("hbins"))
    calculate_yields_for_category("sbin", filters.get("sbins"))
    return final_summary, tuple(dynamic_keys)


def calculate(
    dataframe: pd.DataFrame,
    commit_data: pd.DataFrame,
    filters: Mapping[str, object],
    summary_keys: list[str] | None = None,
) -> CalculationResult:
    """Calculate a filtered YMS04 semantic summary without application I/O."""
    summary_keys = summary_keys or list(DEFAULT_SUMMARY_KEYS)
    data = dataframe.copy()
    data["wafer_end_time"] = pd.to_datetime(data["wafer_end_time"], errors="coerce")
    start_value, end_value = filters.get("start_date"), filters.get("end_date")
    if start_value and end_value:
        start_date, end_date = pd.Timestamp(start_value), pd.Timestamp(end_value)
    else:
        end_date = data["wafer_end_time"].max()
        start_date = (end_date - pd.Timedelta(weeks=13)).normalize()
    filtered = data.query("wafer_end_time >= @start_date and wafer_end_time <= @end_date")
    latest = filtered[filtered["is_latest"] == 1].copy()
    summary = create_final_summary(latest, filtered, commit_data, summary_keys)
    summary, dynamic_keys = add_dynamic_yields(summary, latest, filters, summary_keys)
    return CalculationResult(summary, start_date, end_date, dynamic_keys)

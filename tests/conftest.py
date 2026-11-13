from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def die_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"die_id": "d1", "wafer_no": "01", "wafer_lot_id": "LOT1", "wafer_type": "CCL", "hbin": 1, "sbin": 1, "rework_index": 1, "test_program": "PROD", "product_family": "P1", "process_step": "CP1", "test_temp": 25, "die_run_time": 10, "wafer_end_time": "2026-10-01T10:00:00Z", "is_latest": 1},
            {"die_id": "d2", "wafer_no": "01", "wafer_lot_id": "LOT1", "wafer_type": "CCL", "hbin": 7, "sbin": 9, "rework_index": 2, "test_program": "PROD", "product_family": "P1", "process_step": "CP1", "test_temp": 25, "die_run_time": 20, "wafer_end_time": "2026-10-01T10:00:00Z", "is_latest": 1},
            {"die_id": "d2", "wafer_no": "01", "wafer_lot_id": "LOT1", "wafer_type": "CCL", "hbin": 2, "sbin": 2, "rework_index": 1, "test_program": "PROD", "product_family": "P1", "process_step": "CP1", "test_temp": 25, "die_run_time": 30, "wafer_end_time": "2026-10-01T10:00:00Z", "is_latest": 0},
            {"die_id": "discard", "wafer_no": "02", "wafer_lot_id": "LOT2", "wafer_type": "STD", "hbin": 1, "sbin": 1, "rework_index": 1, "test_program": "ENG debug", "product_family": "P2", "process_step": "FT1", "test_temp": 85, "die_run_time": 8, "wafer_end_time": "2026-10-02T10:00:00Z", "is_latest": 1},
        ]
    )


@pytest.fixture
def die_commits() -> pd.DataFrame:
    return pd.DataFrame(
        [{"product_family": "P1", "process_step": "CP1", "commit_yield": 0.8, "commit_fc_yield": 0.75, "commit_test_time": 15.0, "no_dies_per_wafer": 4, "n": 2, "area": 1.5, "trigger_limit_yield": 0.7, "good_yield": 0.9, "trigger_limit_fc_yield": 0.6, "good_fc_yield": 0.9, "good_test_time": 9.0, "trigger_limit_test_time": 18.0}]
    )

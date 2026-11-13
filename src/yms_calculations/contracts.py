"""Transport-neutral types shared by YMS report calculations."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CalculationResult:
    """The semantic result returned by a report calculation.

    The application owns HTTP serialization and UI presentation.  The data
    package returns the calculated rows, effective time window and canonical
    names of requested dynamic metrics.
    """

    frame: pd.DataFrame
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    dynamic_metric_keys: tuple[str, ...] = ()

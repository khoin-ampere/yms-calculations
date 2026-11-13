# yms-calculations

`yms-calculations` contains the transport-neutral transformations and
aggregations for YMS03, YMS04, and YMS05. It intentionally has no FastAPI,
Redis, Celery, filesystem, credential, or HTTP dependency.

## Public contract

```python
from yms_calculations import get_report

result = get_report("yms04").calculate(dataframe, commitments, filters)
```

The result is a `CalculationResult` with the calculated dataframe, effective
date range, and dynamic metric keys. The consuming application owns storage,
cache, API serialization, scheduling, and deployment.

## Development

```bash
uv sync --extra dev
uv run pytest
```

Release with semantic versioning. Applications must depend on a released,
pinned package version and must not execute a source checkout at runtime.

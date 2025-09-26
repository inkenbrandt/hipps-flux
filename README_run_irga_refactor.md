# Refactor: CalcFlux.run_irga

Changes:
- Signature: `run_irga(self, df, *, rename_map: dict | None = None, ts_col: str | None = None, lazy: bool = False) -> pd.Series`
- Accepts: pandas.DataFrame, polars.DataFrame, or path to CSV/Parquet
- Optional: `rename_map` to override/extend default renaming
- Optional: `ts_col` to enforce datetime casting
- Keeps return type: `pandas.Series` (delegates to original pipeline after preprocessing)

Example:
```python
from ec_polars_refactored import CalcFlux
import polars as pl

df = pl.read_parquet("sample.parquet")  # or pandas.read_parquet / path string
fx = CalcFlux()

rename_map = {"Ux_7200": "Ux", "Uy_7200": "Uy", "Uz_7200": "Uz", "Tsonic": "TA"}
s = fx.run_irga(df, rename_map=rename_map, ts_col="TIMESTAMP")
print(s)
```

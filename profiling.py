import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, nulls, uniqueness, and a quick peek at the values."""
    rows = []
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "nulls": int(series.isna().sum()),
            "null_%": round(series.isna().mean() * 100, 1),
            "unique": int(series.nunique()),
            "min": series.min() if is_numeric else None,
            "max": series.max() if is_numeric else None,
            "sample": ", ".join(map(str, series.dropna().unique()[:3])),
        })
    return pd.DataFrame(rows)

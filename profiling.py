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


def profile_to_text(df: pd.DataFrame, profile: pd.DataFrame, sample_rows: int = 5) -> str:
    """Compact text form of the profile + a small sample, for feeding into an LLM prompt."""
    lines = [f"{len(df)} rows, {len(df.columns)} columns.", "", "Columns:"]
    for _, r in profile.iterrows():
        extra = f"range [{r['min']}, {r['max']}]" if pd.notna(r["min"]) else f"examples: {r['sample']}"
        lines.append(f"- {r['column']} ({r['dtype']}), {r['nulls']} nulls, {extra}")
    lines.append("")
    lines.append(f"First {sample_rows} rows:")
    lines.append(df.head(sample_rows).to_string(index=False))
    return "\n".join(lines)

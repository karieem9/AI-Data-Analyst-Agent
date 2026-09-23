import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MAX_BAR_CATEGORIES = 40


def _looks_like_dates(values) -> bool:
    """True dtype check first; falls back to a parse-rate heuristic since
    pd.read_csv loads date columns as plain strings by default."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return True
    try:
        parsed = pd.to_datetime(pd.Series(values), errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return False
    return parsed.notna().mean() > 0.8


def auto_chart(result):
    """
    Turn a sandboxed result into something the UI can render.
    Returns ("figure", plotly Figure) | ("metric", value) | ("table", DataFrame).
    """
    if isinstance(result, go.Figure):
        return "figure", result
    if isinstance(result, pd.Series):
        return _chart_from_series(result)
    if isinstance(result, pd.DataFrame):
        return _chart_from_dataframe(result)
    if result is None:
        return "metric", "No result."
    return "metric", result


def _chart_from_series(series: pd.Series):
    if series.empty:
        return "table", series.to_frame(name="value")

    label = series.name or "value"

    if _looks_like_dates(series.index):
        x = pd.to_datetime(series.index, errors="coerce", format="mixed")
        fig = px.line(x=x, y=series.values, labels={"x": series.index.name or "date", "y": label})
        return "figure", fig

    if series.nunique() <= MAX_BAR_CATEGORIES and not pd.api.types.is_numeric_dtype(series.index):
        fig = px.bar(x=series.index.astype(str), y=series.values, labels={"x": series.index.name or "category", "y": label})
        return "figure", fig

    return "table", series.to_frame(name=label)


def _chart_from_dataframe(df: pd.DataFrame):
    if df.empty:
        return "table", df

    if len(df.columns) == 1:
        return _chart_from_series(df.iloc[:, 0])

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in df.columns if c not in numeric_cols and _looks_like_dates(df[c])]
    other_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]

    if date_cols and numeric_cols:
        plot_df = df.copy()
        plot_df[date_cols[0]] = pd.to_datetime(plot_df[date_cols[0]], errors="coerce", format="mixed")
        fig = px.line(plot_df.sort_values(date_cols[0]), x=date_cols[0], y=numeric_cols[0])
        return "figure", fig

    if other_cols and numeric_cols and len(df) <= MAX_BAR_CATEGORIES:
        fig = px.bar(df, x=other_cols[0], y=numeric_cols[0])
        return "figure", fig

    return "table", df

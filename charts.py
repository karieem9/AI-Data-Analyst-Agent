import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from forecasting import is_forecast_frame

MAX_BAR_CATEGORIES = 40


MAX_DATE_PARSE_SAMPLE = 200


def looks_like_dates(values) -> bool:
    """True dtype check first; falls back to a parse-rate heuristic since
    pd.read_csv loads date columns as plain strings by default.

    Parses at most MAX_DATE_PARSE_SAMPLE unique values, not the whole column
    -- pd.to_datetime(format="mixed") falls back to per-row dateutil parsing
    on non-standard strings, which is fast on a handful of dates but takes
    seconds per column on a high-cardinality ID column with hundreds of
    thousands of rows (e.g. "PROD-000738" x 400k rows)."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return True
    sample = pd.Series(values).dropna().unique()
    if len(sample) == 0:
        return False
    if len(sample) > MAX_DATE_PARSE_SAMPLE:
        sample = sample[:MAX_DATE_PARSE_SAMPLE]
    try:
        parsed = pd.to_datetime(pd.Series(sample), errors="coerce", format="mixed")
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
    if is_forecast_frame(result):
        return "figure", forecast_chart(result)
    if isinstance(result, pd.DataFrame):
        return _chart_from_dataframe(result)
    if result is None:
        return "metric", "No result."
    return "metric", result


def _chart_from_series(series: pd.Series):
    if series.empty:
        return "table", series.to_frame(name="value")

    if len(series) == 1:
        # A "chart" with one point isn't a chart -- it's a number. This
        # also sidesteps a real Plotly bug: a single-point line chart has
        # no range to compute an axis from, so it auto-generates a
        # nonsensical sub-millisecond-tick range around that one point.
        return "metric", series.iloc[0]

    label = series.name or "value"

    if looks_like_dates(series.index):
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

    if len(df) == 1:
        return "table", df

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in df.columns if c not in numeric_cols and looks_like_dates(df[c])]
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


def forecast_chart(frame: pd.DataFrame) -> go.Figure:
    """History as a solid line, the forecast dashed after it, and the
    10%-90% range as a shaded band around the forecast."""
    history = frame["actual"].dropna()
    future = frame.dropna(subset=["forecast"])
    # Start the forecast line at the last real point so the two connect.
    x_fc = [history.index[-1], *future.index]
    fig = go.Figure([
        go.Scatter(x=future.index, y=future["upper"], mode="lines", line={"width": 0},
                   showlegend=False, hoverinfo="skip"),
        go.Scatter(x=future.index, y=future["lower"], mode="lines", line={"width": 0},
                   fill="tonexty", fillcolor="rgba(99, 110, 250, 0.2)", name="likely range (10-90%)"),
        go.Scatter(x=history.index, y=history.values, mode="lines", name="actual"),
        go.Scatter(x=x_fc, y=[history.iloc[-1], *future["forecast"]], mode="lines",
                   line={"dash": "dash"}, name="forecast"),
    ])
    fig.update_layout(xaxis_title=frame.index.name or "date", hovermode="x unified")
    return fig

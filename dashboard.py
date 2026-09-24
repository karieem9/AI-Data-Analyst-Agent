import numpy as np
import pandas as pd
import plotly.express as px

from charts import looks_like_dates

MAX_NUMERIC_CHARTS = 6
MAX_CATEGORICAL_CHARTS = 6
MAX_CATEGORY_BARS = 12
MAX_CATEGORY_CARDINALITY = 50  # above this, a column is an ID, not a category


def build_dashboard(df: pd.DataFrame) -> list[dict]:
    """Auto-generated overview charts for the whole dataset: missing values,
    a distribution per numeric column, rows over time if a date column
    exists, and top values for low-cardinality categorical columns.
    Returns a list of {"title": str, "figure": plotly Figure}.
    """
    charts = []

    null_pct = (df.isna().mean() * 100).round(1)
    if null_pct.sum() > 0:
        fig = px.bar(
            x=null_pct.index.astype(str), y=null_pct.values,
            labels={"x": "column", "y": "% missing"},
        )
        charts.append({"title": "Missing values by column", "figure": fig})

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in df.columns if c not in numeric_cols and looks_like_dates(df[c])]
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and c not in date_cols and df[c].nunique() <= MAX_CATEGORY_CARDINALITY
    ]

    for col in numeric_cols[:MAX_NUMERIC_CHARTS]:
        values = df[col].dropna()
        if values.empty or values.nunique() < 2:
            continue
        # Bin server-side rather than px.histogram(df, x=col), which ships
        # every raw value into the figure JSON for client-side binning --
        # fine at hundreds of rows, seconds-slow and megabytes-large at
        # hundreds of thousands.
        counts, edges = np.histogram(values, bins=30)
        centers = [f"{(edges[i] + edges[i + 1]) / 2:.3g}" for i in range(len(edges) - 1)]
        fig = px.bar(x=centers, y=counts, labels={"x": col, "y": "count"})
        charts.append({"title": f"Distribution of {col}", "figure": fig})

    if date_cols:
        col = date_cols[0]
        dates = pd.to_datetime(df[col], errors="coerce", format="mixed")
        counts = dates.dt.date.value_counts().sort_index()
        if len(counts) > 1:
            fig = px.line(x=counts.index, y=counts.values, labels={"x": col, "y": "rows"})
            charts.append({"title": f"Rows over time ({col})", "figure": fig})

    for col in categorical_cols[:MAX_CATEGORICAL_CHARTS]:
        counts = df[col].value_counts().head(MAX_CATEGORY_BARS)
        if len(counts) < 2:
            continue
        fig = px.bar(x=counts.index.astype(str), y=counts.values, labels={"x": col, "y": "count"})
        charts.append({"title": f"Top values in {col}", "figure": fig})

    return charts

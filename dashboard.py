import numpy as np
import pandas as pd
import plotly.express as px

from charts import looks_like_dates

MAX_NUMERIC_CHARTS = 6
MAX_CATEGORICAL_CHARTS = 6
MAX_CATEGORY_BARS = 12
MAX_CATEGORY_CARDINALITY = 50  # above this, a column is an ID, not a category

# Numeric columns whose name suggests a real business metric are shown
# before generic ones, so a 10-column dataset doesn't silently cut off
# "profit" just because it's column 9 and the cap is 6. This is a ranked
# priority, not a flat matched/unmatched split -- a flat split does
# nothing on a dataset where every numeric column is money-adjacent
# (quantity, discount_amount, tax_amount all "match" something), which
# is exactly the case that motivated this fix in the first place. Earlier
# entries outrank later ones; a column matches its first hit in this list.
METRIC_KEYWORDS = (
    "profit", "margin", "revenue", "income",
    "cost", "sales", "price",
    "expense", "net", "gross",
    "amount", "total", "value",
    "discount", "tax", "fee", "spend", "budget",
    "quantity", "units", "count",
)


def _priority_numeric_cols(numeric_cols):
    def rank(col):
        name = col.lower()
        for i, kw in enumerate(METRIC_KEYWORDS):
            if kw in name:
                return i
        return len(METRIC_KEYWORDS)
    return sorted(numeric_cols, key=rank)  # stable: ties keep original order


def _iqr_outlier_counts(df: pd.DataFrame, numeric_cols) -> dict:
    """Rows outside 1.5x IQR per numeric column. Fully vectorized (quantile
    + boolean mask), so it stays fast at hundreds of thousands of rows."""
    counts = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        if n_out > 0:
            counts[col] = n_out
    return counts


def build_data_quality(df: pd.DataFrame, numeric_cols) -> list[dict]:
    """Gaps and problems in the raw data -- meant to be looked at before
    cleaning anything, not a summary of the data itself."""
    cards = []

    null_pct = (df.isna().mean() * 100).round(1)
    missing = null_pct[null_pct > 0]
    if len(missing) > 0:
        fig = px.bar(x=missing.index.astype(str), y=missing.values, labels={"x": "column", "y": "% missing"})
        cards.append({"type": "figure", "section": "quality", "title": "Missing values by column", "figure": fig, "severity": "warn"})
    else:
        cards.append({"type": "stat", "section": "quality", "title": "Missing values", "value": "None found", "severity": "ok"})

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        pct = round(dup_count / len(df) * 100, 1)
        cards.append({"type": "stat", "section": "quality", "title": "Duplicate rows", "value": f"{dup_count:,} rows ({pct}%)", "severity": "warn"})
    else:
        cards.append({"type": "stat", "section": "quality", "title": "Duplicate rows", "value": "None found", "severity": "ok"})

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        cards.append({"type": "list", "section": "quality", "title": "Constant columns (no variation)", "items": constant_cols, "severity": "warn"})
    else:
        cards.append({"type": "stat", "section": "quality", "title": "Constant columns", "value": "None found", "severity": "ok"})

    outliers = _iqr_outlier_counts(df, numeric_cols)
    if outliers:
        cols = list(outliers.keys())[:MAX_NUMERIC_CHARTS]
        fig = px.bar(x=cols, y=[outliers[c] for c in cols], labels={"x": "column", "y": "outlier rows"})
        cards.append({"type": "figure", "section": "quality", "title": "Potential outliers per column (IQR method)", "figure": fig, "severity": "warn"})
    else:
        cards.append({"type": "stat", "section": "quality", "title": "Potential outliers", "value": "None found (IQR method)", "severity": "ok"})

    return cards


def build_dashboard(df: pd.DataFrame) -> list[dict]:
    """Data quality cards first, then overview charts. Each card is
    {"type": "figure"|"stat"|"list", "section": "quality"|"overview",
    "title": str, plus figure/value/items, plus "severity" on quality cards}.
    """
    numeric_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in df.columns if c not in numeric_cols_all and looks_like_dates(df[c])]
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols_all and c not in date_cols and df[c].nunique() <= MAX_CATEGORY_CARDINALITY
    ]

    cards = build_data_quality(df, numeric_cols_all)

    numeric_cols = _priority_numeric_cols(numeric_cols_all)
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
        cards.append({"type": "figure", "section": "overview", "title": f"Distribution of {col}", "figure": fig})

    if date_cols:
        col = date_cols[0]
        dates = pd.to_datetime(df[col], errors="coerce", format="mixed")
        counts = dates.dt.date.value_counts().sort_index()
        if len(counts) > 1:
            fig = px.line(x=counts.index, y=counts.values, labels={"x": col, "y": "rows"})
            cards.append({"type": "figure", "section": "overview", "title": f"Rows over time ({col})", "figure": fig})

    for col in categorical_cols[:MAX_CATEGORICAL_CHARTS]:
        counts = df[col].value_counts().head(MAX_CATEGORY_BARS)
        if len(counts) < 2:
            continue
        fig = px.bar(x=counts.index.astype(str), y=counts.values, labels={"x": col, "y": "count"})
        cards.append({"type": "figure", "section": "overview", "title": f"Top values in {col}", "figure": fig})

    return cards

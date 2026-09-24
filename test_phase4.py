import pandas as pd

from charts import auto_chart
from llm import generate_code
from profiling import profile_dataframe, profile_to_text
from sandbox import run_safely

df = pd.read_csv("sample_sales.csv")
profile = profile_dataframe(df)
profile_text = profile_to_text(df, profile)

print("=" * 60)
print("REAL QUESTIONS -> generate -> execute -> auto_chart")
print("=" * 60)

figures_to_export = {}

for q in [
    "What was total revenue?",
    "What are the total units sold per region?",
    "Show revenue by date as a trend.",
]:
    code = generate_code(q, profile_text)
    result = run_safely(code, df)
    kind, payload = auto_chart(result)
    print(f"Q: {q}\nCODE: {code}\nKIND: {kind}")
    if kind == "figure":
        print(f"  traces: {len(payload.data)}, type: {payload.data[0].type}")
        figures_to_export[q] = payload
    else:
        print(f"  payload: {payload!r}")
    print()

print("=" * 60)
print("GENERATED CODE THAT BUILDS ITS OWN FIGURE")
print("=" * 60)
own_fig_code = "result = px.bar(df.groupby('product')['revenue'].sum().reset_index(), x='product', y='revenue')"
result = run_safely(own_fig_code, df)
kind, payload = auto_chart(result)
print(f"CODE: {own_fig_code}\nKIND: {kind} (should be figure, passed through as-is)\n")
figures_to_export["Own figure (px.bar by product)"] = payload

print("=" * 60)
print("EDGE CASES")
print("=" * 60)
edge_cases = {
    "empty series": pd.Series([], dtype=float, name="revenue"),
    "scalar": 26230,
    "none": None,
    "wide categories (60 unique)": pd.Series(range(60), index=[f"item_{i}" for i in range(60)]),
    # Regression guard: a single-point "trend" used to become a broken
    # line chart -- Plotly has no range to compute an axis from with only
    # one point, so it auto-generates a nonsensical sub-millisecond-tick
    # range around it. Found via "total revenue per month" on data that
    # only spans one month (one groupby bucket = one row).
    "single date-indexed row": pd.Series([26230], index=pd.Index(["2026-06"], name="date"), name="revenue"),
}
for label, val in edge_cases.items():
    kind, payload = auto_chart(val)
    print(f"{label}: KIND={kind}")

assert auto_chart(edge_cases["single date-indexed row"])[0] == "metric", \
    "a single-point result must never become a figure"

# Export the figures to one HTML file for a visual look in the browser.
with open("phase4_preview.html", "w", encoding="utf-8") as f:
    f.write("<html><body style='font-family:sans-serif;background:#111;color:#eee'>\n")
    for title, fig in figures_to_export.items():
        f.write(f"<h3>{title}</h3>\n")
        f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    f.write("</body></html>\n")
print("\nWrote phase4_preview.html")

import time

import pandas as pd

from dashboard import build_dashboard

df = pd.read_csv("sample_sales.csv")
charts = build_dashboard(df)

print(f"sample_sales.csv -> {len(charts)} charts:")
for c in charts:
    print(f"  - {c['title']} ({c['figure'].data[0].type})")

assert len(charts) == 8, f"expected 8 charts, got {len(charts)}"
titles = {c["title"] for c in charts}
assert "Missing values by column" in titles, "sample data has a deliberate null, should be flagged"
assert "Rows over time (date)" in titles, "date column should be detected"
assert "Top values in region" in titles
print("PASS: sample_sales.csv dashboard shape is correct\n")

# Performance regression guard: a high-cardinality ID-like string column
# used to take ~12s alone (looks_like_dates falling back to per-row
# dateutil parsing on 400k+ rows). Must stay fast regardless of row count.
big = pd.DataFrame({
    "product_id": [f"PROD-{i:06d}" for i in range(50_000)],
    "revenue": range(50_000),
})
t0 = time.time()
big_charts = build_dashboard(big)
elapsed = time.time() - t0
print(f"50k-row high-cardinality-ID dataset -> {len(big_charts)} charts in {elapsed:.2f}s")
assert elapsed < 3.0, f"dashboard took {elapsed:.2f}s on 50k rows -- looks_like_dates regression?"
print("PASS: performance regression guard\n")

print("All dashboard tests passed.")

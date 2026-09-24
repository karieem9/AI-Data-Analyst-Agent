import time

import pandas as pd

from dashboard import _priority_numeric_cols, build_dashboard

df = pd.read_csv("sample_sales.csv")
cards = build_dashboard(df)

print(f"sample_sales.csv -> {len(cards)} cards:")
for c in cards:
    extra = c.get("value") or c.get("items") or (c["figure"].data[0].type if c["type"] == "figure" else "")
    print(f"  [{c['section']}] {c['type']:6s} {c.get('severity', '-'):5s} {c['title']:40s} {extra}")

quality = [c for c in cards if c["section"] == "quality"]
overview = [c for c in cards if c["section"] == "overview"]
assert len(quality) == 4, f"expected 4 quality cards (missing/dup/constant/outliers), got {len(quality)}"
assert quality == cards[: len(quality)], "quality cards must come before overview cards"

titles = {c["title"] for c in overview}
assert "Rows over time (date)" in titles, "date column should be detected"
assert "Top values in region" in titles
print("PASS: sample_sales.csv dashboard shape is correct\n")

# Data quality checks against a dataset with real, deliberate problems.
dirty = pd.DataFrame({
    "region": ["North", "South", "North", "East"],
    "revenue": [100, 200, 100, 300],
    "flag": ["A", "A", "A", "A"],  # constant
})
from dashboard import build_data_quality
dq = {c["title"]: c for c in build_data_quality(dirty, ["revenue"])}
assert dq["Duplicate rows"]["severity"] == "warn", "one exact duplicate row should be flagged"
assert "1" in dq["Duplicate rows"]["value"]
assert dq["Constant columns (no variation)"]["severity"] == "warn"
assert dq["Constant columns (no variation)"]["items"] == ["flag"]
print("PASS: duplicate rows and constant columns are correctly detected\n")

# Column priority: on a dataset where every numeric column is money-adjacent
# (the case that motivated this fix), the real headline metrics must still
# rank first -- a flat matched/unmatched split does nothing here, since
# everything matches something.
order_like = ["quantity", "unit_price", "discount_percentage", "discount_amount",
              "gross_sales", "tax_amount", "shipping_cost", "net_sales",
              "product_cost", "profit"]
ranked = _priority_numeric_cols(order_like)
assert ranked[0] == "profit", f"profit should rank first, got order: {ranked}"
assert ranked.index("quantity") == len(ranked) - 1, "quantity (not money-specific) should rank last"
print(f"PASS: column priority ranks profit first -> {ranked}\n")

# Performance regression guard: a high-cardinality ID-like string column
# used to take ~12s alone (looks_like_dates falling back to per-row
# dateutil parsing on 400k+ rows). Must stay fast regardless of row count.
big = pd.DataFrame({
    "product_id": [f"PROD-{i:06d}" for i in range(50_000)],
    "revenue": range(50_000),
})
t0 = time.time()
big_cards = build_dashboard(big)
elapsed = time.time() - t0
print(f"50k-row high-cardinality-ID dataset -> {len(big_cards)} cards in {elapsed:.2f}s")
assert elapsed < 3.0, f"dashboard took {elapsed:.2f}s on 50k rows -- looks_like_dates regression?"
print("PASS: performance regression guard\n")

print("All dashboard tests passed.")

import pandas as pd

from explain import _result_to_text, explain_result
from llm import generate_code
from profiling import profile_dataframe, profile_to_text
from sandbox import run_safely

df = pd.read_csv("sample_sales.csv")
profile = profile_dataframe(df)
profile_text = profile_to_text(df, profile)

print("=" * 60)
print("END TO END: generate -> execute -> explain")
print("=" * 60)
for q in [
    "What was total revenue?",
    "Which department has the highest total cost?",
    "What are the total units sold per region?",
    "Why did units sold drop on 2026-06-10?",
]:
    code = generate_code(q, profile_text)
    result = run_safely(code, df)
    explanation = explain_result(q, result)
    print(f"Q: {q}")
    print(f"RESULT: {result!r}")
    print(f"EXPLANATION: {explanation}")
    print()

print("=" * 60)
print("TRUNCATION CHECK (no API call, just formatting)")
print("=" * 60)
big_series = pd.Series(range(50), index=[f"item_{i}" for i in range(50)], name="value")
text = _result_to_text(big_series)
print(text)
print(f"\nLine count: {len(text.splitlines())} (should be capped, not 50 rows)")

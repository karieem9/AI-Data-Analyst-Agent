import pandas as pd

from llm import generate_code
from profiling import profile_dataframe, profile_to_text

df = pd.read_csv("sample_sales.csv")
profile = profile_dataframe(df)
profile_text = profile_to_text(df, profile)

questions = [
    "What was total revenue?",
    "Which department has the highest total cost?",
    "What are the total units sold per region?",
    "What is the average revenue per product?",
    "Which region had the lowest revenue?",
    "Show revenue by date as a trend.",
    # Regression guard: this used to false-positive as a "modify data" request
    # because of the word "drop" -- must compute a real answer, not refuse.
    "Why did units sold drop on 2026-06-10?",
]

for q in questions:
    print("=" * 60)
    print("Q:", q)
    code = generate_code(q, profile_text)
    print("-" * 60)
    print(code)
    print()

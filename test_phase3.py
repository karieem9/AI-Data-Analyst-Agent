import pandas as pd

from llm import generate_code
from profiling import profile_dataframe, profile_to_text
from sandbox import UnsafeCodeError, run_safely

df = pd.read_csv("sample_sales.csv")
profile = profile_dataframe(df)
profile_text = profile_to_text(df, profile)

print("=" * 60)
print("REAL QUESTIONS -> generate -> execute -> real result")
print("=" * 60)
for q in [
    "What was total revenue?",
    "Which department has the highest total cost?",
    "What are the total units sold per region?",
]:
    code = generate_code(q, profile_text)
    try:
        result = run_safely(code, df)
        print(f"Q: {q}\nCODE: {code}\nRESULT:\n{result}\n")
    except Exception as e:
        print(f"Q: {q}\nCODE: {code}\nFAILED: {type(e).__name__}: {e}\n")

print("=" * 60)
print("ADVERSARIAL CODE -> should all be rejected")
print("=" * 60)
attacks = [
    "import os\nresult = os.listdir('.')",
    "result = open('sample_sales.csv').read()",
    "df.drop(index=0, inplace=True)\nresult = df",
    "df = pd.DataFrame({'x': [1]})\nresult = df",
    "df['revenue'] = 0\nresult = df",
    "df.loc[0, 'revenue'] = 0\nresult = df",
    "del df\nresult = 'gone'",
    "result = __import__('os').getcwd()",
    "result = eval('1+1')",
    "while True:\n    pass",
]
for code in attacks:
    try:
        result = run_safely(code, df, timeout=3)
        print(f"CODE: {code!r}\n>> NOT BLOCKED! result={result!r}\n")
    except UnsafeCodeError as e:
        print(f"CODE: {code!r}\n>> blocked at validation: {e}\n")
    except TimeoutError as e:
        print(f"CODE: {code!r}\n>> blocked by timeout: {e}\n")
    except Exception as e:
        print(f"CODE: {code!r}\n>> failed with {type(e).__name__}: {e}\n")

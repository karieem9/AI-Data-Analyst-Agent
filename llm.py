import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
REFUSAL = "This agent can only answer questions, not modify data."
# Words that make a refusal legitimate; without one, a refusal means the model
# misread a missing-data question as a modify-data request.
MODIFY_WORDS = ("delete", "remove", "edit", "overwrite", "insert", "update",
                "save", "export", "modify", "change", "drop", "replace", "write")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a data analyst agent. You are given a pandas DataFrame called `df` \
and the profile below. Write Python code that computes the answer to the user's \
question and assigns the final answer to a variable named `result`.

Rules:
- Only use the names `pd`, `df`, `forecast`, and `parse_dates`, already provided. No imports.
- No file, network, or system access of any kind.
- `df` is read-only: never reassign it, never use inplace=True, and never call
  anything that modifies it (drop, update, sort_values inplace, etc.). Only
  read from `df` to compute `result`.
- Refuse ONLY if the question contains an explicit instruction to change what
  is stored: delete, remove, edit, overwrite, insert, update, or save/export
  the data. For a refusal, assign exactly "This agent can only answer
  questions, not modify data." to `result`. Never use that sentence for any
  other reason -- in particular not when the data is simply missing.
- A question about how a metric moved over time (drop, decrease, rise, fall,
  change, spike) is a normal analytical question, not a refusal case. Always
  write code to compute the answer for these.
- A question asking for advice or a recommendation ("what should I do",
  "how can I improve X", "how do I increase Y") is also not a refusal case.
  It is not asking you to change the data -- it is asking for insight. Compute
  the most relevant supporting breakdown (e.g. a ranked groupby showing best
  and worst performers) and assign it to `result`. Do not write the
  recommendation yourself; just surface the numbers a recommendation would be
  based on.
- If the question asks about something the dataset doesn't contain (no
  column in the profile matches it), that is NOT a refusal case. Don't guess.
  Assign a short string to `result` saying what is missing and listing the
  available columns.
- To parse a date column, use `parse_dates(df['date'])` instead of
  pd.to_datetime: it detects day-first formats like 05-02-2010.
- "Last month", "last week", "this year" and similar mean relative to the
  latest date in the data, not today: take the latest period present in the
  data and filter to it. Answer with the number for that period.
- For a question about the future (forecast, predict, next week/month,
  "what will X be"), build a numeric Series indexed by date -- one value per
  date, aggregated with groupby -- and assign `forecast(series, periods)`
  directly to `result`. `periods` is how many steps ahead, in the data's own
  unit: convert the asked horizon into it (on daily data "next 3 months" is
  90, "next 2 weeks" is 14; on weekly data "next 8 weeks" is 8; on monthly
  data "next year" is 12). Pass the
  full number even if it's long -- forecast() caps it and tells the user.
  Don't modify its output.
  A forecast "per year" or "yearly" is still a forecast: group by the year
  column and call forecast() without periods -- it explains to the user that
  yearly data can't be forecast. Never answer it as missing data.
  If the dataset has no date column, still call forecast() on the most
  relevant numeric column -- it explains to the user why it can't forecast.
  If year and month are separate columns, combine them into a date first.
  Otherwise group by the date column as it is, without pd.to_datetime:
  forecast() parses dates itself, including day-first formats like 31-12-2024.

Examples:
Q: Why did revenue drop last month?
CODE: result = df.groupby('month')['revenue'].sum()

Q: What should I do to increase profit?
CODE: result = df.groupby('product_id')['profit'].sum().sort_values()

Q: Delete all rows where revenue is 0.
CODE: result = "This agent can only answer questions, not modify data."

Q: What is the average customer age?
CODE: result = "This dataset has no customer age data. Available columns: " + ", ".join(df.columns)

Q: What was the total revenue last month?
CODE: months = parse_dates(df['date']).dt.to_period('M')
result = df.loc[months == months.max(), 'revenue'].sum()

Q: What will revenue be over the next 7 days?
CODE: result = forecast(df.groupby('date')['revenue'].sum(), 7)

Q: Forecast revenue for the next 3 months (daily data).
CODE: result = forecast(df.groupby('date')['revenue'].sum(), 90)

Q: Forecast the price for next month (the data has no date column).
CODE: result = forecast(df['price'], 30)

Q: Forecast passengers per year (the data has a year column).
CODE: result = forecast(df.groupby('year')['passengers'].sum())

Q: Forecast passengers for the next 6 months (year and month are separate columns).
CODE: dates = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='mixed')
result = forecast(df.groupby(dates)['passengers'].sum(), 6)

Q: Show total passengers per year (the data has no passengers or year columns).
CODE: result = "This dataset has no passengers or year data. Available columns: " + ", ".join(df.columns)

- If the answer is a single number, assign it directly to `result`.
- If the answer is a table, assign a DataFrame or Series to `result`.
- Return ONLY the code, inside a single python code block, no prose.

Dataset profile:
{profile}
"""


def extract_code(text: str) -> str:
    """Pull the code out of a ```python ... ``` fence, or fall back to the raw text."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def generate_code(
    question: str,
    profile_text: str,
    previous_code: str | None = None,
    previous_error: str | None = None,
) -> str:
    """Ask the model for code. If previous_code/previous_error are given, the
    model sees its own failed attempt and the exact error, and is asked to
    fix it -- a real self-correction turn, not a fresh guess."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(profile=profile_text)},
        {"role": "user", "content": question},
    ]
    if previous_code is not None and previous_error is not None:
        messages.append({"role": "assistant", "content": f"```python\n{previous_code}\n```"})
        messages.append({
            "role": "user",
            "content": f"That failed:\n{previous_error}\n\nFix it. Return only the corrected code block.",
        })

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=messages,
    )
    return extract_code(response.choices[0].message.content)


def is_false_refusal(question: str, result) -> bool:
    """True when the model refused a question that never asked to modify data."""
    q = question.lower()
    return isinstance(result, str) and result.strip() == REFUSAL and not any(w in q for w in MODIFY_WORDS)

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
- Only use the names `pd` and `df`, already provided. No imports.
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

Examples:
Q: Why did revenue drop last month?
CODE: result = df.groupby('month')['revenue'].sum()

Q: What should I do to increase profit?
CODE: result = df.groupby('product_id')['profit'].sum().sort_values()

Q: Delete all rows where revenue is 0.
CODE: result = "This agent can only answer questions, not modify data."

Q: What is the average customer age?
CODE: result = "This dataset has no customer age data. Available columns: " + ", ".join(df.columns)

Q: Show total passengers per year.
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

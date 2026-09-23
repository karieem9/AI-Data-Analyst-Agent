import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
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
  the data. For a refusal, assign a short string to `result` explaining you
  can only answer questions, not modify data.
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

Examples:
Q: Why did revenue drop last month?
CODE: result = df.groupby('month')['revenue'].sum()

Q: What should I do to increase profit?
CODE: result = df.groupby('product_id')['profit'].sum().sort_values()

Q: Delete all rows where revenue is 0.
CODE: result = "This agent can only answer questions, not modify data."

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


def generate_code(question: str, profile_text: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(profile=profile_text)},
            {"role": "user", "content": question},
        ],
    )
    return extract_code(response.choices[0].message.content)

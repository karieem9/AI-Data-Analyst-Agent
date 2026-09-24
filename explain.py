import os

import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI

from forecasting import is_forecast_frame

load_dotenv()

MODEL = "gpt-4o-mini"
MAX_ROWS_FOR_EXPLANATION = 20
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You explain a data analysis result to a manager in plain language.

Rules:
- Write 2-3 short sentences, no more.
- Reference only entities, categories, dates, and numbers that literally
  appear in the result below. Never invent a category, date, product, or
  percentage that isn't in it, even if it would make for a plausible-sounding
  answer -- if the result doesn't contain something, you don't know it.
- Never claim a cause you can't see in the result itself. Describe what the
  data shows -- a drop concentrated in one category, a spike on one date --
  rather than asserting why it happened unless the result makes that explicit.
- If the question asks for advice or a recommendation, you may give one, but
  it must follow directly from the actual values in the result (e.g. "focus
  on product X, it has the lowest profit at $Y"). A recommendation grounded
  in the real numbers is welcome; a recommendation built on invented numbers
  is not.
- If the result is a forecast, say plainly that it's an estimate from a
  forecasting model, give the forecast with its likely range, and compare it
  to the recent actual values. Mention any notes that come with it.
- If the result is a message saying the data isn't available, that it
  can't be forecast, or that the agent can't modify data, restate that message in one sentence and add
  nothing else.
- No preamble like "Based on the data" or "The result shows". Just say it.
"""


def _forecast_to_text(frame: pd.DataFrame) -> str:
    info = frame.attrs.get("forecast", {})
    history = frame["actual"].dropna()
    future = frame.dropna(subset=["forecast"])[["forecast", "lower", "upper"]]
    text = (
        f"Forecast ({info.get('frequency', '')}, {len(future)} steps ahead; lower/upper is the "
        f"10%-90% likely range):\n{future.to_string()}\n\n"
        f"Recent actual values:\n{history.tail(len(future)).to_string()}"
    )
    for note in info.get("notes", []):
        text += f"\nNote: {note}"
    return text


def _result_to_text(result) -> str:
    if is_forecast_frame(result):
        return _forecast_to_text(result)
    if isinstance(result, go.Figure):
        return "A chart was generated directly by the analysis code; no tabular result to describe."
    if isinstance(result, (pd.Series, pd.DataFrame)):
        truncated = len(result) > MAX_ROWS_FOR_EXPLANATION
        text = result.head(MAX_ROWS_FOR_EXPLANATION).to_string()
        if truncated:
            text += f"\n... ({len(result) - MAX_ROWS_FOR_EXPLANATION} more rows, not shown)"
        return text
    return str(result)


def explain_result(question: str, result) -> str:
    result_text = _result_to_text(result)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nResult:\n{result_text}"},
        ],
    )
    return response.choices[0].message.content.strip()

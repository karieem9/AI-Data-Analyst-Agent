import json

import pandas as pd

from charts import auto_chart
from explain import explain_result
from forecasting import ForecastError
from llm import generate_code, is_false_refusal
from sandbox import UnsafeCodeError, run_safely

MAX_ATTEMPTS = 2


def answer_question(question: str, df: pd.DataFrame, profile_text: str) -> dict:
    """
    Generate code for `question`, run it safely, retrying once on failure --
    the model sees its own code and the real error, then gets one shot to
    fix it. Only raises if the model/API itself is unreachable; a code
    generation or execution failure always comes back as payload["error"],
    the real message, never a generic one.
    """
    code = None
    error = None
    result = None
    attempts = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts = attempt
        if attempt == 1:
            code = generate_code(question, profile_text)
        else:
            code = generate_code(question, profile_text, previous_code=code, previous_error=error)

        try:
            result = run_safely(code, df)
            error = None
            break
        except ForecastError as e:
            # The data can't be forecast (no dates, too short...). That's
            # the answer, not a bug -- retrying would only rephrase the code.
            result = str(e)
            error = None
            break
        except UnsafeCodeError as e:
            error = f"Rejected for safety: {e}"
        except TimeoutError as e:
            error = str(e)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

    payload = {"question": question, "code": code, "error": error, "attempts": attempts}

    if error is not None:
        return payload

    if is_false_refusal(question, result):
        columns = ", ".join(map(str, df.columns))
        result = f"This dataset doesn't have the data to answer that. Available columns: {columns}"

    kind, chart_payload = auto_chart(result)
    payload["kind"] = kind
    if kind == "figure":
        payload["figure"] = json.loads(chart_payload.to_json())
    elif kind == "table":
        payload["table"] = chart_payload.reset_index().fillna("").to_dict(orient="records")
    else:
        payload["metric"] = str(chart_payload)

    try:
        payload["explanation"] = explain_result(question, result)
    except Exception as e:
        payload["explanation"] = None
        payload["explanation_error"] = f"Couldn't generate an explanation: {e}"

    return payload

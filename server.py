import json

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from charts import auto_chart
from explain import explain_result
from llm import generate_code
from profiling import profile_dataframe, profile_to_text
from sandbox import UnsafeCodeError, run_safely

app = FastAPI(title="AI Data Analyst Agent")

# Single-user local tool: one global dataset + history, no session management.
STATE = {"df": None, "profile_text": None, "filename": None, "history": []}


class AskRequest(BaseModel):
    question: str


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(400, f"Couldn't read that file: {e}")

    if df.empty:
        raise HTTPException(400, "That file has no rows.")

    profile = profile_dataframe(df)
    STATE["df"] = df
    STATE["profile_text"] = profile_to_text(df, profile)
    STATE["filename"] = file.filename
    STATE["history"] = []

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "profile": profile.fillna("").to_dict(orient="records"),
        "preview": df.head(10).fillna("").to_dict(orient="records"),
    }


@app.post("/api/ask")
async def ask(req: AskRequest):
    if STATE["df"] is None:
        raise HTTPException(400, "Upload a dataset first.")

    df = STATE["df"]

    try:
        code = generate_code(req.question, STATE["profile_text"])
    except Exception as e:
        raise HTTPException(502, f"Couldn't reach the model: {e}")

    payload = {"question": req.question, "code": code, "error": None}

    try:
        result = run_safely(code, df)
    except UnsafeCodeError as e:
        payload["error"] = f"Rejected for safety: {e}"
    except TimeoutError as e:
        payload["error"] = str(e)
    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"

    if payload["error"] is None:
        kind, chart_payload = auto_chart(result)
        payload["kind"] = kind
        if kind == "figure":
            payload["figure"] = json.loads(chart_payload.to_json())
        elif kind == "table":
            payload["table"] = chart_payload.reset_index().fillna("").to_dict(orient="records")
        else:
            payload["metric"] = str(chart_payload)
        try:
            payload["explanation"] = explain_result(req.question, result)
        except Exception as e:
            payload["explanation"] = None
            payload["explanation_error"] = f"Couldn't generate an explanation: {e}"

    STATE["history"].append(payload)
    return payload


@app.get("/api/history")
async def history():
    return STATE["history"]


@app.get("/api/status")
async def status():
    return {
        "has_dataset": STATE["df"] is not None,
        "filename": STATE["filename"],
        "rows": len(STATE["df"]) if STATE["df"] is not None else 0,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")

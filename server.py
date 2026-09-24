import json

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from agent import answer_question
from dashboard import build_dashboard
from profiling import profile_dataframe, profile_to_text

app = FastAPI(title="AI Data Analyst Agent")


class NoCacheMiddleware(BaseHTTPMiddleware):
    """This is a local tool whose static files change often during
    development; without this, browsers can cache them by heuristic (no
    Cache-Control header is sent otherwise) and silently keep serving a
    stale UI after an update."""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response


app.add_middleware(NoCacheMiddleware)

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

    try:
        payload = answer_question(req.question, STATE["df"], STATE["profile_text"])
    except Exception as e:
        raise HTTPException(502, f"Couldn't reach the model: {e}")

    STATE["history"].append(payload)
    return payload


@app.get("/api/dashboard")
async def dashboard():
    if STATE["df"] is None:
        raise HTTPException(400, "Upload a dataset first.")
    charts = build_dashboard(STATE["df"])
    return [{"title": c["title"], "figure": json.loads(c["figure"].to_json())} for c in charts]


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

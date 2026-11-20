import sqlite3
import uuid

import secrets
from datetime import date
from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import os
from notify import notify
from fastapi.responses import FileResponse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from graph_agent import build_graph, start_state
from day2_loop import BASE, fetch_posting

security = HTTPBasic(auto_error=False)
APP_PASSWORD = os.getenv("APP_PASSWORD")             # set only on the private deploy
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "0"))     # 0 = no limit
usage = {"day": date.today(), "count": 0}


def check_password(creds: HTTPBasicCredentials | None = Depends(security)):
    if not APP_PASSWORD:
        return                                        # demo: no password
    if creds is None or not secrets.compare_digest(creds.password, APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Wrong password",
                            headers={"WWW-Authenticate": "Basic"})


def check_limit():
    if not DAILY_LIMIT:
        return
    today = date.today()
    if usage["day"] != today:                         # new day: reset the counter
        usage["day"], usage["count"] = today, 0
    if usage["count"] >= DAILY_LIMIT:
        raise HTTPException(status_code=429, detail="Daily demo limit reached. Try again tomorrow.")
    usage["count"] += 1


app = FastAPI(dependencies=[Depends(check_password)])  # password guards EVERY route

DB_PATH = os.getenv("DB_PATH", str(BASE / "checkpoints.db"))
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
agent = build_graph(fetch_fn=fetch_posting, with_review=True,
                    checkpointer=SqliteSaver(conn))


class AssessRequest(BaseModel):
    company: str
    job_id: str


class ReviewRequest(BaseModel):
    thread_id: str
    decision: str
    note: str = ""

@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")

@app.post("/assess", dependencies=[Depends(check_limit)])
def assess(req: AssessRequest):
    thread_id = str(uuid.uuid4())                           # blank 1: a new, unique ticket
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(start_state(req.company, req.job_id), config)

    if "__interrupt__" not in result:                  # ended in end_error: no pause happened
        return {"thread_id": thread_id, "status": "error", "verdict": result["verdict"]}

    paused = result["__interrupt__"][0].value
    notify(paused, thread_id)
    return {"thread_id": thread_id, "status": "waiting_for_review",
            **paused, "assessment": result["assessment"]}


@app.post("/review")
def review(req: ReviewRequest):
    if req.decision not in ["approve", "fit", "not fit"]:
        raise HTTPException(status_code=400, detail="decision must be approve, fit, or not fit")

    config = {"configurable": {"thread_id": req.thread_id}}     # blank 2: which ticket?
    snapshot = agent.get_state(config)
    if not snapshot.next:                        # nothing is waiting on this ticket
        raise HTTPException(status_code=404, detail="No paused job with that ticket")
    result = agent.invoke(Command(resume={"decision": req.decision, "note": req.note}), config)
    return {"verdict": result["verdict"], "score": result["score"],
            "human_decision": result["human_decision"], "human_note": result["human_note"]}
import sqlite3
import uuid
from notify import notify

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from graph_agent import build_graph, start_state
from day2_loop import BASE, fetch_posting

app = FastAPI()
conn = sqlite3.connect(BASE / "checkpoints.db", check_same_thread=False)
agent = build_graph(fetch_fn=fetch_posting, with_review=True,
                    checkpointer=SqliteSaver(conn))


class AssessRequest(BaseModel):
    company: str
    job_id: str


class ReviewRequest(BaseModel):
    thread_id: str
    decision: str
    note: str = ""


@app.post("/assess")
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
    result = agent.invoke(Command(resume={"decision": req.decision, "note": req.note}), config)
    return {"verdict": result["verdict"], "score": result["score"],
            "human_decision": result["human_decision"], "human_note": result["human_note"]}
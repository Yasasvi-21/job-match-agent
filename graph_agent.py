import copy
import re
import sqlite3
import sys
import time
import uuid
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from day2_loop import (client, tools, system_prompt, get_resume, fetch_posting,
                       load_saved_posting, get_sources, find_bad_quotes,
                       verify_evidence, score_assessment, BASE)

# ---------- settings ----------
MAX_CALLS = 7
BORDER_LOW = 60        # below this: confident "not fit"
BORDER_HIGH = 80       # at or above this: confident "fit"; in between: "borderline"
PRICE_IN = 0.10 / 1_000_000       # $ per input token
PRICE_OUT = 0.50 / 1_000_000      # $ per output token
MAX_TOOLS_PER_TURN = 3
SUSPICIOUS = re.compile(
    r"ignore (all )?(previous|prior) instructions|system note|"
    r"(automated|AI) screening|mark every requirement",
    re.IGNORECASE)


class AgentState(TypedDict):
    company: str
    job_id: str
    messages: list
    assessment: dict | None
    api_calls: int
    score: float | None
    failed_dealbreakers: list
    verdict: str | None
    human_decision: str | None
    human_note: str | None
    input_tokens: int
    output_tokens: int
    seconds: float
    injection_warning: bool


def call_claude(state: AgentState) -> dict:
    start = time.perf_counter()
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=16000,
        tools=tools,
        system=system_prompt,
        messages=state["messages"],
    )
    elapsed = time.perf_counter() - start
    reply = [block.model_dump(exclude_none=True) for block in response.content]

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": reply}],
        "api_calls": state["api_calls"] + 1,
        "input_tokens": state["input_tokens"] + response.usage.input_tokens,
        "output_tokens": state["output_tokens"] + response.usage.output_tokens,
        "seconds": state["seconds"] + elapsed,
    }


def route_after_claude(state: AgentState) -> str:
    last = state["messages"][-1]
    tool_calls = [b for b in last["content"] if b["type"] == "tool_use"]

    if not tool_calls:
        return "end_error"
    if any(b["name"] == "submit_assessment" for b in tool_calls):
        return "check_quotes"                 # always allow a submitted answer to be checked
    if state["api_calls"] >= MAX_CALLS:
        return "end_error"                    # out of budget
    return "run_tools"


def check_quotes(state: AgentState) -> dict:
    last = state["messages"][-1]
    block = next(b for b in last["content"]
                 if b["type"] == "tool_use" and b["name"] == "submit_assessment")

    bad = find_bad_quotes(block["input"], get_sources())
    if not bad:
        return {"assessment": block["input"]}

    feedback = ("These evidence quotes were not found word for word in the resume or candidate profile:\n"
                + "\n".join(bad)
                + "\nResubmit the assessment, copying each quote exactly from one line of the resume or profile.")
    return {"messages": state["messages"] + [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": block["id"], "content": feedback}]}]}


def route_after_check(state: AgentState) -> str:
    if state["assessment"] is not None:
        return "score_and_verdict"
    if state["api_calls"] >= MAX_CALLS:
        return "end_error"                    # stop retrying
    return "call_claude"


def score_and_verdict(state: AgentState) -> dict:
    assessment = copy.deepcopy(state["assessment"])      # work on a copy, not the saved state

    if len(assessment["requirements"]) == 0:
        return {"verdict": "error", "score": None, "failed_dealbreakers": []}

    for item in assessment["requirements"]:
        if re.search(r"\d+\+?\s*years", item["requirement"]):
            item["importance"] = "required"
            item["dealbreaker"] = True

    verify_evidence(assessment, get_sources())
    score = score_assessment(assessment)
    failed = [item["requirement"] for item in assessment["requirements"]
              if item["dealbreaker"] and item["status"] in ["missing", "unverified"]]

    if failed:
        verdict = "not fit"            # dealbreakers win, no matter the score
    elif score >= BORDER_HIGH:
        verdict = "fit"
    elif score >= BORDER_LOW:
        verdict = "borderline"         # too close to call: a human decides
    else:
        verdict = "not fit"

    return {"assessment": assessment, "score": score,
            "failed_dealbreakers": failed, "verdict": verdict}


def end_error(state: AgentState) -> dict:
    return {"verdict": "error"}


def human_review(state: AgentState) -> dict:
    answer = interrupt({
        "message": ("Agent is unsure. You decide." if state["verdict"] == "borderline"
                    else "Approve this verdict?"),
        "job": state["assessment"]["job_title"],
        "verdict": state["verdict"],
        "score": state["score"],
        "failed_dealbreakers": state["failed_dealbreakers"],
        "injection_warning": state["injection_warning"],
    })
    return {"human_decision": answer["decision"], "human_note": answer["note"]}


def build_graph(fetch_fn=fetch_posting, with_review=True, checkpointer=None):

    def run_tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        warning = state["injection_warning"]
        count = 0
        for b in last["content"]:
            if b["type"] != "tool_use":
                continue                     # skip text and thinking blocks
            count += 1
            if count > MAX_TOOLS_PER_TURN:                       # Defense 3: cap
                result = "Error: too many tool calls in one turn. Skipped."
            elif b["name"] == "get_resume":
                result = get_resume()
            elif b["name"] == "fetch_posting":
                text = fetch_fn(b["input"]["company"], b["input"]["job_id"])
                text = text.replace("</job_posting>", "")
                if SUSPICIOUS.search(text):                       # Defense 2: detect
                    warning = True
                result = f"<job_posting>\n{text}\n</job_posting>" # Defense 1: label
            else:
                result = f"Error: no tool named {b['name']}"
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": result})
        return {"messages": state["messages"] + [{"role": "user", "content": results}],
                "injection_warning": warning}

    graph = StateGraph(AgentState)
    graph.add_node("call_claude", call_claude)
    graph.add_node("run_tools", run_tools)
    graph.add_node("check_quotes", check_quotes)
    graph.add_node("score_and_verdict", score_and_verdict)
    graph.add_node("end_error", end_error)

    graph.add_edge(START, "call_claude")
    graph.add_conditional_edges("call_claude", route_after_claude,
                                ["run_tools", "check_quotes", "end_error"])
    graph.add_edge("run_tools", "call_claude")
    graph.add_conditional_edges("check_quotes", route_after_check,
                                ["score_and_verdict", "call_claude", "end_error"])

    if with_review:
        graph.add_node("human_review", human_review)
        graph.add_edge("score_and_verdict", "human_review")
        graph.add_edge("human_review", END)
    else:
        graph.add_edge("score_and_verdict", END)
    graph.add_edge("end_error", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def start_state(company, job_id):
    return {
        "company": company, "job_id": job_id,
        "messages": [{"role": "user",
                      "content": f"Is this job a good fit for me? Company: {company}, job id: {job_id}"}],
        "assessment": None, "api_calls": 0, "score": None,
        "failed_dealbreakers": [], "verdict": None,
        "human_decision": None, "human_note": None,
        "input_tokens": 0, "output_tokens": 0, "seconds": 0.0,
        "injection_warning": False,
    }


def run_graph_agent(company, job_id, fetch_fn=fetch_posting):
    agent = build_graph(fetch_fn=fetch_fn, with_review=False)        # unattended, no pause
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(start_state(company, job_id), config)
    cost = result["input_tokens"] * PRICE_IN + result["output_tokens"] * PRICE_OUT
    return {"verdict": result["verdict"], "score": result["score"],
            "failed_dealbreakers": result["failed_dealbreakers"],
            "api_calls": result["api_calls"], "assessment": result["assessment"],
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
            "seconds": round(result["seconds"], 1), "cost": cost,"injection_warning": result["injection_warning"],}


if __name__ == "__main__":
    mode, company, job_id = sys.argv[1], sys.argv[2], sys.argv[3]

    conn = sqlite3.connect(BASE / "checkpoints.db", check_same_thread=False)
    agent = build_graph(fetch_fn=load_saved_posting, with_review=True,
                        checkpointer=SqliteSaver(conn))
    config = {"configurable": {"thread_id": f"{company}-{job_id}"}}

    if mode == "start":
        result = agent.invoke(start_state(company, job_id), config)
        paused = result["__interrupt__"][0].value
        print("\n⏸  Paused for review:")
        for key, value in paused.items():
            print(f"   {key}: {value}")
        print(f"\nResume any time with:  python graph_agent.py resume {company} {job_id}")

    elif mode == "resume":
        decision = ""
        while decision not in ["approve", "fit", "not fit"]:
            decision = input("approve / fit / not fit: ").strip().lower()
        note = input("note (why): ").strip()
        result = agent.invoke(Command(resume={"decision": decision, "note": note}), config)
        print("\nAgent verdict:", result["verdict"], "| score:", result["score"])
        print("Your decision:", result["human_decision"], "|", result["human_note"])
        print("Total API calls:", result["api_calls"])

    elif mode == "test":
        print(run_graph_agent(company, job_id, fetch_fn=load_saved_posting))
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

import copy
import re

from day2_loop import (client, tools, system_prompt, get_resume, fetch_posting,
                       load_saved_posting, get_sources, find_bad_quotes,
                       verify_evidence, score_assessment)

class AgentState(TypedDict):
    company: str
    job_id: str
    messages: list
    assessment: dict | None          # hint: a dict, or None before Claude submits
    api_calls: int
    score: float | None
    failed_dealbreakers: list
    verdict: str | None

def call_claude(state: AgentState) -> dict:
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=16000,
        tools=tools,
        system=system_prompt,
        messages=state["messages"],
    )
    reply = [block.model_dump(exclude_none=True) for block in response.content]

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": reply}],
        "api_calls": state["api_calls"] + 1,
    }

MAX_CALLS = 7

def route_after_claude(state: AgentState) -> str:
    last = state["messages"][-1]
    tool_calls = [b for b in last["content"] if b["type"] == "tool_use"]

    if not tool_calls:
        return "end_error"
    if any(b["name"] == "submit_assessment" for b in tool_calls):
        return "check_quotes"                 # always allow a submitted answer to be checked
    if state["api_calls"] >= MAX_CALLS:
        return "end_error"                    # NEW: out of budget
    return "run_tools"

def check_quotes(state: AgentState) -> dict:
    last = state["messages"][-1]
    block = next(b for b in last["content"]
                 if b["type"] == "tool_use" and b["name"] == "submit_assessment")

    bad = find_bad_quotes(block["input"], get_sources())
    if not bad:
        return {"assessment": block["input"]}                 # hint: the filled-in form

    feedback = ("These evidence quotes were not found word for word in the resume or candidate profile:\n"
                + "\n".join(bad)
                + "\nResubmit the assessment, copying each quote exactly from one line of the resume or profile.")
    return {"messages": state["messages"] + [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": block["id"], "content": feedback}]}]}

def route_after_check(state: AgentState) -> str:
    if state["assessment"] is not None:
        return "score_and_verdict"
    if state["api_calls"] >= MAX_CALLS:
        return "end_error"                    # NEW: stop retrying
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
        verdict = "not fit"
    elif score >= 70:
        verdict = "fit"
    else:
        verdict = "not fit"

    return {"assessment": assessment, "score": score,
            "failed_dealbreakers": failed, "verdict": verdict}

def end_error(state: AgentState) -> dict:
    return {"verdict": "error"}

def build_graph(fetch_fn=fetch_posting):

    def run_tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        for b in last["content"]:
            if b["type"] != "tool_use":
                continue                     # skip text and thinking blocks
            if b["name"] == "get_resume":
                result = get_resume()
            elif b["name"] == "fetch_posting":
                result = fetch_fn(b["input"]["company"], b["input"]["job_id"])
            else:
                result = f"Error: no tool named {b['name']}"
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": result})

        return {"messages": state["messages"] + [{"role": "user", "content": results}]}
    graph = StateGraph(AgentState)
    graph.add_node("call_claude", call_claude)
    graph.add_node("run_tools", run_tools)
    graph.add_node("check_quotes", check_quotes)
    graph.add_node("score_and_verdict", score_and_verdict)
    graph.add_node("end_error", end_error)

        # the arrows
    graph.add_edge(START, "call_claude")                                      # where does every run begin?
    graph.add_conditional_edges("call_claude", route_after_claude,
                                ["run_tools", "check_quotes", "end_error"])  # a switch: 3 possible tracks
    graph.add_edge("run_tools", "call_claude")                      # plain arrow: always back to Claude
    graph.add_conditional_edges("check_quotes", route_after_check,
                                ["score_and_verdict", "call_claude", "end_error"]) # a switch: retry, score, or stop
    graph.add_edge("score_and_verdict", END)                        # where does a finished run go?
    graph.add_edge("end_error", END)
    

    return graph.compile()

if __name__ == "__main__":
    company, job_id = "stripe", "8144196"
    agent = build_graph(fetch_fn=load_saved_posting)

    result = agent.invoke({
        "company": company, "job_id": job_id,
        "messages": [{"role": "user",
                      "content": f"Is this job a good fit for me? Company: {company}, job id: {job_id}"}],
        "assessment": None, "api_calls": 0, "score": None,
        "failed_dealbreakers": [], "verdict": None,
    })
    print(result["verdict"], result["score"], result["failed_dealbreakers"], result["api_calls"])

    print(agent.get_graph().draw_mermaid())      # bonus: LangGraph draws your graph
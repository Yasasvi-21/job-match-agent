from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

import requests
import html
import re

from datetime import date
today = date.today().strftime("%B %d, %Y")
system_prompt = f"Today's date is {today}. ..."
client = Anthropic()  # Reads ANTHROPIC_API_KEY from your environment

def get_resume():
    with open("resume.txt", "r", encoding="utf-8") as file:
        return file.read()

def fetch_posting(company, job_id):
    url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs/{job_id}?content=true"

    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException:
        return "Error: could not reach Greenhouse (network problem)."

    if response.status_code == 404:
        return f"Error: no job {job_id} found for company '{company}'."
    if response.status_code != 200:          # hint: the code that means "OK"
        return f"Error: Greenhouse returned status {response.status_code}."

    data = response.json()

    # Clean the description: undo the escaping, then strip the HTML tags
    content = html.unescape(data["content"])
    content = re.sub(r"<[^>]+>", " ", content)

    return f"Title: {data['title']}\nLocation: {data['location']['name']}\n\n{content}"

def score_assessment(assessment):
    points = {"met": 1, "partially met": 0.5, "missing": 0}

    req_points, req_count = 0, 0
    pref_points, pref_count = 0, 0

    for item in assessment["requirements"]:
        value = points[item["status"]]                 # hint: look up this item's status
        if item["importance"] == "required":
            req_points += value
            req_count += 1
        else:
            pref_points+= value 
            pref_count += 1             # same thing for preferred
    req_score = req_points / req_count if req_count else 0
    pref_score = pref_points/pref_count if pref_count else 0                        # same pattern

    final = 100 * (0.7 * req_score + 0.3 * pref_score)
    return round(final, 1)

tools = [
    {
        "name": "get_resume",
        "description": "Retrieve the user's resume.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },

    {
    "name": "fetch_posting",
    "description": "Fetches one specific job posting from Greenhouse and returns its title, location, and full description as plain text. Use this when you need the requirements of a particular job. Requires the company's board name and the job id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {
                "type": "string",
                "description": "The company's Greenhouse board name, e.g. 'anthropic' or 'airbnb'"
            },
            "job_id": {
                "type": "string",
                "description": "The job's id number in Greenhouse, e.g. '4123456'"
            }
        },
        "required": ["company", "job_id"]
    }
    },

    {
    "name": "submit_assessment",
    "description": "Submit the final job-fit assessment. Call this once, after reading both the resume and the job posting.",
    "input_schema": {
        "type": "object",
        "properties": {
            "job_title": {"type": "string"},
            "requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement": {"type": "string"},
                        "importance": {"type": "string", "enum": ["required", "preferred"]},
                        "status": {"type": "string", "enum": ["met", "missing", "partially met"]},
                        "evidence": {"type": "string", "description": "Exact quote from the resume, or empty if missing"}
                    },
                    "required": ["requirement", "importance", "status", "evidence"]
                }
            }
        },
        "required": ["job_title", "requirements"]
    }
}
    
]


# (keep your imports, load_dotenv, client, get_resume(), and tools from day1_tools.py)

messages = [
    {"role": "user", "content": "Is this job a good fit for me? Company: anthropic, job id: 5456619008"}
]

max_steps = 5
assessment = None

for step in range(max_steps):
    print(f"\n--- API call {step + 1} ---")

    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=6000,
        tools=tools,
        messages=messages,          #  the history you're building
    )

    print("Stop reason:", response.stop_reason)

    # Case 1: Claude is done
    if response.stop_reason == "end_turn":      #  which stop_reason means "finished"?
        for block in response.content:
            if block.type == "text":
                print("FINAL ANSWER:\n", block.text)   #  Day 1, the field holding the words
        break

    if response.stop_reason == "max_tokens":  
        for block in response.content:
            if block.type == "text":
                print("FINAL ANSWER:\n", block.text)   #  Day 1, the field holding the words
        print("Stopped: hit the max_tokens limit without finishing.")
        break

    # Case 2: Claude wants a tool
    messages.append({"role": "assistant", "content": response.content})   # Claude's whole reply box

    for block in response.content:
        if block.type == "tool_use":            # the type of a tool request block
            if block.name == "submit_assessment":
                assessment = block.input
                break

            elif block.name == "get_resume":
                result = get_resume()
            elif block.name == "fetch_posting":
                result = fetch_posting(block.input["company"], block.input["job_id"])   # hint: pull company and job_id out of the field you just found
            else:
                result = f"Error: no tool named {block.name}"
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,   # the request id
                        "content": result,
                    }
                ],
            })
    if assessment is not None:
        break

else:
    print("Stopped: hit the step limit without finishing.")


print("Job:", assessment["job_title"])
for item in assessment["requirements"]:
    print(f"- [{item['status']}] ({item['importance']}) {item['requirement']}")
print("Score:", score_assessment(assessment))
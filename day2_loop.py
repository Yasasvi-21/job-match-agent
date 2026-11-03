from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()  # Reads ANTHROPIC_API_KEY from your environment

def get_resume():
    with open("resume.txt", "r", encoding="utf-8") as file:
        return file.read()

tools = [
    {
        "name": "get_resume",
        "description": "Retrieve the user's resume.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
]


# (keep your imports, load_dotenv, client, get_resume(), and tools from day1_tools.py)

messages = [
    {"role": "user", "content": "Please review my resume."}
]

max_steps = 5

for step in range(max_steps):
    print(f"\n--- API call {step + 1} ---")

    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1000,
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
            print("Claude asked for tool:", block.name)

            if block.name == "get_resume":
                result = get_resume()
            else:
                result = f"Error: no tool named {block.name}"           # run your function

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
else:
    print("Stopped: hit the step limit without finishing.")


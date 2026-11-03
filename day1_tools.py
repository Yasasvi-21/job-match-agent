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

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=500,
    tools=tools,
    messages=[
        {"role": "user", "content": "Please review my resume."}
    ],
)

print(response)
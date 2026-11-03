from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-haiku-5-5",    
    max_tokens= 200,
    messages=[
        {
            "role": "user",
            "content": "Explain how aeroplanes fly?"
        }
    ]
)       

print("Content:", response.content)
print("Stop Reason:", response.stop_reason)
print("Usage:", response.usage)

print("text: " + response.content[0].text)
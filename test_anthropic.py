import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Your API key
api_key = os.environ["ANTHROPIC_API_KEY"]  # Replace with your actual key
print(api_key)

# Initialize the client
client = anthropic.Anthropic(api_key=api_key)

# Create a simple message
message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ]
)

# Print the response
print(message.content)
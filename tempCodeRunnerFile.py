from dotenv import load_dotenv
from pathlib import Path
import os
from groq import Groq

# Anchor .env to this file's folder so it loads no matter where the script is run from
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)

model = "llama-3.3-70b-versatile"  # fixed: removed stray space
role = "user"
prompt = "Explain How Internet Works?"

message = {
    "role": role,
    "content": prompt
}
messages = [message]

response1 = client.chat.completions.create(model=model, messages=messages)
print(response1)

answer = response1.choices[0].message.content
print("Answer: ", answer)
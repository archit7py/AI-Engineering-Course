import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv() 
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)

model = "llama-3.3-70b-versatile"
role = "user"
prompt1 = "Do you know christiano ronaldo?"
prompt2 = "What is your favorite football player?"
prompt3 = "Who is the best player in the world?"

prompts = [prompt1, prompt2, prompt3]
for prompt in prompts:

    message = {
    "role": role,
    "content" : prompt
}
    messages = [message]
    response = client.chat.completions.create(model=model, messages=messages,max_tokens=500)
    usage = response.usage
    print(f"Prompt: {prompt} -->your tokens: {usage.prompt_tokens} completion_tokens: {usage.completion_tokens}, total_tokens: {usage.total_tokens} Finish reason: {response.choices[0].finish_reason}")


# print(response)

# print("###################################")

# answer = response.choices[0].message.content
# print(answer)




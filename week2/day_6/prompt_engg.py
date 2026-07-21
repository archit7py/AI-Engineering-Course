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


def llm_ans(prompt):
    messages = {
        "role": "user",
        "content": prompt
    }
    messages = [messages]
    response = client.chat.completions.create(model=model, messages=messages)
    ans = response.choices[0].message.content
    return ans

bad_prompt = """
#ROLE:
You are a support assistant at a mobile company 

#TASK:
You have to classify the user complaint into a category
Hardware and software issues are considered technical issues. Billing issues are related to payments, charges, or invoices. Return issues are related to product returns or exchanges.

#Constraints:
You have to classify the complaint into one of the following categories:
1. Billing Issue
2. Technical Issue
3. Return Issue

#OUTPUT FORMAT:
Your answer should be in one word only, either "Billing Issue", "Technical Issue", or "Return Issue".

#Example:
For instance if a user complaint is "I was charged twice for my last purchase", your output should be "Billing Issue".
For instance if a user complaint is "My Laptop is not working properly and keeps crashing", your output should be "Technical Issue".
For instance if a user complaint is "I want to return the product I purchased last week", your output should be "Return Issue".

#FALLBACK:
If you are unable to classify the complaint, respond with "Unable to classify".

This is a user complaint:
My marraige is not working.
"""

print(llm_ans(bad_prompt))

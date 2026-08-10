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

# Step 1 
knowledge_base = {
    "age": "Archit is 25 years old.",
    "net_worth": "Archit's net worth is estimated to be $1 million.",
    "hobbies": "Archit enjoys hiking, reading, and playing chess."
}

# Step 2 (retrieval function)

def retrieve_info(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["net_worth"]
    elif "hobbies" in question:
        return knowledge_base["hobbies"]
    else:
        return "I don't have information about that."

def ask_llm(question):
    context = retrieve_info(question)

    sys_prompt = f" Answer in one line only.Answer only based on the context provided. If the context does not contain the answer, respond with 'I don't know'.Context: {context}"  
    system_message = {
        "role": "system",
        "content": sys_prompt
    }
    message = {
        "role": "user",
        "content": question
    }
    messages = [system_message, message]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

question = " how rich is archit ?"
print(retrieve_info(question))
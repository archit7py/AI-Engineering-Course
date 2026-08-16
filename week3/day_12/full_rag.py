import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
import sys
from sentence_transformers import SentenceTransformer

model  = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

client = Groq(api_key=my_api_key)
groqmodel = "llama-3.3-70b-versatile"

documents = [
    "Employees are entitled to 24 paid leave days per year.",
    "The company provides 24 paid leave days annually for its employees.",
    "Leave days can be taken for personal reasons or medical emergencies.",
    "Employees can request additional leave days in special circumstances.",
    "Employees are encouraged to plan their leave days in advance to ensure smooth workflow.",
    "Employees can carry over a limited number of unused leave days to the next year, subject to company policy.",
    "Employees must submit leave requests through the company's HR portal and await approval from their supervisors.",

]

document_embeddings = model.encode(documents)
print(sys.getsizeof(document_embeddings))
# print(f"Document embeddings shape: {document_embeddings.shape}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query_embedding):
    scores = []
    for i, document in enumerate(document_embeddings):
        score = cosine_similarity(query_embedding, document)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[0] # Return the document with the highest similarity score

def ask_llm(question, context):
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
    response = client.chat.completions.create(model=groqmodel, messages=messages)
    answer = response.choices[0].message.content
    return answer




query = "How many times an employee can take leave  ?"
query_embedding = model.encode([query])
score, context = retrieve(query_embedding[0]) 
# print(f"Most relevant document: {context} with similarity score: {score}")
answer = ask_llm(query, context)
print(f"Question: {query}")
print(f"Answer: {answer}")


import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')# LENGTH  == 784 
text = "Machine Learning is fun"

# res = model.encode(text)
# embedding  = model.encode(text)
# print(f"Embedding shape: {embedding.shape}")
# print(f"Embedding: {embedding[:10]}...")

t1 = " There are 24 paid leave days available" 
t2 = " There are 24 cars in my garage"

v1 = model.encode(t1) # Print first 10 elements of the embedding
v2 = model.encode(t2)
print(f"Similarity: {cosine_similarity(v1, v2)}")

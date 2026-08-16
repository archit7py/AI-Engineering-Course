import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# INITIALIZE QDRANT CLIENT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Qdrant client initialized successfully.")


# ============================================================
# CREATE QDRANT COLLECTION
# ============================================================

COLLECTION_NAME = "Knowledge"

# all-MiniLM-L6-v2 produces 384-dimensional embeddings
EMBEDDING_SIZE = 384


# Delete existing collection if it exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection '{COLLECTION_NAME}'...")
    client.delete_collection(COLLECTION_NAME)


# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    )
)

print(f"Collection '{COLLECTION_NAME}' created successfully.")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance metric: COSINE")


# ============================================================
# LOAD KNOWLEDGE DOCUMENTS
# ============================================================

# Find knowledge.txt relative to this Python file
knowledge_file = Path(__file__).parent / "knowledge.txt"

with open(knowledge_file, "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(documents)} documents from 'knowledge.txt'.")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("Generating embeddings for documents...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedder.encode(documents)

print("Embeddings generated successfully.")
print(f"Generated embeddings shape: {len(embeddings)}")
print(f"Embedding Size: {len(embeddings[0])}")


# ============================================================
# CREATE QDRANT POINTS
# ============================================================

points = []

for i, embedding in enumerate(embeddings):

    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": documents[i]
            }
        )
    )


# ============================================================
# INSERT VECTORS INTO QDRANT
# ============================================================

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(
    f"Inserted {len(points)} points into "
    f"the '{COLLECTION_NAME}' collection."
)


# ============================================================
# SEMANTIC SEARCH FUNCTION
# ============================================================

def search(query, top_k=3):

    # Convert user's question into an embedding
    query_vector = embedder.encode(query).tolist()

    # Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return results


# ============================================================
# TEST SEMANTIC SEARCH
# ============================================================

query = "How many vacation days do I get?"

results = search(query, top_k=3)

print("\nSearch results for query:")

for result in results:

    print(
        f"Score: {result.score:.3f}, "
        f"Text: {result.payload['text']}"
    )

    print()


# ============================================================
# INITIALIZE GROQ CLIENT
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# ASK LLM FUNCTION
# ============================================================

def ask_llm(question, context):

    prompt = f"""
Answer the question based only on the context provided.

If the context does not contain the answer,
respond with "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return answer


# ============================================================
# RAG PIPELINE
# ============================================================

question = "How many vacation days do I get?"


# Step 1: Search Qdrant
results = search(
    question,
    top_k=3
)


# Step 2: Extract relevant text
context = "\n".join(
    result.payload["text"]
    for result in results
)


# Step 3: Send context + question to LLM
answer = ask_llm(
    question,
    context
)


# ============================================================
# FINAL ANSWER
# ============================================================

print("\nFinal Answer:")
print(answer)
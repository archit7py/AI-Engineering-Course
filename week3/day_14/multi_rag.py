import os
import json
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)
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
# CHECK ENVIRONMENT VARIABLES
# ============================================================

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing from .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is missing from .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")


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

    print(
        f"Deleting existing collection "
        f"'{COLLECTION_NAME}'..."
    )

    client.delete_collection(COLLECTION_NAME)


# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    )
)

print(
    f"Collection '{COLLECTION_NAME}' "
    f"created successfully."
)

print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance metric: COSINE")


# ============================================================
# CREATE PAYLOAD INDEX
# ============================================================

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

print(
    "Payload index for 'category' "
    "created successfully."
)


# ============================================================
# LOAD KNOWLEDGE DOCUMENTS
# ============================================================

# knowledge.json is in the same folder
# as multi_rag.py

knowledge_file = (
    Path(__file__).parent / "knowledge.json"
)


with open(
    knowledge_file,
    "r",
    encoding="utf-8"
) as f:

    documents = json.load(f)


print(
    f"Loaded {len(documents)} documents "
    f"from 'knowledge.json'."
)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print(
    "Generating embeddings for documents..."
)


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "Embeddings model loaded successfully."
)


# Extract text from every document
texts = [
    doc["text"]
    for doc in documents
]


# Generate embeddings
embeddings = model.encode(texts)


print(
    "Embeddings generated successfully."
)

print(
    f"Number of embeddings: {len(embeddings)}"
)

print(
    f"Embedding size: {len(embeddings[0])}"
)


# ============================================================
# CREATE QDRANT POINTS
# ============================================================

points = []


for i, embedding in enumerate(embeddings):

    point = PointStruct(

        id=i + 1,

        vector=embedding.tolist(),

        payload=documents[i]
    )

    points.append(point)


print(
    f"Created {len(points)} Qdrant points."
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
# SEMANTIC SEARCH
# ============================================================

def search(query, top_k=3):

    # Convert query into embedding
    query_vector = model.encode(
        query
    ).tolist()


    # Search Qdrant
    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        limit=top_k,

        with_payload=True

    ).points


    return results


# ============================================================
# SEMANTIC SEARCH WITH CATEGORY FILTER
# ============================================================

def search_with_filter(
    query,
    top_k=3,
    category=None
):

    # Convert query into embedding
    query_vector = model.encode(
        query
    ).tolist()


    # No filter by default
    filter_condition = None


    # Create category filter
    if category:

        filter_condition = Filter(

            must=[

                FieldCondition(

                    key="category",

                    match=MatchValue(
                        value=category
                    )

                )

            ]

        )


    # Search Qdrant
    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        limit=top_k,

        with_payload=True,

        query_filter=filter_condition

    ).points


    return results


# ============================================================
# TEST SEMANTIC SEARCH
# ============================================================

query = "How many vacation days do I get?"


# Search without category filter
results = search(
    query,
    top_k=3
)


print("\n")
print("========================================")
print("SEMANTIC SEARCH RESULTS")
print("========================================")


for result in results:

    print(
        f"Score: {result.score:.3f}"
    )

    print(
        f"Category: "
        f"{result.payload.get('category', 'N/A')}"
    )

    print(
        f"Text: "
        f"{result.payload.get('text', '')}"
    )

    print()


# ============================================================
# INITIALIZE GROQ CLIENT
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


print(
    "Groq client initialized successfully."
)


# ============================================================
# ASK LLM FUNCTION
# ============================================================

def ask_llm(question, context):

    prompt = f"""
Answer the question based only on the context provided.

If the context does not contain the answer,
respond with "I don't know".

Do not use outside knowledge.
Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""


    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-120b",

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


# ------------------------------------------------------------
# STEP 1: RETRIEVE RELEVANT DOCUMENTS
# ------------------------------------------------------------

print("\n")
print("Retrieving relevant documents...")


results = search(
    question,
    top_k=3
)


# ------------------------------------------------------------
# STEP 2: EXTRACT CONTEXT
# ------------------------------------------------------------

context = "\n\n".join(

    result.payload.get("text", "")

    for result in results

)


print("\n")
print("========================================")
print("RETRIEVED CONTEXT")
print("========================================")

print(context)


# ------------------------------------------------------------
# STEP 3: SEND CONTEXT TO GROQ
# ------------------------------------------------------------

print("\n")
print("Sending context to Groq...")


answer = ask_llm(
    question,
    context
)


# ============================================================
# FINAL ANSWER
# ============================================================

print("\n")
print("========================================")
print("FINAL ANSWER")
print("========================================")

print(answer)


print("\n")
print("RAG pipeline completed successfully!")
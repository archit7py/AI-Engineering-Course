import os
import re
import ast

from dotenv import load_dotenv
from groq import Groq

# ----------------------------
# Load API Key
# ----------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")

client = Groq(api_key=api_key)

MODEL = "llama-3.3-70b-versatile"

# ----------------------------
# Tools
# ----------------------------

PRODUCTS = {
    "iphone17": 100000,
    "iphone15": 50000,
    "iphone16": 75000,
    "macbook air": 95000,
}


def get_product_price(product):
    """
    Returns the product price.
    """
    product = product.strip().lower()

    if product in PRODUCTS:
        return PRODUCTS[product]

    return "Product not found"


def calculator(expression):
    """
    Evaluates a simple mathematical expression.
    """

    try:
        return eval(expression, {"__builtins__": {}})
    except Exception:
        return "Calculation Error"


tools = {
    "get_product_price": get_product_price,
    "calculator": calculator,
}

# ----------------------------
# System Prompt
# ----------------------------

system_prompt = """
You are a Shopping Assistant.

You have access to these tools:

1. get_product_price(product)
2. calculator(expression)

Rules:

- Think before acting.
- Call ONLY ONE tool at a time.
- After writing an Action, STOP.
- Wait for the Observation.
- Never invent tool results.
- Continue reasoning after Observation.
- Finish with Final Answer.

Tool Call Format:

Thought: explain briefly
Action: get_product_price("IPhone17")

or

Thought: calculate remaining money
Action: calculator("10000-5000")

When done:

Final Answer: ...
"""

# ----------------------------
# Agent
# ----------------------------


def run_agent(question):

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    for step in range(10):

        print("\n" + "=" * 50)
        print(f"STEP {step+1}")
        print("=" * 50)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
        )

        answer = response.choices[0].message.content

        print(answer)

        # ----------------------------
        # Final Answer
        # ----------------------------

        if "Final Answer:" in answer:
            print("\nAgent Finished.")
            return

        # ----------------------------
        # Extract Tool Call
        # ----------------------------

        match = re.search(
            r"Action\s*:\s*(\w+)\((.*?)\)",
            answer,
            re.DOTALL,
        )

        if not match:
            print("\nNo valid tool call found.")
            return

        tool_name = match.group(1)
        raw_argument = match.group(2).strip()

        # ----------------------------
        # Parse Tool Input
        # ----------------------------

        try:
            tool_input = ast.literal_eval(raw_argument)
        except Exception:
            tool_input = raw_argument

        # ----------------------------
        # Execute Tool
        # ----------------------------

        if tool_name not in tools:
            observation = f"Unknown Tool: {tool_name}"
        else:
            observation = tools[tool_name](tool_input)

        print("\nObservation:", observation)

        # ----------------------------
        # Update Conversation
        # ----------------------------

        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}",
            }
        )


# ----------------------------
# Test
# ----------------------------

question = """
I have 10000 rupees.

What is the price of iPhone17?

How much money will I have left?
"""

run_agent(question)
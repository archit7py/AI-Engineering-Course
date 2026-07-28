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
from pydantic import BaseModel
class Ticket(BaseModel):

    name: str
    address: str
    email: str
    issue: str
    contact_number: str

schema = Ticket.model_json_schema()
print(schema)
response_format = {

    "type" : "json_object"
}

system_prompt = f"""
Extract the personal information from the customer ticket strictly based on the schema and return it in the json output format.{schema}"""

message_system = {
    "role": "system",
    "content" : system_prompt
}
text = "Hello My name is Archit . Yesterday i drive a car in the rain . I have an iphone which is not working properly , My address is Delhi , My email is archit@gmail.com, My contact number is 98745"
prompt = f"""
This is a customer Ticket.Please extract the personal information from this.{text}
"""
message = {
    "role": role,
    "content" : prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model=model, messages=messages,temperature = 0,response_format=response_format)




answer = response.choices[0].message.content
print(answer)

import json 
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)

print(ticket.name)
print(ticket.address)
print(ticket.email)
print(ticket.issue)
print(ticket.contact_number)




import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

OPENAI_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(api_key=OPENAI_KEY)

response = client.responses.create(
    model="gpt-5-nano",
    instructions="Sen bir matematik öğretmenisin.",
    input="Derstesin herkese selam ver!"
)

print(response.output_text)


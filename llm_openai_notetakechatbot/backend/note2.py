import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(find_dotenv())

OPENAI_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(api_key=OPENAI_KEY)

class JsonFormat(BaseModel):
    name: str
    surname: str
    topic: str
    important_date: str

response = client.responses.parse(
    model="gpt-5-nano",
    instructions=f"Bugünün tarihi: {datetime.now().strftime('%Y-%m-%d')}. İletilen girdiyi istenen json formatına dönüştür.",
    input="Ahmet selam! Ben Mikail Karadeniz. Yarın sabah önemli bir iş toplantımız var onu hatırlatmak istedim.",
    text_format=JsonFormat
)

print(response.output_text)

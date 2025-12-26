# Chain of Thought (CoT): LLM'in problemi adım adım düşünerek çözmesini sağlayan bir prompting tekniğidir.
# Model, sonuca doğrudan atlamak yerine ara adımları göstererek daha doğru sonuçlar üretir.

import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(find_dotenv())

OPENAI_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(api_key=OPENAI_KEY)

class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

response = client.responses.parse(
    model="gpt-5-nano",
    input= [
        {
            "role": "system",
            "content": "Sen uzman bir matematikçisin. İletilen problemi adım adım açıklayarak sonuca ulaş"
        },
        {
            "role":"user",
            "content": "Nasıl çözerim? 2x^2 + 16 = 0 x=?"
        }
    ],
    text_format=MathReasoning
)

math_result = response.output_parsed

print("=" * 50)
print("ÇÖZÜM ADIMLARI")
print("=" * 50)

for i, step in enumerate(math_result.steps, 1):
    print(f"\nAdım {i}:")
    print(f"  Açıklama: {step.explanation}")
    print(f"  Sonuç: {step.output}")

print("\n" + "=" * 50)
print(f"SONUÇ: {math_result.final_answer}")
print("=" * 50)
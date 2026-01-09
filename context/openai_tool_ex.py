import os
import subprocess
import platform
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import json

load_dotenv(find_dotenv())

OPENAI_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(api_key=OPENAI_KEY)


def ping_site(url: str) -> str:
    """Belirtilen URL'e ping atar ve sonucu döndürür."""
    # URL'den domain'i çıkar (http:// veya https:// varsa kaldır)
    domain = url.replace("https://", "").replace("http://", "").split("/")[0]

    # İşletim sistemine göre ping komutu
    param = "-n" if platform.system().lower() == "windows" else "-c"

    try:
        result = subprocess.run(
            ["ping", param, "4", domain],
            capture_output=True,
            text=True,
            timeout=10
        )
        return f"Ping sonucu:\n{result.stdout}" if result.returncode == 0 else f"Ping başarısız:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"{domain} adresine ping zaman aşımına uğradı."
    except Exception as e:
        return f"Hata: {str(e)}"


tools = [
    {
        "type": "function",
        "name": "ping_site",
        "description": "Belirtilen URL'e ping atar ve sonucu döndürür.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Ping atılacak site URL'i (örn: google.com)"
                }
            },
            "required": ["url"]
        }
    }
]

input_list = [
    {"role": "system", "content": "Sen yardımcı bir asistansın."},
    {"role": "user", "content": "websiteme ulaşamıyorum birde sen dener misin? mikailkaradeniz.dev"}
]

print("=" * 50)
print("1. İLK İSTEK GÖNDERİLİYOR...")
print("=" * 50)

response = client.responses.create(
    model="gpt-5-nano",
    input=input_list,
    tools=tools
)

print(f"Model yanıtı: {response.output}")
print()

input_list += response.output

for item in response.output:
    if item.type == "function_call":
        if item.name == "ping_site":
            print("=" * 50)
            print("2. TOOL ÇAĞRISI TESPİT EDİLDİ")
            print("=" * 50)
            print(f"Fonksiyon: {item.name}")
            print(f"Argümanlar: {item.arguments}")
            print()

            print("3. FONKSİYON ÇALIŞTIRILIYOR...")
            ping_status = ping_site(**json.loads(item.arguments))
            print(f"Sonuç:\n{ping_status}")
            print()

            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"ping_status": ping_status})
            })

print("=" * 50)
print("4. TOOL SONUCU İLE FİNAL İSTEK GÖNDERİLİYOR...")
print("=" * 50)

final_response = client.responses.create(
    model="gpt-5-nano",
    input=input_list,
    tools=tools
)

print(final_response.output_text)
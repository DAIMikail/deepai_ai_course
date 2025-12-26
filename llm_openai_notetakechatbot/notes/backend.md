# Gerekli Modüller

```bash
uv add openai python-dotenv pydantic fastapi uvicorn
```

# Uygulama1

```python
# Gerekli kütüphaneleri import ediyoruz
import os
from dotenv import load_dotenv, find_dotenv  # .env dosyasını okumak için
from openai import OpenAI  # OpenAI API client

# .env dosyasını bul ve yükle (API anahtarını güvenli şekilde saklamak için)
load_dotenv(find_dotenv())

# .env dosyasından OPENAI_KEY değerini oku
OPENAI_KEY = os.getenv("OPENAI_KEY")

# OpenAI client'ını API anahtarıyla oluştur
client = OpenAI(api_key=OPENAI_KEY)

# API'ye istek gönder
response = client.responses.create(
    model="gpt-5-nano",  # Kullanılacak model
    instructions="Sen bir matematik öğretmenisin.",  # Sistem talimatı (rol tanımı)
    input="Derstesin herkese selam ver!"  # Kullanıcı girdisi
)

# Modelden gelen yanıtı ekrana yazdır
print(response.output_text)
```

# Uygulama2

```python
# Gerekli kütüphaneleri import ediyoruz
import os
from datetime import datetime  # Tarih işlemleri için
from dotenv import load_dotenv, find_dotenv  # .env dosyasını okumak için
from openai import OpenAI  # OpenAI API client
from pydantic import BaseModel  # Yapılandırılmış çıktı için şema tanımlama

# .env dosyasını bul ve yükle
load_dotenv(find_dotenv())

# .env dosyasından OPENAI_KEY değerini oku
OPENAI_KEY = os.getenv("OPENAI_KEY")

# OpenAI client'ını API anahtarıyla oluştur
client = OpenAI(api_key=OPENAI_KEY)

# Pydantic ile JSON çıktı formatını tanımla
class JsonFormat(BaseModel):
    name: str  # İsim alanı
    surname: str  # Soyisim alanı
    topic: str  # Konu alanı
    important_date: str  # Önemli tarih alanı

# API'ye yapılandırılmış çıktı isteği gönder (parse metodu)
response = client.responses.parse(
    model="gpt-5-nano",  # Kullanılacak model
    instructions=f"Bugünün tarihi: {datetime.now().strftime('%Y-%m-%d')}. İletilen girdiyi istenen json formatına dönüştür.",  # Dinamik tarihli sistem talimatı
    input="Ahmet selam! Ben Mikail Karadeniz. Yarın sabah önemli bir iş toplantımız var onu hatırlatmak istedim.",  # Kullanıcı girdisi
    text_format=JsonFormat  # Çıktı formatı şeması
)

# Modelden gelen yapılandırılmış yanıtı ekrana yazdır
print(response.output_text)
```

# Uygulama3 - Tool Kullanımı (Function Calling)

```python
import os
import subprocess
import platform
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import json

load_dotenv(find_dotenv())

OPENAI_KEY = os.getenv("OPENAI_KEY")

client = OpenAI(api_key=OPENAI_KEY)


# Tool olarak kullanılacak fonksiyon
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


# OpenAI tool şeması tanımı
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

# Mesaj listesi (system ve user rolleriyle)
input_list = [
    {"role": "system", "content": "Sen yardımcı bir asistansın."},
    {"role": "user", "content": "websiteme ulaşamıyorum birde sen dener misin? mikailkaradeniz.dev"}
]

print("=" * 50)
print("1. İLK İSTEK GÖNDERİLİYOR...")
print("=" * 50)

# İlk istek - model tool çağrısı yapabilir
response = client.responses.create(
    model="gpt-5-nano",
    input=input_list,
    tools=tools
)

print(f"Model yanıtı: {response.output}")
print()

# Model çıktısını input listesine ekle
input_list += response.output

# Tool çağrılarını işle
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
            # Fonksiyonu çalıştır
            ping_status = ping_site(**json.loads(item.arguments))
            print(f"Sonuç:\n{ping_status}")
            print()

            # Tool sonucunu input listesine ekle
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"ping_status": ping_status})
            })

print("=" * 50)
print("4. TOOL SONUCU İLE FİNAL İSTEK GÖNDERİLİYOR...")
print("=" * 50)

# Tool sonucuyla birlikte final istek
final_response = client.responses.create(
    model="gpt-5-nano",
    input=input_list,
    tools=tools
)

# Modelin kullanıcıya yanıtı
print(final_response.output_text)
```

# Uygulama4 - Tool Decorator

Python fonksiyonlarını otomatik olarak OpenAI tool şemasına çeviren decorator.

```python
import inspect
from typing import Callable, get_type_hints


def openai_tool(func: Callable) -> Callable:
    """
    Python fonksiyonunu OpenAI tool şemasına çevirir.
    Fonksiyon çalışmaya devam eder, şemaya func.schema ile erişilir.

    Kullanım:
        @openai_tool
        def get_weather(city: str, unit: str = "celsius") -> str:
            '''Belirtilen şehrin hava durumunu getirir.

            Args:
                city: Hava durumu sorgulanacak şehir adı
                unit: Sıcaklık birimi
            '''
            return f"{city}: 20°C"

        # Fonksiyonu çağır
        print(get_weather("İstanbul"))

        # Tool şemasına eriş
        tools = [get_weather.schema]
    """
    name = func.__name__
    doc = func.__doc__ or ""
    description = doc.split("\n")[0].strip() if doc else f"{name} fonksiyonu"

    # Type hints ve parametreleri al
    type_hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    sig = inspect.signature(func)
    param_descriptions = _parse_docstring_params(doc)

    # Properties ve required listesini oluştur
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, str)
        json_type = _python_type_to_json(param_type)

        properties[param_name] = {
            "type": json_type,
            "description": param_descriptions.get(param_name, f"{param_name} parametresi")
        }

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Şemayı fonksiyona attribute olarak ekle
    func.schema = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }

    return func


def _python_type_to_json(python_type) -> str:
    """Python tipini JSON Schema tipine çevirir."""
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    return type_mapping.get(python_type, "string")


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Docstring'den parametre açıklamalarını çıkarır."""
    params = {}
    if not docstring:
        return params

    lines = docstring.split("\n")
    in_args = False

    for line in lines:
        line = line.strip()
        if line.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if line.startswith("Returns:") or line.startswith("Raises:") or not line:
                if not line:
                    continue
                break
            if ":" in line:
                parts = line.split(":", 1)
                param_name = parts[0].strip()
                param_desc = parts[1].strip() if len(parts) > 1 else ""
                params[param_name] = param_desc

    return params
```

# Uygulama5 - FastAPI GET ve POST

```python
# FastAPI ile basit GET ve POST örneği

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Pydantic model - POST isteği için veri şeması
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True


# Basit bir veritabanı simülasyonu
items_db = []


# GET - Tüm itemleri listele
@app.get("/items")
def get_items():
    return {"items": items_db}


# GET - Tek bir item getir
@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id < len(items_db):
        return {"item": items_db[item_id]}
    return {"error": "Item bulunamadı"}


# POST - Yeni item ekle
@app.post("/items")
def create_item(item: Item):
    items_db.append(item.model_dump())
    return {"message": "Item eklendi", "item": item}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

# Uygulama5 - API Testi

```python
# note5.py API testi

import httpx

BASE_URL = "http://localhost:8000"


def test_api():
    print("=" * 50)
    print("1. POST - Yeni item ekleniyor...")
    print("=" * 50)

    item1 = {"name": "Laptop", "price": 999.99, "in_stock": True}
    response = httpx.post(f"{BASE_URL}/items", json=item1)
    print(f"Yanıt: {response.json()}")
    print()

    item2 = {"name": "Mouse", "price": 29.99}
    response = httpx.post(f"{BASE_URL}/items", json=item2)
    print(f"Yanıt: {response.json()}")
    print()

    print("=" * 50)
    print("2. GET - Tüm itemler listeleniyor...")
    print("=" * 50)

    response = httpx.get(f"{BASE_URL}/items")
    print(f"Yanıt: {response.json()}")
    print()

    print("=" * 50)
    print("3. GET - Tek item getiriliyor (id=0)...")
    print("=" * 50)

    response = httpx.get(f"{BASE_URL}/items/0")
    print(f"Yanıt: {response.json()}")
    print()

    print("=" * 50)
    print("4. GET - Olmayan item (id=99)...")
    print("=" * 50)

    response = httpx.get(f"{BASE_URL}/items/99")
    print(f"Yanıt: {response.json()}")


if __name__ == "__main__":
    test_api()
```

# Uygulama6 - Not Alma Chatbot Backend (main.py)

Tek `/chat` endpoint'i ve CRUD işlemleri tool olarak tanımlanmış tam bir chatbot backend'i.

## Adım 1: Import ve Kurulum

```python
import os
import json
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Frontend için CORS
from pydantic import BaseModel
from openai import OpenAI
from tool_decorator import openai_tool  # Önceden yazdığımız decorator

load_dotenv(find_dotenv())

app = FastAPI()

# CORS ayarları - Frontend'in API'ye erişebilmesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
```

## Adım 2: In-Memory Veritabanı ve Konuşma Geçmişi

```python
# In-memory veritabanı - program kapanınca sıfırlanır
notes_db = []
note_id_counter = 0

# Konuşma geçmişi - chatbot önceki mesajları hatırlar
conversation_history = []
```

## Adım 3: Tool Fonksiyonları (@openai_tool decorator ile)

```python
@openai_tool
def create_note(title: str, content: str) -> str:
    """Yeni bir not oluşturur.

    Args:
        title: Notun başlığı
        content: Notun içeriği
    """
    global note_id_counter
    note = {"id": note_id_counter, "title": title, "content": content}
    notes_db.append(note)
    note_id_counter += 1
    return json.dumps({"message": "Not oluşturuldu", "note": note}, ensure_ascii=False)


@openai_tool
def get_notes() -> str:
    """Tüm notları listeler."""
    return json.dumps({"notes": notes_db}, ensure_ascii=False)


@openai_tool
def get_note(note_id: int) -> str:
    """Belirtilen ID'ye sahip notu getirir.

    Args:
        note_id: Notun ID'si
    """
    for note in notes_db:
        if note["id"] == note_id:
            return json.dumps({"note": note}, ensure_ascii=False)
    return json.dumps({"error": "Not bulunamadı"}, ensure_ascii=False)


@openai_tool
def update_note(note_id: int, title: str, content: str) -> str:
    """Belirtilen notu günceller.

    Args:
        note_id: Güncellenecek notun ID'si
        title: Yeni başlık
        content: Yeni içerik
    """
    for note in notes_db:
        if note["id"] == note_id:
            note["title"] = title
            note["content"] = content
            return json.dumps({"message": "Not güncellendi", "note": note}, ensure_ascii=False)
    return json.dumps({"error": "Not bulunamadı"}, ensure_ascii=False)


@openai_tool
def delete_note(note_id: int) -> str:
    """Belirtilen notu siler.

    Args:
        note_id: Silinecek notun ID'si
    """
    for i, note in enumerate(notes_db):
        if note["id"] == note_id:
            deleted = notes_db.pop(i)
            return json.dumps({"message": "Not silindi", "note": deleted}, ensure_ascii=False)
    return json.dumps({"error": "Not bulunamadı"}, ensure_ascii=False)
```

## Adım 4: Tool Listesi ve Registry

```python
# Tool şemaları - decorator sayesinde otomatik oluşuyor
tools = [
    create_note.schema,
    get_notes.schema,
    get_note.schema,
    update_note.schema,
    delete_note.schema
]

# İsimden fonksiyona eşleme
tool_functions = {
    "create_note": create_note,
    "get_notes": get_notes,
    "get_note": get_note,
    "update_note": update_note,
    "delete_note": delete_note
}
```

## Adım 5: API Endpoint'leri

```python
# Frontend için notları döndüren endpoint
@app.get("/notes")
def api_get_notes():
    """Frontend için notları döndürür."""
    return {"notes": notes_db}
```

## Adım 6: Chat Endpoint (Ana İşlem)

```python
class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    print("=" * 50)
    print("1. KULLANICI MESAJI ALINDI")
    print("=" * 50)
    print(f"Mesaj: {request.message}")
    print()

    # System prompt + geçmiş konuşmalar + yeni mesaj
    input_list = [
        {"role": "system", "content": "Sen bir not alma asistanısın. Kullanıcının notlarını yönetmesine yardım et."}
    ]
    input_list += conversation_history
    input_list.append({"role": "user", "content": request.message})

    print("=" * 50)
    print("2. İLK İSTEK GÖNDERİLİYOR...")
    print("=" * 50)

    # İlk istek - model tool çağrısı yapabilir
    response = client.responses.create(
        model="gpt-5-nano",
        input=input_list,
        tools=tools
    )

    print(f"Model yanıtı: {response.output}")
    print()

    input_list += response.output

    # Tool çağrılarını işle
    for item in response.output:
        if item.type == "function_call":
            print("=" * 50)
            print("3. TOOL ÇAĞRISI TESPİT EDİLDİ")
            print("=" * 50)
            print(f"Fonksiyon: {item.name}")
            print(f"Argümanlar: {item.arguments}")
            print()

            func = tool_functions.get(item.name)
            if func:
                print("4. FONKSİYON ÇALIŞTIRILIYOR...")
                result = func(**json.loads(item.arguments))
                print(f"Sonuç: {result}")
                print()

                # Tool sonucunu input'a ekle
                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result
                })

    print("=" * 50)
    print("5. FİNAL İSTEK GÖNDERİLİYOR...")
    print("=" * 50)

    # Tool sonuçlarıyla final yanıt
    final_response = client.responses.create(
        model="gpt-5-nano",
        input=input_list,
        tools=tools
    )

    print(f"Final yanıt: {final_response.output_text}")
    print("=" * 50)
    print()

    # Konuşma geçmişine ekle
    conversation_history.append({"role": "user", "content": request.message})
    conversation_history.append({"role": "assistant", "content": final_response.output_text})

    print(f"Geçmiş uzunluğu: {len(conversation_history)} mesaj")
    print()

    return {"response": final_response.output_text}
```

## Adım 7: Uygulamayı Çalıştır

```python
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

## Çalıştırma

```bash
python main.py
```

## Test

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Alışveriş listesi başlığıyla bir not oluştur"}'
```

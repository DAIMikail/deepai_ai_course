# Not Alma Chatbot Backend

import os
import json
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from tool_decorator import openai_tool

load_dotenv(find_dotenv())

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# In-memory veritabanı
notes_db = []
note_id_counter = 0

# Konuşma geçmişi
conversation_history = []


# ============================================
# TOOL FONKSİYONLARI
# ============================================

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


# Tool listesi
tools = [
    create_note.schema,
    get_notes.schema,
    get_note.schema,
    update_note.schema,
    delete_note.schema
]

# Tool fonksiyonları registry
tool_functions = {
    "create_note": create_note,
    "get_notes": get_notes,
    "get_note": get_note,
    "update_note": update_note,
    "delete_note": delete_note
}


# ============================================
# API ENDPOİNTLERİ
# ============================================

@app.get("/notes")
def api_get_notes():
    """Frontend için notları döndürür."""
    return {"notes": notes_db}


# ============================================
# CHAT ENDPOİNT
# ============================================

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

                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result
                })

    print("=" * 50)
    print("5. FİNAL İSTEK GÖNDERİLİYOR...")
    print("=" * 50)

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


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

# pip install fastapi uvicorn python-dotenv openai

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import OpenAIClient
from context import ContextManager
from memory import MemoryManager
from summarizer import Summarizer
from embedding import bilgi_ara, get_embedder


app = FastAPI(title="TeknoElektronik Chatbot API")

# Embedding veritabanini baslat (lazy init)
# İlk arama yapıldığında otomatik yüklenecek
print("Embedding veritabani hazirlaniyor...")
get_embedder()
print("Embedding veritabani hazir!")

# Tool listesi
tools = [bilgi_ara]

# Context Manager
context_manager = ContextManager()

# Memory Manager - Konusma gecmisini yonet
memory_manager = MemoryManager(max_messages=10)

# OpenAI client - tools ile
client = OpenAIClient(tools=tools)

# Summarizer - Konusma ozetleyici
summarizer = Summarizer(client)

# CORS ayarlari
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System prompt
SYSTEM_PROMPT = """Sen TeknoElektronik A.S. musteri hizmetleri asistanisin.
Urunler, kargo, iade, garanti ve calisan bilgileri konularinda yardimci oluyorsun.
Kibar ve profesyonel bir dil kullan. Yanitlarin kisa ve oz olsun.

ONEMLI: Musteri sorularina yanit vermeden once mutlaka bilgi_ara aracini kullanarak
sirket dokumanlarindan ilgili bilgileri ara. Tahmin yapma, her zaman arama yap."""


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def root():
    return {"message": "TeknoElektronik Chatbot API"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Kullanici mesajina yanit verir."""
    session_id = request.session_id

    # Kullanici mesajini memory'ye ekle
    memory_manager.add_user_message(session_id, request.message)

    # 10 mesaj dolduğunda özet oluştur ve temizle
    if memory_manager.get_message_count(session_id) >= 10:
        history_to_summarize = memory_manager.get_history_as_text(session_id)
        summarizer.summarize(session_id, history_to_summarize)
        memory_manager.clear_session(session_id)
        print(f"Session: {session_id[:8]}... | OZET OLUSTURULDU, memory temizlendi")

    # Konusma gecmisini al
    history = memory_manager.get_history_as_text(session_id)

    # Ozeti al (varsa)
    summary = summarizer.get_summary(session_id)

    # Context, ozet, tools ve gecmis ile zenginlestirilmis prompt olustur
    enriched_prompt = context_manager.build_prompt_with_context(
        SYSTEM_PROMPT, history, summary, tools=tools
    )

    # Token sayisini yazdir
    print(f"\n{'#'*60}")
    print(f"YENI ISTEK")
    print(f"{'#'*60}")
    print(f"Session: {session_id[:8]}...")
    print(f"Mesaj: {request.message}")
    print(f"Gecmis mesaj sayisi: {memory_manager.get_message_count(session_id)}")
    print(f"Ozet: {'Var' if summary else 'Yok'}")
    print(f"Prompt token sayisi: {context_manager.get_last_prompt_token_count()}")
    print(f"{'#'*60}")

    response = client.chat(
        user_input=request.message,
        instructions=enriched_prompt
    )

    # Yaniti yazdir
    print(f"\n{'#'*60}")
    print(f"MODEL YANITI")
    print(f"{'#'*60}")
    response_preview = response if len(response) < 500 else response[:500] + "..."
    print(response_preview)
    print(f"{'#'*60}\n")

    # Asistan yanitini memory'ye ekle
    memory_manager.add_assistant_message(session_id, response)

    return ChatResponse(response=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

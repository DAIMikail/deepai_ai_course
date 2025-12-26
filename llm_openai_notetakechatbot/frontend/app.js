const API_URL = "http://localhost:8000";

// DOM elementleri
const notesList = document.getElementById("notes-list");
const refreshBtn = document.getElementById("refresh-btn");
const chatMessages = document.getElementById("chat-messages");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");

// Notları yükle
async function loadNotes() {
    try {
        const response = await fetch(`${API_URL}/notes`);
        const data = await response.json();

        if (data.notes.length === 0) {
            notesList.innerHTML = '<p class="empty-message">Henüz not yok</p>';
            return;
        }

        notesList.innerHTML = data.notes
            .map(note => `
                <div class="note-card">
                    <h3>${note.title}</h3>
                    <p>${note.content}</p>
                    <div class="note-id">ID: ${note.id}</div>
                </div>
            `)
            .join("");
    } catch (error) {
        notesList.innerHTML = '<p class="empty-message">Notlar yüklenemedi</p>';
        console.error("Notlar yüklenirken hata:", error);
    }
}

// Mesaj gönder
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // Kullanıcı mesajını ekle
    addMessage(message, "user");
    messageInput.value = "";
    sendBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        addMessage(data.response, "assistant");

        // Notları yenile (yeni not eklenmiş olabilir)
        loadNotes();
    } catch (error) {
        addMessage("Bir hata oluştu. Lütfen tekrar deneyin.", "assistant");
        console.error("Mesaj gönderilirken hata:", error);
    } finally {
        sendBtn.disabled = false;
    }
}

// Mesajı chat'e ekle
function addMessage(text, role) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Event listeners
refreshBtn.addEventListener("click", loadNotes);
sendBtn.addEventListener("click", sendMessage);
messageInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

// Sayfa yüklendiğinde notları getir
loadNotes();

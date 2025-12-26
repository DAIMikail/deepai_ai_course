# Not Alma Chatbot Frontend

Solda notların gösterildiği, sağda chat arayüzü olan basit bir frontend.

## Adım 1: HTML Yapısı (index.html)

```html
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Not Alma Chatbot</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <!-- Sol Panel - Notlar -->
        <div class="notes-panel">
            <div class="panel-header">
                <h2>Notlarım</h2>
                <button id="refresh-btn">Yenile</button>
            </div>
            <div id="notes-list" class="notes-list">
                <!-- Notlar buraya yüklenecek -->
            </div>
        </div>

        <!-- Sağ Panel - Chat -->
        <div class="chat-panel">
            <div class="panel-header">
                <h2>Chat</h2>
            </div>
            <div id="chat-messages" class="chat-messages">
                <!-- Mesajlar buraya eklenecek -->
            </div>
            <div class="chat-input">
                <input type="text" id="message-input" placeholder="Mesajınızı yazın...">
                <button id="send-btn">Gönder</button>
            </div>
        </div>
    </div>

    <script src="app.js"></script>
</body>
</html>
```

## Adım 2: CSS - Reset ve Genel Stiller (style.css)

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f5f5f5;
    height: 100vh;
}

.container {
    display: flex;
    height: 100vh;
}
```

## Adım 3: CSS - Sol Panel (Notlar)

```css
/* Sol Panel - Notlar */
.notes-panel {
    width: 35%;
    background-color: #fff;
    border-right: 1px solid #ddd;
    display: flex;
    flex-direction: column;
}

.panel-header {
    padding: 16px;
    border-bottom: 1px solid #ddd;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.panel-header h2 {
    font-size: 18px;
    color: #333;
}

#refresh-btn {
    padding: 8px 16px;
    background-color: #4a90d9;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

#refresh-btn:hover {
    background-color: #3a7bc8;
}

.notes-list {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
}

.note-card {
    background-color: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
}

.note-card h3 {
    font-size: 14px;
    color: #333;
    margin-bottom: 8px;
}

.note-card p {
    font-size: 13px;
    color: #666;
}

.note-card .note-id {
    font-size: 11px;
    color: #999;
    margin-top: 8px;
}

.empty-message {
    text-align: center;
    color: #999;
    padding: 20px;
}
```

## Adım 4: CSS - Sağ Panel (Chat)

```css
/* Sağ Panel - Chat */
.chat-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    background-color: #fff;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
}

.message {
    margin-bottom: 12px;
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 80%;
}

.message.user {
    background-color: #4a90d9;
    color: white;
    margin-left: auto;
}

.message.assistant {
    background-color: #e9e9e9;
    color: #333;
}

.chat-input {
    padding: 16px;
    border-top: 1px solid #ddd;
    display: flex;
    gap: 10px;
}

#message-input {
    flex: 1;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 14px;
}

#message-input:focus {
    outline: none;
    border-color: #4a90d9;
}

#send-btn {
    padding: 12px 24px;
    background-color: #4a90d9;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
}

#send-btn:hover {
    background-color: #3a7bc8;
}

#send-btn:disabled {
    background-color: #ccc;
    cursor: not-allowed;
}
```

## Adım 5: JavaScript - API URL ve DOM Elementleri (app.js)

```javascript
const API_URL = "http://localhost:8000";

// DOM elementleri
const notesList = document.getElementById("notes-list");
const refreshBtn = document.getElementById("refresh-btn");
const chatMessages = document.getElementById("chat-messages");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
```

## Adım 6: JavaScript - Notları Yükleme Fonksiyonu

```javascript
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
```

## Adım 7: JavaScript - Mesaj Gönderme Fonksiyonu

```javascript
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
```

## Adım 8: JavaScript - Event Listeners ve Başlatma

```javascript
// Event listeners
refreshBtn.addEventListener("click", loadNotes);
sendBtn.addEventListener("click", sendMessage);
messageInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

// Sayfa yüklendiğinde notları getir
loadNotes();
```

## Çalıştırma

```bash
# Frontend klasöründe basit HTTP server başlat
cd frontend
python -m http.server 3000
```

Tarayıcıda `http://localhost:3000` adresini aç.

## Kullanım

1. Sağdaki chat alanına mesaj yaz (örn: "Alışveriş listesi oluştur")
2. Enter veya Gönder butonuna tıkla
3. Chatbot notu oluşturur ve sol panelde görünür
4. "Yenile" butonu ile notları manuel yenileyebilirsin

"""Memory Manager - Konusma gecmisini yonetir."""

from collections import deque


class MemoryManager:
    """Session bazli konusma gecmisini yoneten sinif."""

    def __init__(self, max_messages: int = 10):
        """
        MemoryManager'i baslatir.

        Args:
            max_messages: Maksimum tutulacak mesaj sayisi (varsayilan: 10)
        """
        self.max_messages = max_messages
        self.sessions: dict[str, deque] = {}

    def _get_session(self, session_id: str) -> deque:
        """
        Session'i getirir, yoksa olusturur.

        Args:
            session_id: Oturum kimigi

        Returns:
            Session'a ait mesaj kuyrugu
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_messages)
        return self.sessions[session_id]

    def add_user_message(self, session_id: str, message: str) -> None:
        """
        Kullanici mesaji ekler.

        Args:
            session_id: Oturum kimligi
            message: Kullanici mesaji
        """
        session = self._get_session(session_id)
        session.append({"role": "USER", "content": message})

    def add_assistant_message(self, session_id: str, message: str) -> None:
        """
        Asistan mesaji ekler.

        Args:
            session_id: Oturum kimligi
            message: Asistan mesaji
        """
        session = self._get_session(session_id)
        session.append({"role": "ASISTAN", "content": message})

    def get_history(self, session_id: str) -> list[dict]:
        """
        Session'a ait konusma gecmisini getirir.

        Args:
            session_id: Oturum kimligi

        Returns:
            Mesaj listesi
        """
        session = self._get_session(session_id)
        return list(session)

    def get_history_as_text(self, session_id: str) -> str:
        """
        Session'a ait konusma gecmisini metin olarak getirir.

        Args:
            session_id: Oturum kimligi

        Returns:
            Formatlanmis konusma gecmisi
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        lines = []
        for msg in history:
            lines.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """
        Belirtilen session'i temizler.

        Args:
            session_id: Oturum kimligi
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def clear_all(self) -> None:
        """Tum session'lari temizler."""
        self.sessions.clear()

    def get_message_count(self, session_id: str) -> int:
        """
        Session'daki mesaj sayisini dondurur.

        Args:
            session_id: Oturum kimligi

        Returns:
            Mesaj sayisi
        """
        if session_id not in self.sessions:
            return 0
        return len(self.sessions[session_id])

    def session_exists(self, session_id: str) -> bool:
        """
        Session'in var olup olmadigini kontrol eder.

        Args:
            session_id: Oturum kimligi

        Returns:
            True eger session varsa
        """
        return session_id in self.sessions


if __name__ == "__main__":
    # Test
    mm = MemoryManager(max_messages=5)

    session_id = "test-session-123"

    mm.add_user_message(session_id, "Merhaba")
    mm.add_assistant_message(session_id, "Merhaba! Size nasil yardimci olabilirim?")
    mm.add_user_message(session_id, "Laptop fiyatlarini ogrenmek istiyorum")
    mm.add_assistant_message(session_id, "Laptop modellerimiz 15.000 TL'den basliyor.")

    print("Konusma gecmisi:")
    print(mm.get_history_as_text(session_id))
    print(f"\nMesaj sayisi: {mm.get_message_count(session_id)}")

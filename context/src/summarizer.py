"""Summarizer - Konusma gecmisini ozetler."""

from model import OpenAIClient


class Summarizer:
    """Konusma gecmisini ozetleyen sinif."""

    def __init__(self, client: OpenAIClient):
        """
        Summarizer'i baslatir.

        Args:
            client: OpenAI client instance
        """
        self.client = client
        self.summaries: dict[str, str] = {}  # session_id -> summary

    def summarize(self, session_id: str, history: str) -> str:
        """
        Konusma gecmisini ozetler. Onceki ozet varsa kumulatif olarak gunceller.

        Args:
            session_id: Oturum kimligi
            history: Ozetlenecek konusma gecmisi

        Returns:
            Ozetlenmis metin
        """
        # Onceki ozet varsa dahil et
        previous_summary = self.summaries.get(session_id, "")

        if previous_summary:
            content = f"Onceki ozet:\n{previous_summary}\n\nYeni konusmalar:\n{history}"
            prompt = """Asagidaki onceki ozet ve yeni konusmalari birlestirerek guncel bir ozet olustur.
Sadece onemli bilgileri (kullanicinin adi, sordugu urunler, aldigi bilgiler, alinan kararlar) iceren 3-4 cumlelik bir ozet olustur.
Ozeti "Onceki konusmada:" ile baslat."""
        else:
            content = history
            prompt = """Asagidaki konusma gecmisini kisa ve oz bir sekilde ozetle.
Sadece onemli bilgileri (kullanicinin adi, sordugu urunler, aldigi bilgiler) iceren 2-3 cumlelik bir ozet olustur.
Ozeti "Onceki konusmada:" ile baslat."""

        summary = self.client.chat(
            user_input=content,
            instructions=prompt
        )

        # Ozeti sakla
        self.summaries[session_id] = summary
        return summary

    def get_summary(self, session_id: str) -> str:
        """
        Session'a ait ozeti getirir.

        Args:
            session_id: Oturum kimligi

        Returns:
            Ozet metni veya bos string
        """
        return self.summaries.get(session_id, "")

    def has_summary(self, session_id: str) -> bool:
        """
        Session'in ozeti olup olmadigini kontrol eder.

        Args:
            session_id: Oturum kimligi

        Returns:
            True eger ozet varsa
        """
        return session_id in self.summaries

    def clear_summary(self, session_id: str) -> None:
        """
        Belirtilen session'in ozetini siler.

        Args:
            session_id: Oturum kimligi
        """
        if session_id in self.summaries:
            del self.summaries[session_id]


if __name__ == "__main__":
    # Test
    client = OpenAIClient()
    summarizer = Summarizer(client)

    test_history = """USER: Merhaba, benim adim Ahmet.
ASISTAN: Merhaba Ahmet! Size nasil yardimci olabilirim?
USER: Laptop almak istiyorum.
ASISTAN: Laptop modellerimiz arasinda ProBook X1 ve UltraSlim S5 bulunmaktadir.
USER: ProBook X1 fiyati nedir?
ASISTAN: ProBook X1 fiyati 25.000 TL'dir."""

    summary = summarizer.summarize("test-session", test_history)
    print(f"Ozet:\n{summary}")

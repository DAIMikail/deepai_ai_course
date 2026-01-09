"""Context Manager - Prompt'a eklenecek context'i yonetir."""

from pathlib import Path
from typing import Callable
from tokenizer import Tokenizer


class ContextManager:
    """Prompt'a eklenecek context'i yoneten sinif."""

    def __init__(self, docs_dir: str = None):
        """
        ContextManager'i baslatir.

        Args:
            docs_dir: Dokumanların bulundugu dizin (opsiyonel)
        """
        self.context = ""
        self.docs_dir = Path(docs_dir) if docs_dir else Path(__file__).parent.parent
        self.tokenizer = Tokenizer()
        self.last_prompt = ""

    def load_document(self, filename: str) -> str:
        """
        Belirtilen dokumani yukler.

        Args:
            filename: Dokumanin adi

        Returns:
            Dokumanin icerigi
        """
        doc_path = self.docs_dir / filename
        if doc_path.exists():
            return doc_path.read_text(encoding="utf-8")
        return ""

    def add_context(self, text: str) -> None:
        """
        Context'e metin ekler.

        Args:
            text: Eklenecek metin
        """
        if text:
            self.context += text + "\n\n"

    def add_document(self, filename: str) -> None:
        """
        Belirtilen dokumani context'e ekler.

        Args:
            filename: Dokumanin adi
        """
        content = self.load_document(filename)
        if content:
            self.add_context(f"## {filename}\n{content}")

    def get_context(self) -> str:
        """
        Mevcut context'i dondurur.

        Returns:
            Context metni
        """
        return self.context.strip()

    def clear_context(self) -> None:
        """Context'i temizler."""
        self.context = ""

    def add_tools_to_prompt(self, tools: list[Callable]) -> str:
        """
        Tool fonksiyonlarinin aciklamalarini prompt'a eklenecek formatta dondurur.

        Args:
            tools: @openai_tool decorator'u ile tanimlanmis fonksiyon listesi

        Returns:
            Tool aciklamalarini iceren metin
        """
        if not tools:
            return ""

        tool_descriptions = []
        tool_descriptions.append("# Kullanilabilir Araclar (Tools)")
        tool_descriptions.append("Asagidaki araclari kullanarak kullaniciya yardimci olabilirsin:\n")

        for func in tools:
            if hasattr(func, 'schema'):
                schema = func.schema
                name = schema.get('name', func.__name__)
                description = schema.get('description', 'Aciklama yok')
                params = schema.get('parameters', {}).get('properties', {})
                required = schema.get('parameters', {}).get('required', [])

                tool_descriptions.append(f"## {name}")
                tool_descriptions.append(f"**Aciklama:** {description}\n")

                if params:
                    tool_descriptions.append("**Parametreler:**")
                    for param_name, param_info in params.items():
                        param_type = param_info.get('type', 'string')
                        param_desc = param_info.get('description', '')
                        is_required = "(zorunlu)" if param_name in required else "(opsiyonel)"
                        tool_descriptions.append(f"- `{param_name}` ({param_type}) {is_required}: {param_desc}")
                    tool_descriptions.append("")

        return "\n".join(tool_descriptions)

    def build_prompt_with_context(
        self,
        system_prompt: str,
        history: str = "",
        summary: str = "",
        tools: list[Callable] = None
    ) -> str:
        """
        System prompt, context, tools, ozet ve konusma gecmisini birlestirir ve saklar.

        Args:
            system_prompt: Temel sistem prompt'u
            history: Konusma gecmisi (opsiyonel)
            summary: Onceki konusmalarin ozeti (opsiyonel)
            tools: @openai_tool decorator'u ile tanimlanmis fonksiyon listesi (opsiyonel)

        Returns:
            Context ve tool bilgileri ile zenginlestirilmis prompt
        """
        self.last_prompt = system_prompt

        if self.context:
            self.last_prompt += f"\n\n# Baglam Bilgileri\n{self.get_context()}"

        if tools:
            tool_info = self.add_tools_to_prompt(tools)
            if tool_info:
                self.last_prompt += f"\n\n{tool_info}"

        if summary:
            self.last_prompt += f"\n\n# Onceki Konusmalarin Ozeti\n{summary}"

        if history:
            self.last_prompt += f"\n\n# Konusma Gecmisi\n{history}"

        return self.last_prompt

    def get_last_prompt_token_count(self) -> int:
        """
        Son olusturulan prompt'un token sayisini dondurur.

        Returns:
            Token sayisi
        """
        return self.tokenizer.count_tokens(self.last_prompt)

    def get_last_prompt_details(self) -> dict:
        """
        Son olusturulan prompt hakkinda detayli bilgi dondurur.

        Returns:
            Token sayisi, karakter sayisi ve diger bilgileri iceren dict
        """
        return self.tokenizer.get_token_details(self.last_prompt)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace("/src/context.py", ""))
    from tool_decorator import openai_tool

    # Ornek tool tanimla
    @openai_tool
    def hava_durumu(sehir: str, birim: str = "celsius") -> str:
        """Belirtilen sehrin hava durumunu getirir.

        Args:
            sehir: Hava durumu sorgulanacak sehir adi
            birim: Sicaklik birimi (celsius veya fahrenheit)

        Returns:
            Hava durumu bilgisi
        """
        return f"{sehir}: 20°C"

    @openai_tool
    def hesapla(sayi1: int, sayi2: int, islem: str = "topla") -> str:
        """Iki sayi uzerinde matematiksel islem yapar.

        Args:
            sayi1: Birinci sayi
            sayi2: Ikinci sayi
            islem: Yapilacak islem

        Returns:
            Islem sonucu
        """
        return str(sayi1 + sayi2)

    # Test
    cm = ContextManager()
    cm.add_context("Bu bir test context'idir.")

    system_prompt = "Sen yardimci bir asistansin."
    tools = [hava_durumu, hesapla]

    prompt = cm.build_prompt_with_context(system_prompt, tools=tools)

    print(f"Prompt:\n{prompt}\n")
    print("=" * 50)
    print(f"Token sayisi: {cm.get_last_prompt_token_count()}")
    print(f"Detaylar: {cm.get_last_prompt_details()}")

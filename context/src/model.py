# pip install openai python-dotenv

import os
import json
from typing import Callable
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


class OpenAIClient:
    """OpenAI API ile iletisim kurmak icin kullanilan sinif."""

    def __init__(self, model: str = "gpt-4.1-nano", tools: list[Callable] = None):
        """
        OpenAI client'i baslatir.

        Args:
            model: Kullanilacak model adi (varsayilan: gpt-4.1-nano)
            tools: Kullanilacak tool fonksiyonlari listesi (@openai_tool decorator ile)
        """
        load_dotenv(find_dotenv())
        OPENAI_KEY = os.getenv("OPENAI_KEY")

        self.client = OpenAI(api_key=OPENAI_KEY)
        self.model = model

        # Tool'lari kaydet
        self.tool_functions = {}  # name -> function mapping
        self.tool_schemas = []    # OpenAI tool schemas

        if tools:
            for func in tools:
                if hasattr(func, 'schema'):
                    self.tool_schemas.append(func.schema)
                    self.tool_functions[func.__name__] = func

    def chat(self, user_input: str, instructions: str = None) -> str:
        """
        OpenAI'a istek gonderir ve yaniti dondurur.
        Tool cagrisi varsa otomatik olarak isler.

        Args:
            user_input: Kullanici girdisi
            instructions: Sistem talimatlari (opsiyonel)

        Returns:
            Model yaniti
        """
        # Mesaj listesi olustur
        input_list = []

        if instructions:
            input_list.append({"role": "system", "content": instructions})

        input_list.append({"role": "user", "content": user_input})

        # Ilk istegi gonder
        request_params = {
            "model": self.model,
            "input": input_list
        }

        if self.tool_schemas:
            request_params["tools"] = self.tool_schemas

        response = self.client.responses.create(**request_params)

        # Tool cagrisi dongusunu isle
        tool_call_count = 0
        while self._has_tool_calls(response.output):
            tool_call_count += 1
            print(f"\n{'='*50}")
            print(f"TOOL CAGRISI TESPIT EDILDI (Tur: {tool_call_count})")
            print(f"{'='*50}")

            # Tool cagrilarini isle
            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    print(f"  Tool: {item.name}")
                    print(f"  Argumanlar: {item.arguments}")

                    tool_result = self._execute_tool(item.name, item.arguments)

                    # Sonucu kisalt (cok uzunsa)
                    result_preview = str(tool_result)
                    if len(result_preview) > 300:
                        result_preview = result_preview[:300] + "..."
                    print(f"  Sonuc: {result_preview}")

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(tool_result, ensure_ascii=False)
                    })

            print(f"{'='*50}")
            print(f"Tool sonucu modele gonderiliyor...")
            print(f"{'='*50}\n")

            # Sonucu tekrar modele gonder
            request_params["input"] = input_list
            response = self.client.responses.create(**request_params)

        if tool_call_count > 0:
            print(f"Toplam {tool_call_count} tool cagrisi tamamlandi.")

        return response.output_text

    def _has_tool_calls(self, output: list) -> bool:
        """
        Yanit icinde tool cagrisi olup olmadigini kontrol eder.

        Args:
            output: Model yaniti

        Returns:
            Tool cagrisi varsa True
        """
        if not output:
            return False

        for item in output:
            if hasattr(item, 'type') and item.type == "function_call":
                return True
        return False

    def _execute_tool(self, tool_name: str, arguments: str) -> dict:
        """
        Belirtilen tool'u calistirir.

        Args:
            tool_name: Tool adi
            arguments: JSON formatinda argumanlar

        Returns:
            Tool sonucu
        """
        if tool_name not in self.tool_functions:
            return {"error": f"Tool bulunamadi: {tool_name}"}

        try:
            func = self.tool_functions[tool_name]
            args = json.loads(arguments)
            result = func(**args)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def get_tool_schemas(self) -> list[dict]:
        """
        Kayitli tool semalarini dondurur.

        Returns:
            Tool semalari listesi
        """
        return self.tool_schemas

    def get_tool_names(self) -> list[str]:
        """
        Kayitli tool isimlerini dondurur.

        Returns:
            Tool isimleri listesi
        """
        return list(self.tool_functions.keys())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace("/src/model.py", ""))
    from tool_decorator import openai_tool

    # Ornek tool tanimla
    @openai_tool
    def hesapla(sayi1: int, sayi2: int, islem: str = "topla") -> str:
        """Iki sayi uzerinde matematiksel islem yapar.

        Args:
            sayi1: Birinci sayi
            sayi2: Ikinci sayi
            islem: Yapilacak islem (topla, cikar, carp, bol)

        Returns:
            Islem sonucu
        """
        if islem == "topla":
            return f"{sayi1} + {sayi2} = {sayi1 + sayi2}"
        elif islem == "cikar":
            return f"{sayi1} - {sayi2} = {sayi1 - sayi2}"
        elif islem == "carp":
            return f"{sayi1} * {sayi2} = {sayi1 * sayi2}"
        elif islem == "bol":
            return f"{sayi1} / {sayi2} = {sayi1 / sayi2}"
        return "Bilinmeyen islem"

    # Test
    client = OpenAIClient(tools=[hesapla])
    print(f"Kayitli tool'lar: {client.get_tool_names()}")

    response = client.chat(
        user_input="25 ile 17'yi toplar misin?",
        instructions="Sen yardimci bir matematik asistanisin."
    )
    print(f"Yanit: {response}")

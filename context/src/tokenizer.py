# pip install tiktoken

import tiktoken
from pathlib import Path


class Tokenizer:
    """Metin veya belgeleri tokenize ederek token sayisini belirleyen sinif."""

    def __init__(self, encoding_name: str = "o200k_base"):
        """
        Tokenizer'i baslatir.

        Args:
            encoding_name: Tiktoken encoding adi (varsayilan: o200k_base - GPT-4.1-nano icin)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    def tokenize(self, text: str) -> list[int]:
        """
        Metni tokenlara ayirir.

        Args:
            text: Tokenize edilecek metin

        Returns:
            Token ID listesi
        """
        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """
        Token listesini metne cevirir.

        Args:
            tokens: Token ID listesi

        Returns:
            Cozumlenmis metin
        """
        return self.encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """
        Metindeki token sayisini dondurur.

        Args:
            text: Token sayisi hesaplanacak metin

        Returns:
            Token sayisi
        """
        return len(self.tokenize(text))

    def count_tokens_from_file(self, file_path: str) -> int:
        """
        Dosyadaki token sayisini dondurur.

        Args:
            file_path: Dosya yolu

        Returns:
            Token sayisi
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadi: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.count_tokens(content)

    def get_token_details(self, text: str) -> dict:
        """
        Metin hakkinda detayli token bilgisi dondurur.

        Args:
            text: Analiz edilecek metin

        Returns:
            Token sayisi, token listesi ve karakter sayisi iceren dict
        """
        tokens = self.tokenize(text)
        return {
            "token_count": len(tokens),
            "char_count": len(text),
            "tokens": tokens,
            "avg_chars_per_token": len(text) / len(tokens) if tokens else 0
        }


if __name__ == "__main__":
    # Test
    tokenizer = Tokenizer()

    # Ornek metin
    text = "Merhaba, bu bir test metnidir. Tiktoken ile token sayisi hesaplanacak."

    print(f"Encoding: {tokenizer.encoding_name}")
    print(f"Metin: {text}")
    print(f"Token sayisi: {tokenizer.count_tokens(text)}")

    details = tokenizer.get_token_details(text)
    print(f"Karakter sayisi: {details['char_count']}")
    print(f"Token basina ortalama karakter: {details['avg_chars_per_token']:.2f}")
    print(f"Tokens: {details['tokens']}")

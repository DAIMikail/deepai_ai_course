from tokenizer import Tokenizer
from pathlib import Path


def main():
    tokenizer = Tokenizer()

    # MD dosyalari
    base_path = Path(__file__).parent.parent
    md_files = [
        base_path / "employees.md",
        base_path / "product_catalogs.md",
        base_path / "cargo.md",
    ]

    print("=" * 50)
    print("TOKEN SAYILARI")
    print("=" * 50)

    total_tokens = 0

    for file_path in md_files:
        token_count = tokenizer.count_tokens_from_file(str(file_path))
        total_tokens += token_count
        print(f"{file_path.name}: {token_count:,} token")

    print("-" * 50)
    print(f"TOPLAM: {total_tokens:,} token")
    print("=" * 50)


if __name__ == "__main__":
    main()

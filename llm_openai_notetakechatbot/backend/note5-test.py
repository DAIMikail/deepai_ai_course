# note5.py API testi

import httpx

BASE_URL = "http://localhost:8000"


def test_api():
    print("=" * 50)
    print("1. POST - Yeni item ekleniyor...")
    print("=" * 50)

    item1 = {"name": "Laptop", "price": 999.99, "in_stock": True}
    response = httpx.post(f"{BASE_URL}/items", json=item1)
    print(f"Yanıt: {response.json()}")
    print()

    item2 = {"name": "Mouse", "price": 29.99}
    response = httpx.post(f"{BASE_URL}/items", json=item2)
    print(f"Yanıt: {response.json()}")
    print()

    print("=" * 50)
    print("2. GET - Tüm itemler listeleniyor...")
    print("=" * 50)

    response = httpx.get(f"{BASE_URL}/items")
    print(f"Yanıt: {response.json()}")
    print()

    print("=" * 50)
    print("3. GET - Tek item getiriliyor (id=0)...")
    print("=" * 50)

    response = httpx.get(f"{BASE_URL}/items/0")
    print(f"Yanıt: {response.json()}")
    print()

    print("=" * 50)
    print("4. GET - Olmayan item (id=99)...")
    print("=" * 50)

    response = httpx.get(f"{BASE_URL}/items/99")
    print(f"Yanıt: {response.json()}")


if __name__ == "__main__":
    test_api()

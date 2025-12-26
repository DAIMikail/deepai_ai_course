# FastAPI ile basit GET ve POST örneği

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Pydantic model - POST isteği için veri şeması
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True


# Basit bir veritabanı simülasyonu
items_db = []


# GET - Tüm itemleri listele
@app.get("/items")
def get_items():
    return {"items": items_db}


# GET - Tek bir item getir
@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id < len(items_db):
        return {"item": items_db[item_id]}
    return {"error": "Item bulunamadı"}


# POST - Yeni item ekle
@app.post("/items")
def create_item(item: Item):
    items_db.append(item.model_dump())
    return {"message": "Item eklendi", "item": item}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

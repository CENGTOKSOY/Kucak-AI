from pymongo import MongoClient
from datetime import datetime
from models.chat_model import Chat, Message


class ChatRepository:
    def __init__(self):
        self.client = MongoClient("mongodb_uri")
        self.db = self.client["kucak_ai_db"]
        self.chats = self.db["mother_chats"]

    async def create_chat(self, user_id: str) -> Chat:
        chat = {
            "user_id": user_id,
            "title": "Yeni Anne Sohbeti",
            "messages": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "chat_type": "pregnancy"  # veya "postpartum", "newborn_care" gibi
        }
        result = self.chats.insert_one(chat)
        return Chat(**{**chat, "id": str(result.inserted_id)})

    async def add_message(self, chat_id: str, message: Message):
        self.chats.update_one(
            {"_id": chat_id},
            {
                "$push": {"messages": message.dict()},
                "$set": {"updated_at": datetime.now()}
            }
        )

    async def get_mother_chats(self, user_id: str, chat_type: str = None):
        query = {"user_id": user_id}
        if chat_type:
            query["chat_type"] = chat_type
        return list(self.chats.find(query).sort("updated_at", -1))
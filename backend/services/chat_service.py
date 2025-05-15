from typing import List, Optional
from fastapi import HTTPException
import openai
import os
import uuid
from dotenv import load_dotenv
from ..models.chat_model import ChatSession, Message, MessageRole
from ..repositories.vector_repo import VectorRepository

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatService:
    def __init__(self):
        self.vector_repo = VectorRepository()
        self.pregnancy_system_prompt = """
        You are Kucak-AI's pregnancy specialist assistant. You must provide reliable and personalized health guidance that complies with WHO standards. Your answers:
        - Must be based on scientific evidence
        - Must not cause unnecessary anxiety to the mother
        - Must definitely refer to a doctor in emergencies
        - Must support pregnancy development week by week
        """

    async def generate_response(self, user_id: str, session_id: str, question: str,
                                pregnancy_week: Optional[int] = None) -> str:
        try:
            # 1. Vektör arama için gömme oluştur
            embedding = self._get_embedding(question)

            # 2. Hamilelik haftasına göre filtreleme
            filter = {"pregnancy_week": pregnancy_week} if pregnancy_week else None

            # 3. Vektör veritabanından ilgili içerikleri sorgula
            query_result = await self.vector_repo.query_health_data(
                vector=embedding,
                filter=filter
            )

            # 4. Bağlam oluştur
            context = self._build_context(query_result.matches)

            # 5. LLM'den yanıt al
            response = await self._call_llm(
                user_id=user_id,
                session_id=session_id,
                question=question,
                context=context,
                pregnancy_week=pregnancy_week
            )

            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Anne-bebek sağlığı asistanı hatası: {str(e)}")

    def _get_embedding(self, text: str) -> List[float]:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']

    def _build_context(self, matches: List[dict]) -> str:
        context = "Referans Bilgiler:\n"
        for match in matches:
            context += f"\n• Kaynak: {match['metadata']['source']}\n"
            context += f"İçerik: {match['metadata']['text']}\n"
            if 'pregnancy_week' in match['metadata']:
                context += f"İlgili Hafta: {match['metadata']['pregnancy_week']}. hafta\n"
        return context

    async def _call_llm(self, user_id: str, session_id: str, question: str, context: str,
                        pregnancy_week: Optional[int]) -> str:
        prompt = f"""
        Kullanıcı Profili:
        - Anne Adayı ID: {user_id}
        - Gebelik Haftası: {pregnancy_week or "Belirtilmemiş"}

        Bağlam:
        {context}

        Soru: {question}

        Lütfen yanıtınızı şu şekilde yapılandırın:
        1. Kısa ve net bir özet
        2. Detaylı açıklama
        3. İlgili hafta için özel öneriler (varsa)
        4. Acil durum uyarıları (gerekirse)
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.pregnancy_system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Daha tutarlı yanıtlar için
            max_tokens=1000
        )

        return response.choices[0].message['content']

    async def create_chat_session(self, user_id: str, title: Optional[str] = None,
                                  chat_type: str = "pregnancy") -> ChatSession:
        return ChatSession(
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            title=title or f"Yeni {self._get_chat_type_title(chat_type)} Sohbeti",
            chat_type=chat_type
        )

    def _get_chat_type_title(self, chat_type: str) -> str:
        titles = {
            "pregnancy": "Gebelik",
            "postpartum": "Doğum Sonrası",
            "newborn": "Yenidoğan Bakımı"
        }
        return titles.get(chat_type, "Danışmanlık")
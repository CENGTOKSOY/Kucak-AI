import openai
from typing import List
from models.chat_model import RAGResponse
from repositories.vector_repo import VectorRepository


class RAGService:
    def __init__(self):
        self.vector_repo = VectorRepository()
        self.embed_model = "text-embedding-ada-002"
        self.llm_model = "gpt-4"

    async def generate_response(self, question: str, chat_id: str, namespace: str = "mother_health",
                                **filters) -> RAGResponse:
        """
        Kucak-AI için özelleştirilmiş RAG servisi
        namespace: mother_health veya baby_development
        filters: pregnancy_week, baby_age_weeks gibi filtreler
        """
        try:
            # 1. Gömme vektörü oluştur
            query_embed = self._get_embedding(question)

            # 2. Vektör veritabanında arama yap
            results = await self.vector_repo.query_vectors(
                vector=query_embed,
                namespace=namespace,
                filter=filters
            )

            # 3. Bağlam dokümanı oluştur
            context = self._build_context(results.matches)

            # 4. LLM'den yanıt al
            response = await self._generate_llm_response(
                question=question,
                context=context,
                namespace=namespace,
                filters=filters
            )

            return RAGResponse(
                answer=response,
                references=[m.metadata['source'] for m in results.matches],
                suggested_actions=self._extract_suggested_actions(response)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Kucak-AI bilgi sistemi hatası: {str(e)}")

    def _get_embedding(self, text: str) -> List[float]:
        return openai.Embedding.create(
            input=[text],
            model=self.embed_model
        ).data[0].embedding

    def _build_context(self, matches) -> str:
        context = []
        for m in matches:
            source = m.metadata.get('source', 'Bilinmeyen Kaynak')
            text = m.metadata['text']
            context.append(f"📌 Kaynak: {source}\n{text}")

            # Hamilelik veya bebek haftası bilgisi ekle
            if 'pregnancy_week' in m.metadata:
                context.append(f"🔹 İlgili Gebelik Haftası: {m.metadata['pregnancy_week']}")
            elif 'baby_age_weeks' in m.metadata:
                context.append(f"👶 Bebek Yaşı: {m.metadata['baby_age_weeks']} haftalık")

        return "\n\n".join(context)

    async def _generate_llm_response(self, question: str, context: str, namespace: str, filters: dict) -> str:
        system_prompt = self._get_system_prompt(namespace, filters)

        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        return response.choices[0].message.content

    def _get_system_prompt(self, namespace: str, filters: dict) -> str:
        if namespace == "baby_development":
            baby_age = filters.get('baby_age_weeks', 'bilinmiyor')
            return f"""
            Sen Kucak-AI'nın yenidoğan bakım uzmanısın. {baby_age} haftalık bebekler için:
            - Bilimsel temelli gelişim bilgileri ver
            - Ebeveynlere pratik öneriler sun
            - Acil durum belirtilerinde doktora yönlendir
            - WHO'nun bebek bakımı kılavuzlarına atıf yap
            """
        else:
            pregnancy_week = filters.get('pregnancy_week', 'bilinmiyor')
            return f"""
            Sen Kucak-AI'nın gebelik uzmanısın. {pregnancy_week}. hafta için:
            - Annenin fiziksel ve duygusal değişimlerini açıkla
            - Bebek gelişimini haftaya özel anlat
            - Beslenme ve egzersiz önerileri ver
            - Risk belirtilerinde hemen sağlık kuruluşuna yönlendir
            """

    def _extract_suggested_actions(self, response: str) -> List[str]:
        # Yanıttan otomatik eylem önerileri çıkar
        actions = []
        if "doktora başvurun" in response.lower():
            actions.append("ACİL DURUM: En yakın sağlık kuruluşuna gitmelisiniz")
        if "beslenme" in response.lower():
            actions.append("ÖNERİ: Beslenme planınızı gözden geçirin")
        return actions
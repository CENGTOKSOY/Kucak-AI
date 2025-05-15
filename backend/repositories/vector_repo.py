import pinecone
from typing import List, Optional
from fastapi import HTTPException
import os
from dotenv import load_dotenv

load_dotenv()


class VectorRepository:
    def __init__(self):
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

        if not pinecone_api_key or not pinecone_env:
            raise ValueError("Pinecone credentials not configured")

        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        self.index_name = "kucak-ai-health"

        if self.index_name not in pinecone.list_indexes():
            self._create_index()

        self.index = pinecone.Index(self.index_name)

    def _create_index(self, dimension=1536, metric="cosine"):
        pinecone.create_index(
            name=self.index_name,
            dimension=dimension,
            metric=metric,
            pod_type="p1"
        )

    async def upsert_health_vectors(self, vectors: List[dict]):
        """Hamilelik ve doğum sonrası sağlık verilerini saklar"""
        try:
            return self.index.upsert(
                vectors=vectors,
                namespace="mother_health"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def query_health_data(self, vector: List[float], top_k: int = 5,
                              filter: Optional[dict] = None):
        """WHO standartlarına uygun sağlık önerilerini sorgular"""
        try:
            return self.index.query(
                vector=vector,
                top_k=top_k,
                namespace="mother_health",
                filter=filter,
                include_metadata=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def upsert_baby_vectors(self, vectors: List[dict]):
        """Bebek gelişim verilerini saklar"""
        try:
            return self.index.upsert(
                vectors=vectors,
                namespace="baby_development"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def query_baby_data(self, vector: List[float], top_k: int = 5,
                            baby_age_weeks: Optional[int] = None):
        """Bebek yaşına uygun gelişim önerilerini sorgular"""
        filter = {"age_weeks": baby_age_weeks} if baby_age_weeks else None
        try:
            return self.index.query(
                vector=vector,
                top_k=top_k,
                namespace="baby_development",
                filter=filter,
                include_metadata=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_vectors(self, ids: List[str], namespace: str = ""):
        try:
            return self.index.delete(ids=ids, namespace=namespace)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
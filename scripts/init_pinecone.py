import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_pinecone():
    """Kucak-AI için Pinecone vektör veritabanını başlatır ve gerekli indeksleri oluşturur"""
    try:
        # Pinecone kimlik bilgilerini kontrol et
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

        if not pinecone_api_key or not pinecone_env:
            raise ValueError("Pinecone kimlik bilgileri bulunamadı")

        # Pinecone'u başlat
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        print("✅ Pinecone başarıyla başlatıldı")

        # Anne sağlığı ve bebek gelişimi için iki ayrı indeks
        health_index = "kucak-ai-health"
        development_index = "kucak-ai-development"

        # İndeksleri oluştur veya kontrol et
        self._setup_index(health_index, "Anne sağlığı verileri için temel indeks")
        self._setup_index(development_index, "Bebek gelişim verileri için indeks")

        print("\n🔍 Mevcut indeksler:", pinecone.list_indexes())

    except Exception as e:
        print(f"❌ Pinecone başlatma hatası: {str(e)}")
        raise

def _setup_index(index_name: str, description: str):
    """Belirtilen isimde indeks oluşturur veya varlığını kontrol eder"""
    try:
        if index_name not in pinecone.list_indexes():
            print(f"\n📦 '{description}' indeksi oluşturuluyor: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding boyutu
                metric="cosine",
                pod_type="p1",
                metadata_config={"indexed": ["pregnancy_week", "baby_age_weeks"]}
            )
            print(f"✅ '{index_name}' indeksi başarıyla oluşturuldu")
        else:
            print(f"ℹ️ '{index_name}' indeksi zaten mevcut")
    except Exception as e:
        print(f"❌ '{index_name}' indeksi oluşturma hatası: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_pinecone()
from adapters.chroma_adapter import ChromaAdapter
from services.embedder_service import EmbedderService
from utils.logger import get_logger

logger = get_logger(__name__)

class RetrieverService:
    def __init__(self, embedder_service=None, retriever_adapter=None):
        self.embedder_service = embedder_service or EmbedderService()
        self.retriever_adapter = retriever_adapter or ChromaAdapter()
        logger.info("RetrieverService initialized")

    async def retrieve(self, query: str, top_k: int = 5):
        query_embedding = await self.embedder_service.embed_text(query)
        return self.retriever_adapter.search(query_embedding, top_k=top_k)

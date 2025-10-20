from services.embedder_service import EmbedderService
from adapters.chroma_adapter import ChromaAdapter
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorDatasetService:
    def __init__(self, embedder_service=None, chroma_adapter=None):
        self.embedder = embedder_service or EmbedderService()
        self.chroma = chroma_adapter or ChromaAdapter()
        
    async def create_dataset(self, documents: list[str]):
        embeddings = [await self.embedder.embed_text(doc) for doc in documents]
        ids = [str(i) for i in range(len(documents))]
        metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]

        self.chroma.add_documents(ids, embeddings, metadatas, documents)
        logger.info(f"Vector dataset created with {len(documents)} documents")

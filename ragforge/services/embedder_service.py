import logging
from .vertexAI.embedding_service import embed_text   

logger = logging.getLogger(__name__)

class EmbedderService:
    def __init__(self):
        logger.info("EmbedderService initialized with VertexAI embeddings")

    async def embed_text(self, text: str) -> list[float]:
        return await embed_text(text)

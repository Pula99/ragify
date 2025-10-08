from utils.logger import get_logger
from adapters.openai_adapter import OpenAIAdapter

logger = get_logger(__name__)

class EmbedderService:
    def __init__(self, adapter=None):
        self.adapter = adapter or OpenAIAdapter()
        logger.info("EmbedderService initialized")

    async def embed_text(self, text: str):
        embedding = await self.adapter.generate_embedding(text)
        return embedding

    async def embed_batch(self, texts: list[str]):
        return [await self.embed_text(t) for t in texts]

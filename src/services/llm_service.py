from adapters.openai_adapter import OpenAIAdapter
from adapters.vertex_ai_adapter import VertexAIAdapter
from utils.logger import get_logger

logger = get_logger(__name__)

class LLMService:
    def __init__(self, adapter=None):
        self.adapter = adapter or VertexAIAdapter()
        logger.info("LLMService initialized")

    async def generate_response(self, query: str, context: str):
        return await self.adapter.generate_text(query, context)

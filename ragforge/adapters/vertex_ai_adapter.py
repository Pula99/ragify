from interfaces.ai_model_adapter import IAIModelAdapter
from services.vertexAI.generate_response_service import generate_response
from services.vertexAI.embedding_service import embed_text
from utils.logger import get_logger

logger = get_logger(__name__)

class VertexAIAdapter(IAIModelAdapter):

    def __init__(self, config: dict):
        self.config = config
        logger.info(f"VertexAIAdapter config: {config}")

    async def generate_text(self, query: str, context: str = ""):
        try:
            response = generate_response(context, query,  self.config, )
            return response
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return f"[VertexAI error for prompt:"

    async def generate_embedding(self, text: str):
        try:
            embedding = await embed_text(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 512

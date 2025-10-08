from interfaces.ai_model_adapter import IAIModelAdapter
from config.vertex_ai.vertexai_configuration import load_vertexai_config
from src.services.vertexAI.generate_response_service import generate_response
from src.services.vertexAI.embedding_service import embed_text
from utils.logger import get_logger

logger = get_logger(__name__)

class VertexAIAdapter(IAIModelAdapter):

    def __init__(self, config: dict, model_name="gemini-1.5-pro"):
        load_vertexai_config()
        self.config = config
        self.model_name = model_name
        logger.info(f"VertexAIAdapter initialized with model: {model_name}")

    async def generate_text(self, query: str, prompt: str, context: str = ""):
        try:
            response = await generate_response(context, query, prompt, "", self.config)
            return response
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return f"[VertexAI error for prompt: {prompt}]"

    async def generate_embedding(self, text: str):
        try:
            embedding = await embed_text(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 512

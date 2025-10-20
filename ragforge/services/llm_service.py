import os
from adapters.openai_adapter import OpenAIAdapter
from adapters.vertex_ai_adapter import VertexAIAdapter
from utils.logger import get_logger
from dotenv import load_dotenv

logger = get_logger(__name__)

class LLMService:
    def __init__(self, adapter=None):
        load_dotenv()

        default_vertexAI_model_config = {
            "model_name": os.getenv("VERTEX_AI_MODEL_NAME"),
            "temperature": float(os.getenv("VERTEX_AI_TEMPERATURE")),
            "top_p":float(os.getenv("VERTEX_AI_TOP_P")),
            "top_k": int(os.getenv("VERTEX_AI_TOP_K")),
            "max_output_tokens": int(os.getenv("VERTEX_AI_MAX_OUTPUT_TOKENS")),
        }

        self.adapter = adapter or VertexAIAdapter(config=default_vertexAI_model_config)
        logger.info("LLMService initialized")

    async def generate_response(self, query: str, context: str):
        return await self.adapter.generate_text(query, context)

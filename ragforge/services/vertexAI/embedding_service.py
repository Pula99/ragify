import logging
import os
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def embed_text(text: str) -> list[float]:
    try:
        load_dotenv()
        vertex_text_embedding_model_name=os.getenv("VERTEX_TEXT_EMBEDDING_MODEL")
        embedder = VertexAIEmbeddings(model_name=vertex_text_embedding_model_name)
        embedding = embedder.embed_query(text)
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return []


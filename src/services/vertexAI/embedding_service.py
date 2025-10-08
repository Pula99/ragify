import logging
from langchain.embeddings import VertexAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedder = VertexAIEmbeddings(model_name="text-embedding-004")
logger.info("VertexAI embedder initialized with model: text-embedding-004")

async def embed_text(text: str) -> list[float]:
    try:
        embedding = embedder.embed_query(text)
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return []


from services.llm_service import LLMService
from services.retriever_service import RetrieverService
from utils.logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    """
    Orchestrates the end-to-end RAG process:
    - Retrieve relevant documents
    - Generate an AI response using LLM
    """
    def __init__(self, llm_service=None, retriever_service=None):
        self.llm_service = llm_service or LLMService()
        self.retriever_service = retriever_service or RetrieverService()
        logger.info("RAGPipeline initialized")

    async def run(self, query: str):
        logger.info(f"Running RAG pipeline for query: {query}")
        retrieved_docs = await self.retriever_service.retrieve(query)
        context = "\n".join(retrieved_docs["documents"][0])
        response = await self.llm_service.generate_response(query, context)
        return response

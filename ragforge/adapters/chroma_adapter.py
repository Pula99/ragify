import chromadb
from chromadb.utils import embedding_functions
from utils.logger import get_logger

logger = get_logger(__name__)

class ChromaAdapter:
    def __init__(self, collection_name="ragforge_store", persist_directory="ragforge_store"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
 
    def add_documents(self, ids, embeddings, metadatas, documents):
        self.collection.add(
            ids=ids, 
            embeddings=embeddings, 
            metadatas=metadatas, 
            documents=documents
        )
        logger.info(f"Added {len(documents)} documents to Chroma.")

    def search(self, query_embedding, top_k=5):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        logger.info(f"Retrieved {len(results['documents'][0])} documents.")
        return results

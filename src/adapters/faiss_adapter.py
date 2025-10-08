import faiss
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

class FAISSAdapter:
    def __init__(self, dimension=768):
        self.index = faiss.IndexFlatL2(dimension)
        self.docs = []
        logger.info(f"FAISSAdapter initialized (dim={dimension})")

    def add_embeddings(self, embeddings, documents):
        self.index.add(np.array(embeddings).astype('float32'))
        self.docs.extend(documents)
        logger.info(f"Added {len(documents)} documents to FAISS index.")

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        results = [self.docs[i] for i in indices[0]]
        logger.info(f"Retrieved {len(results)} documents from FAISS.")
        return results

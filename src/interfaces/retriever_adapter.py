from abc import ABC, abstractmethod

class IRetrieverAdapter(ABC):
    @abstractmethod
    def add_documents(self, ids, embeddings, metadatas, documents):
        pass

    @abstractmethod
    def search(self, query_embedding, top_k=5):
        pass

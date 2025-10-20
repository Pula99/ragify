from abc import ABC, abstractmethod

class IEmbedderAdapter(ABC):
    @abstractmethod
    async def generate_embedding(self, text: str):
        pass

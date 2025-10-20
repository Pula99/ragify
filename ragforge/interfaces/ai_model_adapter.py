from abc import ABC, abstractmethod

class IAIModelAdapter(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str, context: str = ""):
        pass

    @abstractmethod
    async def generate_embedding(self, text: str):
        pass

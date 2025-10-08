class RAGConfig:
    def __init__(self, model_provider="openai", retriever="chroma", embedder="openai"):
        self.model_provider = model_provider
        self.retriever = retriever
        self.embedder = embedder

    def summary(self):
        return {
            "model_provider": self.model_provider,
            "retriever": self.retriever,
            "embedder": self.embedder
        }

import os
import openai
from interfaces.ai_model_adapter import IAIModelAdapter
from utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIAdapter(IAIModelAdapter):
    def __init__(self, model_name="gpt-4-turbo", embedding_model="text-embedding-3-small"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        logger.info("OpenAIAdapter initialized")

    async def generate_text(self, prompt: str, context: str = ""):
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{context}\n\n{prompt}"}]
        response = await openai.ChatCompletion.acreate(model=self.model_name, messages=messages)
        return response.choices[0].message["content"]

    async def generate_embedding(self, text: str):
        response = await openai.Embedding.acreate(model=self.embedding_model, input=text)
        return response.data[0].embedding

import asyncio
from dotenv import load_dotenv

import os
from ragforge import RAGPipeline, load_vertexai_config, VectorDatasetService

async def main():
    load_dotenv()

    print("Vertex AI Project:", os.getenv("VERTEX_PROJECT_ID"))
    print("Vertex AI Region:", os.getenv("VERTEX_PROJECT_ID"))
    print("OpenAI API Key Loaded:", bool(os.getenv("VERTEX_PROJECT_ID")))

    load_vertexai_config()

    docs = [
        "Hello world",
        "Ragify is awesome",
        "This is a test document"
    ]
    vector_service = VectorDatasetService()
    await vector_service.create_dataset(docs)

    rag = RAGPipeline()

    query = "Explain the Adapter pattern in Ragify"
    response = await rag.run(query)

    print("Query response:", response)

if __name__ == "__main__":
    asyncio.run(main())

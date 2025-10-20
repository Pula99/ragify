import logging
from typing import List, Optional
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_response(context: str, query: str,  config: dict):

    prompt = f"""
        
        Context: {context}
        Question: {query}    
    """


    model = GenerativeModel(config.get("model_name"))
    generation_config = GenerationConfig(
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        top_k=config.get("top_k"),
        max_output_tokens=config.get("max_output_tokens"),
    )

    try:
        responses = model.generate_content(
            contents=[prompt],
            generation_config=generation_config,
        )   
        if not responses:
            logger.error("No response generated.")
            return {"error": "No response generated."}

        return responses.text

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, something went wrong."
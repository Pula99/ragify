import os
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
from dotenv import load_dotenv

def load_vertexai_config():
    
    load_dotenv()

    project_id = os.getenv("VERTEX_PROJECT_ID")
    location = os.getenv("VERTEX_LOCATION")

    service_account_info = {
        "type": "service_account",
        "project_id": project_id,
        "private_key_id": os.getenv("VERTEX_PRIVATE_KEY_ID"),
        "private_key": os.getenv("VERTEX_PRIVATE_KEY").replace("\\n", "\n"),
        "client_email": os.getenv("VERTEX_CLIENT_EMAIL"),
        "client_id": os.getenv("VERTEX_CLIENT_ID"),
        "auth_uri": os.getenv("VERTEX_AUTH_URI"),
        "token_uri": os.getenv("VERTEX_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("VERTEX_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("VERTEX_CLIENT_CERT_URL"),
        "universe_domain": os.getenv("VERTEX_UNIVERSE_DOMAIN"),
    }

    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    aiplatform.init(project=project_id, location=location, credentials=credentials)
    vertexai.init(project=project_id, location=location, credentials=credentials)

    print("✅ Vertex AI initialized successfully.")

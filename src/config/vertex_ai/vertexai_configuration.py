import os
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai

def load_vertexai_config():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'configs/vertex_ai/vertexai_config.json'
    
    credentials = service_account.Credentials.from_service_account_file('configs/vertex_ai/vertexai_config.json')

    aiplatform.init(
        project="project-qra-462518",
        location="us-central1",
        credentials=credentials
    )   

    vertexai.init(
        project="project-qra-462518",
        location="us-central1"
    )

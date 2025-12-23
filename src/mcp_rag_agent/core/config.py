import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Config(BaseSettings):
    """Configuration settings for the MCP RAG Agent."""
    # Application settings
    app_name: str = "MCP RAG Agent"
    app_version: str = "0.1.0"
    # Logging settings
    log_level: str = "INFO"
    debug: bool = False
    # Database settings
    db_url: str = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
    db_name: str = os.environ['MONGODB_ATLAS_DB_NAME']
    db_users_collection: str = os.environ.get("MONGODB_USERS_COLLECTION", "users")
    db_conversations_collection: str = os.environ.get("MONGODB_CONVERSATIONS_COLLECTION", "conversations")
    db_messages_collection: str = os.environ.get("MONGODB_MESSAGES_COLLECTION", "messages")
    db_documents_collection: str = os.environ.get("MONGODB_DOCUMENTS_COLLECTION", "documents")
    db_vector_collection: str = os.environ.get("MONGODB_VECTOR_COLLECTION", "vectors")
    db_vector_index_name: str = os.environ.get("MONGODB_VECTOR_INDEX_NAME", "vector_index")
    # Model settings
    model_api_key: str = os.environ['OPENAI_API_KEY']
    embedding_model: str = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    embedding_dimension: int = int(os.environ.get("EMBEDDING_DIMENSION", "256"))
    text_model: str = os.environ.get("TEXT_MODEL_NAME", "gpt-4.1")
    text_generation_kwargs: dict = {
        "max_tokens": 2048,
        "temperature": 0,
        "top_p": 0.1,
    }
    evaluation_model: str = os.environ.get("EVALUATION_MODEL_NAME", "gpt-4o-mini")
    # MCP server settings
    mcp_name: str = os.environ.get("MCP_SERVER_NAME", "mongodb-semantic-search")
    mcp_host: str = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
    mcp_port: int = int(os.environ.get("MCP_SERVER_PORT", "8000"))
    # Search settings
    semantic_weight: float = float(os.environ.get("SEMANTIC_WEIGHT", "0.7"))  # 0.7 = 70% semantic, 30% keyword
    # Feature flags
    ff_mcp_server: bool = os.environ.get("FEATURE_FLAG_MCPSERVER_ENABLED", "false").lower() == "true"
    ff_web_search: bool = os.environ.get("FEATURE_FLAG_WEBSEARCH_ENABLED", "false").lower() == "true"
    # Evaluation settings
    ingested_doc_dir: str = os.environ.get("INGESTED_DOC_DIRECTORY", "./data/ingested_documents")
    evaluation_doc_dir: str = os.environ.get("EVALUATION_DOC_DIRECTORY", "./data/evaluation_documents")
    # Additional settings can be added here as needed

config = Config()

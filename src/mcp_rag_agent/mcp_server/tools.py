"""MCP server for RAG agent with hybrid search capabilities."""
import logging

from mcp_rag_agent.core.config import config
from mcp_rag_agent.core.log_setup import setup_logging
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator
from mcp_rag_agent.embeddings.hybrid_search import HybridSearch

setup_logging()
logger = logging.getLogger("Retriever")


# Initialize MongoDB client
logger.info("Initializing MongoDB client...")
mongo_client = MongoDBClient(uri=config.db_url, database_name=config.db_name)
mongo_client.connect()

# Initialize embedding generator
logger.info("Initializing embedding generator...")
embedding_generator = EmbeddingGenerator(
    api_key=config.model_api_key,
    model=config.embedding_model,
    dimensions=config.embedding_dimension
)

# Initialize hybrid search
logger.info("Initializing hybrid search...")
hybrid_search = HybridSearch(
    mongo_client=mongo_client,
    embedding_generator=embedding_generator,
    default_collection=config.db_vector_collection,
    default_vector_index=config.db_vector_index_name,
    default_text_index="text_index"
)

async def search_documents(
    query: str,
    top_k: int = 3
) -> list[dict]:
    """
    Perform hybrid search on indexed documents using vector and text search.

    This tool searches through the document collection using hybrid search that
    combines semantic similarity (vector embeddings) and keyword matching (text search)
    using Reciprocal Rank Fusion (RRF). This provides the best of both worlds:
    conceptual understanding from embeddings and precise keyword matching from text search.
    The results can be used to ground the user's answer with the most relevant documents.

    Args:
        query (str): The search query text to find relevant documents.
        top_k (int, optional): The maximum number of results to return.
            Defaults to 3.

    Returns:
        list[dict]: A list of matching documents, ordered by RRF score (relevance).
    """
    logger.info(f"Hybrid search started: {query}")

    results = await hybrid_search.search(
        query=query,
        limit=top_k,
        semantic_weight=config.semantic_weight
    )

    logger.info(f"Found {len(results)} relevant documents")

    return results

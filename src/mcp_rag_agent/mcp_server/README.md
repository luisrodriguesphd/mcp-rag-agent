# MCP Server Module

The MCP (Model Context Protocol) server module provides a standardized interface for hybrid search and RAG (Retrieval-Augmented Generation) capabilities. It exposes tools and prompts that can be consumed by LangChain agents and other MCP-compatible clients.

## Overview

This module implements an MCP server that:
- Exposes a `search_documents` tool for **hybrid search** combining vector similarity and keyword matching
- Provides a `grounded_qa_prompt` for ensuring factual, context-based responses
- Integrates with MongoDB for vector storage, text indexing, and retrieval
- Uses OpenAI embeddings for semantic similarity matching
- Employs **Reciprocal Rank Fusion (RRF)** to combine search results
- Communicates via stdio transport for local integration

## Architecture

```
mcp_server/
└── server.py          # MCP server implementation with tools and prompts
```

## What is MCP?

The **Model Context Protocol (MCP)** is a standardized protocol for connecting AI models with external tools and data sources. It enables:

- **Tool Exposure**: Make custom functions available to language models
- **Resource Sharing**: Provide access to data, APIs, and services
- **Prompt Templates**: Share reusable prompt patterns
- **Transport Flexibility**: Support for stdio, HTTP, and other transports

### Benefits

1. **Standardization**: Common protocol for tool integration across frameworks
2. **Modularity**: Separate tool implementation from agent logic
3. **Reusability**: Tools can be used by multiple agents/applications
4. **Isolation**: Server runs in separate process, improving reliability

## Components

### `server.py`

The main MCP server implementation using FastMCP framework.

#### Initialization

```python
# MCP Server
mcp = FastMCP(name=config.mcp_name)

# MongoDB Client
mongo_client = MongoDBClient(uri=config.db_url, database_name=config.db_name)

# Embedding Generator
embedding_generator = EmbeddingGenerator(
    api_key=config.model_api_key,
    model=config.embedding_model,
    dimensions=config.embedding_dimension
)

# Hybrid Search
hybrid_search = HybridSearch(
    mongo_client=mongo_client,
    embedding_generator=embedding_generator,
    default_collection=config.db_vector_collection,
    default_vector_index=config.db_vector_index_name,
    default_text_index="text_index"
)
```

#### Tools

##### `search_documents`

Performs **hybrid search** on indexed documents combining vector similarity and keyword matching using Reciprocal Rank Fusion (RRF).

**Signature:**
```python
async def search_documents(query: str, top_k: int = 3) -> list[dict]
```

**Parameters:**
- `query` (str): The search query text to find relevant documents
- `top_k` (int, optional): Maximum number of results to return (default: 3)

**Returns:**
- `list[dict]`: List of matching documents ordered by relevance

**How It Works:**
1. Receives user query text
2. Generates embedding vector for the query
3. **Parallel Execution**:
   - Performs vector similarity search in MongoDB (semantic matching)
   - Performs full-text search in MongoDB (keyword matching)
4. **RRF Fusion**: Combines both result sets using Reciprocal Rank Fusion
5. Returns top-k most relevant documents ordered by combined RRF score

**Benefits of Hybrid Search:**
- **Semantic Understanding**: Finds conceptually related content via embeddings
- **Keyword Precision**: Captures exact terms and technical language
- **Best of Both Worlds**: RRF algorithm balances both approaches
- **Better Recall**: Finds documents missed by single-method searches

**Example Usage (via Agent):**
```python
# The agent automatically calls this tool when needed
results = await agent_tool.search_documents(
    query="What is the remote working policy?",
    top_k=5
)
```

**Example Response:**
```python
[
    {
        "file_name": "1 - Remote Working.txt",
        "content": "Employees may work remotely up to 3 days per week...",
        "rrf_score": 0.0323,  # Combined RRF score
        "vector_rank": 1,      # Rank in semantic search
        "text_rank": 2         # Rank in keyword search
    },
    {
        "file_name": "4 - IT Security.txt", 
        "content": "When working remotely, employees must...",
        "rrf_score": 0.0303,
        "vector_rank": 3,
        "text_rank": 1
    }
]
```

#### Prompts

##### `grounded_qa_prompt`

Provides a prompt template for grounded question-answering.

**Signature:**
```python
def grounded_qa_prompt() -> str
```

**Returns:**
- `str`: Prompt text enforcing grounded responses

**Purpose:**
Instructs language models to:
- Answer ONLY using provided document context
- Avoid hallucination or speculation
- Explicitly state when information is missing

**Prompt Text:**
```
You answer ONLY using the provided documents. 
If information is missing, say you don't know.
```

## Configuration

The server requires configuration via environment variables (see `src/mcp_rag_agent/core/config.py`):

### Required Settings

```python
# MCP Server
MCP_NAME = "rag_server"

# MongoDB
DB_URL = "mongodb://localhost:27017"
DB_NAME = "rag_database"
DB_DOCUMENTS_COLLECTION = "documents"
DB_VECTOR_COLLECTION = "document_embeddings"
DB_VECTOR_INDEX_NAME = "vector_index"

# OpenAI
OPENAI_API_KEY = "sk-..."
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Search Settings
SEMANTIC_WEIGHT = 0.7  # 0.7 = 70% semantic, 30% keyword (default)
```

### Hybrid Search Configuration

The `SEMANTIC_WEIGHT` parameter controls the balance between semantic and keyword search:

- **1.0**: Pure semantic search (100% vector similarity)
- **0.7**: Semantic-focused (70% semantic, 30% keyword) - **Recommended default**
- **0.5**: Balanced hybrid search
- **0.3**: Keyword-focused (30% semantic, 70% keyword)
- **0.0**: Pure keyword search (100% text matching)

**When to adjust:**
- **Higher semantic weight** (0.7-1.0): Exploratory queries, conceptual questions
- **Lower semantic weight** (0.0-0.5): Precise technical terms, exact phrase matching

## Running the Server

### Development Mode (with UI)

FastMCP provides a development UI for testing:

```bash
mcp dev src/mcp_rag_agent/mcp_server/server.py
```

This opens an interactive interface where you can:
- Test the `search_documents` tool with different queries
- View tool schemas and documentation
- Inspect responses and debug issues

### Production Mode (stdio)

The server runs in stdio mode when called programmatically:

```bash
python -m mcp_rag_agent.mcp_server.server
```

Or as a module:

```python
from mcp_rag_agent.mcp_server.server import mcp
mcp.run(transport="stdio")
```

### Integration with Agent

The agent automatically connects to the server:

```python
MCP_CONNECTIONS = {
    "rag_server": {
        "command": "python",
        "args": ["-m", "mcp_rag_agent.mcp_server.server"],
        "transport": "stdio",
    }
}
```

## Tool Schema

The MCP server automatically generates JSON schemas for all tools. For `search_documents`:

```json
{
  "name": "search_documents",
  "description": "Perform semantic search on indexed documents using vector embeddings...",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query text to find relevant documents"
      },
      "top_k": {
        "type": "integer",
        "description": "The maximum number of results to return",
        "default": 3
      }
    },
    "required": ["query"]
  }
}
```

## Hybrid Search Pipeline

The complete hybrid search flow:

```
User Query: "What is the annual leave policy?"
    ↓
1. Embedding Generation
   Query → OpenAI API → Vector [0.12, -0.34, 0.56, ...]
    ↓
2. Parallel Search Execution
   ├─→ Vector Search: MongoDB vector index (cosine similarity)
   └─→ Text Search: MongoDB text index (keyword matching)
    ↓
3. Reciprocal Rank Fusion (RRF)
   Combine rankings: RRF_score = Σ(1 / (k + rank_i))
   Apply semantic_weight to balance vector vs text
    ↓
4. Results Retrieval
   Fetch top-k documents ordered by RRF score
    ↓
5. Response Formation
   [{file_name, content, rrf_score, vector_rank, text_rank}, ...]
    ↓
Agent: Uses retrieved documents to formulate grounded answer
```

## Error Handling

The server includes robust error handling:

### Connection Errors
```python
# MongoDB connection failure
logger.error("Failed to connect to MongoDB")
# Agent receives empty results or error message
```

### Embedding Errors
```python
# OpenAI API failure
logger.error("Failed to generate embeddings")
# Falls back gracefully or returns error
```

### Search Errors
```python
# Vector index not found
logger.warning("Vector index not available")
# Returns empty results
```

## Logging

The server uses structured logging:

```python
logger.info("Initializing MongoDB client...")
logger.info("Initializing embedding generator...")
logger.info("Initializing hybrid search...")
logger.info(f"Hybrid search started: {query}")
```

Log output includes:
- Initialization steps (MongoDB, embeddings, hybrid search)
- Search queries and parameters (query text, top_k, semantic_weight)
- Results counts and RRF scores
- Error messages and stack traces

## Dependencies

- `mcp`: Model Context Protocol SDK
- `fastmcp`: FastMCP framework for building MCP servers
- `pymongo`: MongoDB driver
- `openai`: OpenAI API client
- `asyncio`: Asynchronous operations

## Integration Flow

```
┌─────────────────┐
│  LangChain Agent│
│  (Client)       │
└────────┬────────┘
         │ Uses MCP Tools
         ↓
┌─────────────────┐
│  MCP Server     │
│  (stdio)        │
├─────────────────┤
│ - search_docs   │
│ - grounded_qa   │
└────────┬────────┘
         │ Calls
         ↓
┌─────────────────┐
│  Hybrid Search  │
│ Module          │
└────────┬────────┘
         │ Queries
         ↓
┌─────────────────┐
│ MongoDB         │
│ (Vector Store)  │
└─────────────────┘
```

## Best Practices

1. **Vector Index Creation**: Ensure MongoDB vector index exists before running server
2. **Connection Pooling**: MongoDB client maintains connection pool for efficiency
3. **Error Logging**: Monitor logs for embedding/search failures
4. **Top-K Tuning**: Adjust `top_k` based on document size and query complexity
5. **Embedding Cache**: Consider caching embeddings for frequently searched queries

## Testing the Server

### Using MCP Dev UI

```bash
mcp dev src/mcp_rag_agent/mcp_server/server.py
```

Test queries:
- "remote working policy"
- "annual leave entitlement"
- "expense reimbursement process"
- "IT security requirements"

### Programmatic Testing

```python
import asyncio
from mcp_rag_agent.mcp_server.server import search_documents

async def test():
    results = await search_documents(
        query="What are the sustainability policies?",
        top_k=5
    )
    for doc in results:
        print(f"File: {doc['file_name']}")
        print(f"Score: {doc['score']}")
        print(f"Content: {doc['content'][:100]}...")
        print()

asyncio.run(test())
```

## Troubleshooting

### Server won't start

- **Check MongoDB**: Ensure MongoDB is running and accessible
- **Verify credentials**: Check `OPENAI_API_KEY` is valid
- **Review config**: Validate all required environment variables are set

### No search results

- **Index missing**: Run document indexing script first
- **Wrong collection**: Verify `DB_VECTOR_COLLECTION` points to correct collection
- **Index name**: Check `DB_VECTOR_INDEX_NAME` matches MongoDB index

### Poor search quality

- **Increase top_k**: Return more results for better context
- **Check embeddings**: Verify embedding model matches indexed documents
- **Review queries**: Ensure queries are clear and specific

### Connection timeouts

- **MongoDB timeout**: Increase connection timeout in MongoDB client
- **Network issues**: Check firewall/network settings
- **Resource limits**: Monitor MongoDB memory and CPU usage

## Extending the Server

### Adding New Tools

```python
@mcp.tool()
async def summarize_document(file_name: str) -> str:
    """Summarize a specific document."""
    # Implementation
    pass
```

### Adding New Prompts

```python
@mcp.prompt()
def citation_prompt() -> str:
    """Prompt for including citations."""
    return "Always cite your sources using [file_name]."
```

### Adding Resources

```python
@mcp.resource("policy://remote-working")
async def get_remote_policy() -> str:
    """Provide remote working policy as resource."""
    # Implementation
    pass
```

## See Also

- [Agent Module](../agent/README.md) - MCP client integration
- [Embeddings Module](../embeddings/README.md) - Vector generation and indexing
- [MongoDB Client](../mongodb/README.md) - Database connectivity
- [FastMCP Documentation](https://github.com/jlowin/fastmcp) - MCP framework

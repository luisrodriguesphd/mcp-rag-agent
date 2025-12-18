# MongoDB Module

The MongoDB module provides a clean interface for database operations, specializing in vector storage and semantic search capabilities for RAG applications.

> **📚 For detailed search method comparisons, examples, and best practices, see [SEARCH_GUIDE.md](./SEARCH_GUIDE.md)**

## Overview

This module implements a MongoDB client wrapper that:
- Manages MongoDB connections with automatic connection pooling
- Provides CRUD operations for documents
- Supports vector search index creation and management
- Performs vector similarity searches using MongoDB Atlas Vector Search
- Handles collection lifecycle operations
- Integrates seamlessly with the embeddings and MCP server modules

## Architecture

```
mongodb/
├── __init__.py          # Module exports
└── client.py            # MongoDB client implementation
```

## Components

### `MongoDBClient`

A comprehensive wrapper around PyMongo that simplifies MongoDB operations and adds vector search capabilities.

#### Initialization

```python
from mcp_rag_agent.mongodb.client import MongoDBClient

client = MongoDBClient(
    uri="mongodb://localhost:27017",
    database_name="rag_database"
)
```

**Parameters:**
- `uri` (str): MongoDB connection URI
- `database_name` (str): Name of the database to use

#### Connection Management

##### `connect()`

Establishes connection to MongoDB.

```python
client.connect()
```

- Creates MongoClient instance
- Initializes database reference
- Uses lazy connection (actual connection on first operation)

##### `disconnect()`

Closes MongoDB connection.

```python
client.disconnect()
```

- Closes all connections in the pool
- Cleans up client resources
- Should be called when shutting down

##### `db` Property

Gets the database instance with automatic connection.

```python
database = client.db
```

- Auto-connects if not already connected
- Returns PyMongo Database instance
- Used internally by all operations

#### Collection Operations

##### `get_collection()`

Retrieves a collection by name.

```python
collection = client.get_collection("documents")
```

**Parameters:**
- `collection_name` (str): Name of the collection

**Returns:**
- PyMongo Collection instance

##### `list_collections()`

Lists all collections in the database.

```python
collections = client.list_collections()
# ['documents', 'document_embeddings', 'metadata']
```

**Returns:**
- List of collection names (strings)

##### `create_collection()`

Creates a new collection.

```python
collection = client.create_collection("new_collection")
```

**Parameters:**
- `collection_name` (str): Name of the collection to create

**Returns:**
- PyMongo Collection instance

**Note:** MongoDB creates collections automatically on first insert, but this method is useful for setting up collections explicitly.

##### `collection_exists()`

Checks if a collection exists.

```python
exists = client.collection_exists("documents")
# True or False
```

**Parameters:**
- `collection_name` (str): Name of the collection to check

**Returns:**
- Boolean indicating existence

#### Document Operations

##### `insert_document()`

Inserts a single document.

```python
doc_id = client.insert_document(
    collection_name="documents",
    document={
        "file_name": "policy.txt",
        "content": "Remote working policy...",
        "metadata": {"department": "HR"}
    }
)
```

**Parameters:**
- `collection_name` (str): Target collection
- `document` (dict): Document to insert

**Returns:**
- Document ID as string

##### `insert_documents()`

Inserts multiple documents.

```python
doc_ids = client.insert_documents(
    collection_name="documents",
    documents=[
        {"file_name": "policy1.txt", "content": "..."},
        {"file_name": "policy2.txt", "content": "..."},
    ]
)
```

**Parameters:**
- `collection_name` (str): Target collection
- `documents` (list[dict]): List of documents to insert

**Returns:**
- List of document IDs as strings

##### `find_documents()`

Finds documents matching a query.

```python
results = client.find_documents(
    collection_name="documents",
    query={"department": "HR"},
    limit=10
)
```

**Parameters:**
- `collection_name` (str): Collection to search
- `query` (dict): MongoDB query filter
- `limit` (int, optional): Maximum results (default: 10)

**Returns:**
- List of matching documents

**Example Queries:**
```python
# Find by field value
results = client.find_documents("docs", {"status": "active"})

# Find with operators
results = client.find_documents("docs", {"score": {"$gt": 0.8}})

# Find with regex
results = client.find_documents("docs", {"title": {"$regex": "policy"}})
```

##### `delete_documents()`

Deletes documents matching a query.

```python
deleted_count = client.delete_documents(
    collection_name="documents",
    query={"status": "archived"}
)
```

**Parameters:**
- `collection_name` (str): Collection to delete from
- `query` (dict): MongoDB query filter

**Returns:**
- Number of documents deleted (int)

#### Text Search Operations

##### `create_text_search_index()`

Creates a full-text search index for keyword-based searching.

```python
client.create_text_search_index(
    collection_name="documents",
    index_name="text_search_idx",
    text_fields=["content", "title", "description"],
    weights={"title": 10, "content": 5, "description": 1}
)
```

**Parameters:**
- `collection_name` (str): Collection to create index on
- `index_name` (str): Name for the text search index
- `text_fields` (list[str]): Fields to include in text index
- `weights` (dict[str, int], optional): Field importance weights (1-999)

**Features:**
- Enables keyword-based searching across text fields
- Supports natural language queries with stemming
- Automatic stop word removal
- Relevance scoring based on term frequency and field weights
- Only one text index allowed per collection

**Use Cases:**
- Keyword search (e.g., "remote work policy")
- Phrase matching (e.g., "annual leave")
- Multi-field search with weighted importance
- Combining with filters for refined searches

**Requirements:**
- Text fields must contain string values
- Index is automatically maintained as documents change
- For MongoDB Atlas, consider Atlas Search for advanced features

#### Vector Search Operations

##### `create_vector_search_index()`

Creates a vector search index for similarity searches.

```python
client.create_vector_search_index(
    collection_name="document_embeddings",
    index_name="vector_index",
    vector_field="embedding",
    dimensions=1536,
    similarity="cosine"
)
```

**Parameters:**
- `collection_name` (str): Collection containing vectors
- `index_name` (str): Name for the search index
- `vector_field` (str): Field containing vector embeddings
- `dimensions` (int): Number of dimensions in vectors (e.g., 1536 for text-embedding-3-small)
- `similarity` (str, optional): Similarity metric - `"cosine"`, `"euclidean"`, or `"dotProduct"` (default: "cosine")

**Requirements:**
- MongoDB Atlas cluster (vector search not available in self-hosted MongoDB)
- Collection must exist before creating index
- Index creation may take several minutes for large collections

**Note:** This is a one-time setup operation. Once created, the index persists and is automatically maintained.

##### `vector_search()`

Performs vector similarity search with optional score threshold filtering.

```python
results = client.vector_search(
    collection_name="document_embeddings",
    index_name="vector_index",
    vector_field="embedding",
    query_vector=[0.12, -0.34, 0.56, ...],  # 1536 dimensions
    limit=5,
    num_candidates=100,
    filter_query={"department": "HR"},
    min_score=0.7  # Only return high-quality matches
)
```

**Parameters:**
- `collection_name` (str): Collection to search
- `index_name` (str): Name of the vector search index
- `vector_field` (str): Field containing embeddings
- `query_vector` (list[float]): Query vector for similarity search
- `limit` (int, optional): Maximum results (default: 10)
- `num_candidates` (int, optional): Number of candidates to consider (default: 100)
- `filter_query` (dict, optional): Additional filter to apply to results
- `min_score` (float, optional): Minimum similarity score threshold (0-1 for cosine). Only documents with score >= min_score are returned. Default: None (no filtering)

**Returns:**
- List of documents with similarity scores

**Result Format:**
```python
[
    {
        "_id": ObjectId("..."),
        "file_name": "policy.txt",
        "content": "Remote working policy...",
        "embedding": [0.12, -0.34, ...],
        "score": 0.89  # Similarity score (0-1)
    },
    ...
]
```

**How It Works:**
1. Computes similarity between query_vector and all vectors in collection
2. Returns top-k most similar documents
3. Includes similarity score in results
4. Optionally filters results based on metadata
5. Applies score threshold if specified

**Score Threshold Examples:**
```python
# High relevance only (recommended for precision)
results = client.vector_search(..., min_score=0.75)

# Moderate relevance (balanced)
results = client.vector_search(..., min_score=0.6)

# Exploratory search (broader results)
results = client.vector_search(..., min_score=0.5)
```

**Recommended Thresholds:**
- **0.7-0.8**: High relevance (strict filtering)
- **0.6-0.7**: Moderate relevance (balanced)
- **0.5-0.6**: Lower relevance (exploratory)
- **None**: No filtering (all results)

#### Hybrid Search Operations

##### `hybrid_search()`

Performs hybrid search combining vector similarity and full-text search using Reciprocal Rank Fusion (RRF) with optional three-tier threshold filtering.

```python
# Generate query embedding
query_vector = await embedder.generate_embedding("remote work policy")

# Perform hybrid search with quality thresholds
results = client.hybrid_search(
    collection_name="documents",
    vector_index_name="vector_idx",
    text_index_name="text_search_idx",
    vector_field="embedding",
    query_vector=query_vector,
    query_text="remote work policy",
    limit=5,
    num_candidates=100,
    vector_weight=0.7,
    text_weight=0.3,
    filter_query={"department": "HR"},
    rrf_k=60,
    min_vector_score=0.6,   # Filter vector results before RRF
    min_text_score=1.0,     # Filter text results before RRF
    min_rrf_score=0.01      # Filter final combined results
)
```

**Parameters:**
- `collection_name` (str): Collection to search
- `vector_index_name` (str): Name of vector search index
- `text_index_name` (str): Name of text search index
- `vector_field` (str): Field containing embeddings
- `query_vector` (list[float]): Query embedding vector
- `query_text` (str): Text query string
- `limit` (int, optional): Maximum results (default: 10)
- `num_candidates` (int, optional): Candidates for vector search (default: 100)
- `semantic_weight` (float, optional): Weight controlling semantic vs keyword search (0-1, default: 0.7). Higher values favor semantic similarity, lower values favor keyword matching
- `filter_query` (dict, optional): Metadata filter
- `rrf_k` (int, optional): RRF constant (default: 60)
- `min_vector_score` (float, optional): Minimum vector similarity threshold. Filters vector results BEFORE RRF fusion. Default: None
- `min_text_score` (float, optional): Minimum text relevance threshold. Filters text results BEFORE RRF fusion. Default: None
- `min_rrf_score` (float, optional): Minimum RRF score threshold. Filters combined results AFTER RRF fusion. Default: None

**Returns:**
- List of documents with combined RRF scores and ranking information

**Result Format:**
```python
[
    {
        "_id": ObjectId("..."),
        "content": "Remote working policy...",
        "embedding": [0.12, -0.34, ...],
        "rrf_score": 0.0234,         # Combined RRF score
        "vector_rank": 2,            # Rank from vector search
        "text_rank": 1,              # Rank from text search
        "vector_score": 0.89,        # Original vector similarity
        "text_score": 12.5           # Original text relevance
    },
    ...
]
```

**Three-Tier Threshold Filtering:**

Hybrid search supports sophisticated quality control with three independent thresholds:

1. **Pre-RRF Filtering** (`min_vector_score`, `min_text_score`):
   - Filters individual search results BEFORE combining them
   - Removes low-quality matches early
   - Example: Filter out vector scores < 0.6 and text scores < 1.0

2. **RRF Fusion**:
   - Combines remaining results using RRF algorithm
   - Deduplicates documents found by both searches

3. **Post-RRF Filtering** (`min_rrf_score`):
   - Final quality gate on combined RRF scores
   - Ensures only high-quality combined results are returned
   - Example: Only return results with RRF score >= 0.01

**Threshold Examples:**
```python
# Strict quality control (high precision)
results = client.hybrid_search(
    ...,
    min_vector_score=0.75,  # High semantic similarity
    min_text_score=2.0,     # Strong keyword match
    min_rrf_score=0.015     # High combined quality
)

# Balanced approach (recommended)
results = client.hybrid_search(
    ...,
    min_vector_score=0.6,   # Moderate semantic
    min_text_score=1.0,     # Reasonable keywords
    min_rrf_score=0.01      # Standard quality gate
)

# Exploratory search (broad results)
results = client.hybrid_search(
    ...,
    min_vector_score=0.5,   # Accept broader matches
    min_text_score=None,    # No text filtering
    min_rrf_score=None      # No final filtering
)
```

**How RRF Works:**

Reciprocal Rank Fusion (RRF) is an industry-standard algorithm that combines rankings from multiple search systems:

1. **Execute Both Searches**: Runs vector and text search in parallel
2. **Apply Pre-RRF Thresholds**: Filters each result set independently
3. **Assign Ranks**: Documents are ranked by their position (1, 2, 3, ...)
4. **Calculate RRF Scores**: For each document:
   ```
   RRF_score = (vector_weight / (k + vector_rank)) + (text_weight / (k + text_rank))
   
   where:
       semantic_weight = 0.7 (default, adjustable 0-1)
       vector_weight = semantic_weight
       text_weight = 1.0 - semantic_weight
       k = 60 (default, reduces impact of high ranks)
   ```
5. **Deduplicate**: Documents in both result sets have combined scores
6. **Apply Post-RRF Threshold**: Filters final results by combined score
7. **Rank by Score**: Final results sorted by RRF score (highest first)

**Why RRF?**
- **Robust**: No score normalization needed (combines ranks, not scores)
- **Industry Standard**: Used by Elasticsearch, Weaviate, Vespa
- **Balanced**: Weights control importance of semantic vs keyword matching
- **Proven**: Based on research from University of Waterloo

**Use Cases:**
- **Semantic + Keyword**: "remote work" finds both similar concepts and exact matches
- **Technical Terms**: Captures both embeddings context and precise terminology
- **Mixed Queries**: Natural language questions with specific keywords
- **Improved Recall**: Finds relevant documents missed by single approach

**Best Practices:**
- Use `semantic_weight=0.7` (default) for semantic-heavy queries
- Use `semantic_weight=0.5` for balanced importance
- Use `semantic_weight=0.3` for keyword-heavy queries
- Use `semantic_weight=1.0` for pure semantic search (vector only)
- Use `semantic_weight=0.0` for pure keyword search (text only)
- Adjust `rrf_k` (1-100): lower values favor top results
- Ensure query_text and query_vector represent same query
- Create both indexes before first search
- Start with moderate thresholds (0.6, 1.0, 0.01) and adjust based on results
- Use strict thresholds for compliance/legal searches
- Use lenient thresholds for exploratory research

> **📚 For detailed threshold strategies, use cases, and tuning guidelines, see [SEARCH_GUIDE.md](./SEARCH_GUIDE.md#score-thresholds)**

## Configuration

The client uses environment variables via the `Config` object:

```python
from mcp_rag_agent.core.config import config

client = MongoDBClient(
    uri=config.db_url,
    database_name=config.db_name
)
```

**Required Environment Variables:**
```bash
DB_URL=mongodb+srv://user:pass@cluster.mongodb.net/
DB_NAME=rag_database
DB_DOCUMENTS_COLLECTION=documents
DB_VECTOR_COLLECTION=document_embeddings
DB_VECTOR_INDEX_NAME=vector_index
```

## Usage Examples

### Basic Document Operations

```python
from mcp_rag_agent.mongodb.client import MongoDBClient

# Initialize
client = MongoDBClient(
    uri="mongodb://localhost:27017",
    database_name="my_database"
)
client.connect()

# Insert documents
doc_id = client.insert_document(
    "policies",
    {
        "title": "Remote Working Policy",
        "content": "Employees may work remotely...",
        "department": "HR"
    }
)

# Query documents
hr_docs = client.find_documents(
    "policies",
    {"department": "HR"},
    limit=10
)

# Clean up
client.disconnect()
```

### Vector Search Workflow

```python
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator

# Setup
client = MongoDBClient("mongodb://localhost:27017", "rag_db")
embedder = EmbeddingGenerator(api_key="sk-...", model="text-embedding-3-small")

# 1. Create vector search index (one-time setup)
client.create_vector_search_index(
    collection_name="embeddings",
    index_name="semantic_index",
    vector_field="vector",
    dimensions=1536,
    similarity="cosine"
)

# 2. Generate query embedding
query_text = "What is the remote working policy?"
query_vector = await embedder.generate_embedding(query_text)

# 3. Search for similar documents
results = client.vector_search(
    collection_name="embeddings",
    index_name="semantic_index",
    vector_field="vector",
    query_vector=query_vector,
    limit=5
)

# 4. Process results
for doc in results:
    print(f"File: {doc['file_name']}")
    print(f"Score: {doc['score']:.3f}")
    print(f"Content: {doc['content'][:100]}...")
    print()
```

### Text Search Workflow

```python
from mcp_rag_agent.mongodb.client import MongoDBClient

# Setup
client = MongoDBClient("mongodb://localhost:27017", "rag_db")
client.connect()

# 1. Create text search index (one-time setup)
client.create_text_search_index(
    collection_name="documents",
    index_name="text_search_idx",
    text_fields=["content", "title"],
    weights={"title": 10, "content": 1}  # Title matches are 10x more important
)

# 2. Insert documents with text content
client.insert_documents(
    "documents",
    [
        {
            "title": "Remote Working Policy",
            "content": "Employees may work remotely up to 3 days per week...",
            "department": "HR"
        },
        {
            "title": "Annual Leave Guidelines",
            "content": "All employees are entitled to 25 days annual leave...",
            "department": "HR"
        }
    ]
)

# 3. Perform text search (Note: For standard MongoDB, use $text query with find)
# For MongoDB Atlas, you can use aggregation with $search
results = client.find_documents(
    "documents",
    {"$text": {"$search": "remote work"}},
    limit=5
)

# 4. Process results
for doc in results:
    print(f"Title: {doc['title']}")
    print(f"Content: {doc['content'][:100]}...")
    print()

client.disconnect()
```

### Hybrid Search Workflow

```python
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator

# Setup
client = MongoDBClient("mongodb://localhost:27017", "rag_db")
embedder = EmbeddingGenerator(api_key="sk-...", model="text-embedding-3-small")
client.connect()

# 1. Create both indexes (one-time setup)
# Vector search index
client.create_vector_search_index(
    collection_name="documents",
    index_name="vector_idx",
    vector_field="embedding",
    dimensions=1536,
    similarity="cosine"
)

# Text search index (for MongoDB Atlas Search)
client.create_text_search_index(
    collection_name="documents",
    index_name="text_idx",
    text_fields=["content", "title"],
    weights={"title": 10, "content": 1}
)

# 2. Insert documents with both embeddings and text
query = "What is the remote working policy?"
embedding = await embedder.generate_embedding(query)

documents = [
    {
        "title": "Remote Working Policy",
        "content": "Employees may work remotely up to 3 days per week...",
        "embedding": await embedder.generate_embedding("Employees may work remotely..."),
        "department": "HR"
    }
]
client.insert_documents("documents", documents)

# 3. Perform hybrid search
query_text = "remote working from home"
query_vector = await embedder.generate_embedding(query_text)

results = client.hybrid_search(
    collection_name="documents",
    vector_index_name="vector_idx",
    text_index_name="text_idx",
    vector_field="embedding",
    query_vector=query_vector,
    query_text=query_text,
    limit=5,
    semantic_weight=0.7,  # 70% semantic, 30% keyword (default)
    filter_query={"department": "HR"}  # Optional filter
)

# 4. Analyze results
for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Title: {doc['title']}")
    print(f"RRF Score: {doc['rrf_score']:.4f}")
    print(f"Vector Rank: {doc['vector_rank']} (Score: {doc.get('vector_score', 'N/A')})")
    print(f"Text Rank: {doc['text_rank']} (Score: {doc.get('text_score', 'N/A')})")
    print(f"Content: {doc['content'][:150]}...")
    
    # Documents found by both methods have higher combined scores
    if doc['vector_rank'] and doc['text_rank']:
        print("✓ Found by both vector and text search")

client.disconnect()
```

### Comparing Search Methods

```python
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator

async def compare_search_methods():
    """Compare vector-only, text-only, and hybrid search."""
    client = MongoDBClient("mongodb://localhost:27017", "rag_db")
    embedder = EmbeddingGenerator(api_key="sk-...", model="text-embedding-3-small")
    
    query = "What are the sustainability initiatives?"
    query_vector = await embedder.generate_embedding(query)
    
    # Vector search only (semantic understanding)
    vector_results = client.vector_search(
        collection_name="documents",
        index_name="vector_idx",
        vector_field="embedding",
        query_vector=query_vector,
        limit=5
    )
    
    # Text search only (keyword matching)
    text_results = client.find_documents(
        collection_name="documents",
        query={"$text": {"$search": "sustainability initiatives"}},
        limit=5
    )
    
    # Hybrid search (best of both)
    hybrid_results = client.hybrid_search(
        collection_name="documents",
        vector_index_name="vector_idx",
        text_index_name="text_idx",
        vector_field="embedding",
        query_vector=query_vector,
        query_text="sustainability initiatives",
        limit=5
    )
    
    print("Vector Search: Found", len(vector_results), "results")
    print("Text Search: Found", len(text_results), "results")
    print("Hybrid Search: Found", len(hybrid_results), "results")
    
    # Hybrid search typically has better recall and precision
    return hybrid_results

# Run comparison
results = await compare_search_methods()
```

### Using Context Manager Pattern

```python
from mcp_rag_agent.mongodb.client import MongoDBClient

def process_documents():
    client = MongoDBClient("mongodb://localhost:27017", "rag_db")
    
    try:
        client.connect()
        
        # Perform operations
        docs = client.find_documents("policies", {})
        
        for doc in docs:
            # Process each document
            pass
            
    finally:
        client.disconnect()

process_documents()
```

### Running the Demo

The module includes a demo script:

```bash
python -m mcp_rag_agent.mongodb.client
```

This will:
- Connect to MongoDB using config settings
- List all collections in the database
- Display collection names
- Disconnect cleanly

## Vector Search Deep Dive

### Index Creation

Vector search indexes are created using MongoDB Atlas Vector Search:

```python
client.create_vector_search_index(
    collection_name="embeddings",
    index_name="vector_index",
    vector_field="embedding",
    dimensions=1536,
    similarity="cosine"
)
```

**Similarity Metrics:**

1. **Cosine** (recommended for text embeddings):
   - Measures angle between vectors
   - Range: -1 to 1 (normalized to 0-1 in results)
   - Best for semantic similarity

2. **Euclidean**:
   - Measures straight-line distance
   - Range: 0 to ∞
   - Best for spatial data

3. **Dot Product**:
   - Measures vector alignment
   - Range: -∞ to ∞
   - Best for magnitude-sensitive comparisons

### Search Pipeline

The vector search uses MongoDB's aggregation pipeline:

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": [0.12, -0.34, ...],
            "numCandidates": 100,
            "limit": 10
        }
    },
    {
        "$addFields": {
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]
```

**Parameters Explained:**
- `numCandidates`: Number of documents to consider (larger = slower but more accurate)
- `limit`: Final number of results to return
- `score`: Similarity score added to each result

## Best Practices

1. **Connection Management**:
   - Always call `disconnect()` when done
   - Use try/finally blocks to ensure cleanup
   - Reuse client instances when possible

2. **Vector Index Creation**:
   - Create indexes during setup, not runtime
   - Wait for index to be built before searching
   - Use appropriate dimensions for your embedding model

3. **Query Performance**:
   - Adjust `num_candidates` based on collection size
   - Use filters to narrow search space
   - Index frequently queried fields

4. **Error Handling**:
   - Handle connection timeouts
   - Catch and log MongoDB exceptions
   - Validate document structure before insert

5. **Batch Operations**:
   - Use `insert_documents()` for bulk inserts
   - Process results in batches for large queries
   - Consider memory usage with large result sets

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to MongoDB
```
pymongo.errors.ServerSelectionTimeoutError
```

**Solutions:**
- Verify MongoDB is running
- Check connection URI format
- Verify network connectivity
- Check firewall rules
- For Atlas: whitelist IP address

### Vector Search Errors

**Problem**: Vector search returns no results
```python
results = []  # Empty list
```

**Solutions:**
- Verify index exists and is built
- Check dimensions match embedding model
- Ensure documents have vector field
- Verify query vector is correct shape
- Wait for index build to complete (can take minutes)

**Problem**: "Index not found" error

**Solutions:**
- Create the index first using `create_vector_search_index()`
- Verify index name matches search call
- Check MongoDB Atlas version supports vector search

### Performance Issues

**Problem**: Slow vector searches

**Solutions:**
- Increase `num_candidates` cautiously (impacts speed)
- Add pre-filters to reduce search space
- Use appropriate similarity metric
- Consider query caching for common queries
- Monitor MongoDB Atlas performance metrics

## Dependencies

- `pymongo`: Official MongoDB Python driver
- `typing`: Type hints support

## Integration with Other Modules

### Embeddings Module
```python
# Client stores embeddings generated by EmbeddingGenerator
embeddings = embedding_generator.generate_embeddings(texts)
client.insert_documents("embeddings", embeddings)
```

### Semantic Search Module
```python
# Semantic search uses client for vector operations
search = SemanticSearch(
    mongo_client=client,
    embedding_generator=embedder
)
```

### MCP Server
```python
# MCP server initializes client for tool operations
mongo_client = MongoDBClient(uri=config.db_url, database_name=config.db_name)
mongo_client.connect()
```

## See Also

- [Embeddings Module](../embeddings/README.md) - Vector generation and indexing
- [MCP Server](../mcp_server/README.md) - Tool integration using MongoDB client
- [SEARCH_GUIDE.md](./SEARCH_GUIDE.md) - Detailed search methods comparison and best practices

## References

- [MongoDB: How to Perform Hybrid Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/hybrid-search/) - Official guide on Reciprocal Rank Fusion and hybrid search use cases
- [MongoDB Atlas Vector Search Overview](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/) - Official documentation on vector search 

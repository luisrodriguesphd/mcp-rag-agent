# MongoDB Search Implementation Guide

This guide explains the three search methods available in the MongoDB RAG Agent: Vector Search, Text Search, and Hybrid Search.

## Table of Contents
- [Overview](#overview)
- [Vector Search](#vector-search)
- [Text Search](#text-search)
- [Hybrid Search](#hybrid-search)
- [Comparison & Use Cases](#comparison--use-cases)
- [Best Practices](#best-practices)

---

## Overview

The MongoDB RAG Agent provides three complementary search approaches:

| Search Type | Method | Best For | Requires |
|-------------|--------|----------|----------|
| **Vector Search** | Semantic similarity | Conceptual queries, synonyms | Vector embeddings |
| **Text Search** | Keyword matching | Exact terms, technical terms | Text index |
| **Hybrid Search** | Combined (RRF) | Best overall results | Both indexes |

---

## Vector Search

### What It Does
Vector search finds documents based on **semantic similarity** using embedding vectors. It can find relevant documents even when they don't contain the exact query terms.

### How It Works
1. Converts query text to embedding vector (1536 dimensions for text-embedding-3-small)
2. Uses cosine similarity to find nearest vectors in MongoDB Atlas
3. Returns documents ranked by similarity score (0-1 scale)

### Key Features
- ✅ **Semantic understanding**: Finds conceptually related content
- ✅ **Synonym handling**: "AI" matches "artificial intelligence" 
- ✅ **No exact matches needed**: Finds relevant content without keywords
- ✅ **Cross-lingual**: Can work across languages with multilingual models
- ❌ **May miss specific terms**: Doesn't guarantee keyword presence

### Example

```python
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.semantic_search import SemanticSearch
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator

# Setup
client = MongoDBClient(uri="mongodb://...", database_name="mydb")
embedder = EmbeddingGenerator(api_key="...", model="text-embedding-3-small")
search = SemanticSearch(client, embedder)

# Search
results = await search.search(
    query="What is artificial intelligence?",
    limit=5
)

# Results might include documents about:
# - "Machine learning and AI"  ✓ (high similarity)
# - "Neural networks and deep learning"  ✓ (conceptually related)
# - "Python programming"  ✓ (often associated with AI)
# - "Cloud computing"  ✗ (low similarity)
```

### When to Use
- General knowledge questions
- Conceptual queries
- When synonyms/related terms are acceptable
- Cross-domain search (e.g., "how to improve code quality" matches various topics)

---

## Text Search

### What It Does
Text search finds documents containing **specific keywords** from your query using MongoDB's full-text search capabilities.

### How It Works
1. Processes query: removes stop words, applies stemming
2. Searches text index for matching terms
3. Returns documents containing query terms, ranked by relevance score

### Key Features Built-In

#### 1. **Stop Word Removal** ✅
Automatically removes common words like "the", "is", "what", "a":
```
Query: "What is machine learning?"
Processed: "machine learning"  (removed: "what", "is")
```

#### 2. **Stemming/Lemmatization** ✅
Matches word roots automatically:
```
"running" matches "run", "runs", "ran"
"machines" matches "machine"
"learning" matches "learn", "learned"
```

#### 3. **Case-Insensitive** ✅
```
"AI" matches "ai", "Ai", "AI"
```

### Features NOT Supported

#### ❌ Fuzzy Matching (Typo Tolerance)
```
"machne" does NOT match "machine"
"lerning" does NOT match "learning"
```
**Solution**: Use MongoDB Atlas Search with `use_atlas_search=True`

#### ❌ Wildcard/Partial Matches
```
Cannot search: "mach*" or "learn?"
```

#### ❌ Automatic Synonym Expansion
```
"AI" does NOT automatically match "artificial intelligence"
"ML" does NOT automatically match "machine learning"
```
**Solution**: Use hybrid search (vector component handles synonyms)

### Example

```python
from mcp_rag_agent.mongodb.client import MongoDBClient

client = MongoDBClient(uri="mongodb://...", database_name="mydb")

# Standard text search (self-hosted MongoDB)
results = client.text_search(
    collection_name="documents",
    index_name="text_index",
    query_text="machine learning artificial intelligence",
    limit=5,
    use_atlas_search=False  # Default: standard text search
)

# Results ONLY include documents containing:
# - "machine" or "learning" or "artificial" or "intelligence"
# - With stemming: "machines", "learned", "artificially", etc.
```

### Text Search Behavior Examples

**Query:** "What is machine learning and AI?"

**Processing:**
```
1. Stop word removal: "machine learning AI"
2. Stemming applied: "machin learn ai"
3. Search for: documents containing any of these terms
```

**Sample Results:**

| Document | Contains Terms | Match? | Score |
|----------|---------------|--------|-------|
| "Machine learning is a subset of AI..." | machine, learning, AI | ✅ Yes | 2.45 |
| "Artificial intelligence and ML..." | AI (variant) | ✅ Yes | 1.20 |
| "Python programming language..." | none | ❌ No | - |
| "Cloud computing services..." | none | ❌ No | - |

### When to Use
- Searching for specific technical terms
- Finding exact product names, codes, or identifiers
- When precision is more important than recall
- Compliance/legal searches requiring exact terms

---

## Hybrid Search

### What It Does
Hybrid search **combines vector and text search** using Reciprocal Rank Fusion (RRF) to provide the best of both worlds: semantic understanding with keyword precision.

### How It Works

1. **Parallel Execution**:
   ```
   Vector Search: Find top N semantically similar documents
   Text Search:   Find top N keyword-matching documents
   ```

2. **Reciprocal Rank Fusion (RRF)**:
   ```python
   For each document:
       rrf_score = vector_weight / (k + vector_rank) + text_weight / (k + text_rank)
   
   Where:
       k = 60 (default constant, reduces high-rank impact)
       vector_rank = position in vector results (1, 2, 3, ...)
       text_rank = position in text results (1, 2, 3, ...)
       semantic_weight = 0.7 (default, adjustable 0-1)
       vector_weight = semantic_weight
       text_weight = 1.0 - semantic_weight
   ```

3. **Score Combination**:
   - Documents in both result sets get combined scores
   - Documents in only one result set get partial scores
   - Final ranking by RRF score (highest first)

### Why RRF (Not Simple Score Addition)?

RRF is superior to simple score normalization because:
- ✅ **Scale-independent**: Works regardless of score magnitudes
- ✅ **Robust**: Not affected by outlier scores
- ✅ **Rank-based**: Focuses on relative positions, not absolute scores
- ✅ **Industry standard**: Used by Elasticsearch, Weaviate, Vespa

### Example

```python
from mcp_rag_agent.embeddings.hybrid_search import HybridSearch

hybrid = HybridSearch(
    mongo_client=client,
    embedding_generator=embedder
)

results = await hybrid.search(
    query="What is machine learning and AI?",
    limit=5,
    semantic_weight=0.7,  # 0.7 semantic, 0.3 keyword (default)
    rrf_k=60              # Standard RRF constant
)

# Each result includes:
for doc in results:
    print(f"RRF Score: {doc['rrf_score']:.4f}")
    print(f"Vector Rank: {doc['vector_rank']} (similarity: {doc.get('vector_score', 'N/A')})")
    print(f"Text Rank: {doc['text_rank']} (relevance: {doc.get('text_score', 'N/A')})")
    print(f"Content: {doc['content'][:100]}...")
```

### Hybrid Search Example Walkthrough

**Query:** "machine learning and AI"

**Step 1: Vector Search Results**
```
Rank 1: "Artificial intelligence and ML revolutionize..." (score: 0.89)
Rank 2: "Machine learning models learn from data..." (score: 0.85)
Rank 3: "Python programming for data science..." (score: 0.72)
```

**Step 2: Text Search Results**
```
Rank 1: "Machine learning is a subset of AI..." (score: 2.45)
Rank 2: "AI and machine learning applications..." (score: 2.20)
```

**Step 3: RRF Calculation** (k=60, semantic_weight=0.7 → vector_weight=0.7, text_weight=0.3)

| Document | Vector Rank | Text Rank | RRF Calculation | Final Score |
|----------|-------------|-----------|-----------------|-------------|
| "AI and ML revolutionize..." | 1 | 2 | 0.7/(60+1) + 0.3/(60+2) = 0.0115 + 0.0048 | **0.0163** ⭐ |
| "Machine learning is subset..." | 2 | 1 | 0.7/(60+2) + 0.3/(60+1) = 0.0113 + 0.0049 | **0.0162** |
| "Python for data science..." | 3 | None | 0.7/(60+3) + 0 = 0.0111 | **0.0111** |

**Step 4: Final Ranked Results**
```
1. "AI and ML revolutionize..." (RRF: 0.0163) - Found in BOTH searches
2. "Machine learning is subset..." (RRF: 0.0162) - Found in BOTH searches  
3. "Python for data science..." (RRF: 0.0111) - Found in vector only
```

### Tuning Semantic Weight

Adjust `semantic_weight` (0-1) to control the balance between semantic and keyword search:

```python
# Pure semantic search (exploratory, conceptual)
results = await hybrid.search(
    query="innovative AI solutions",
    semantic_weight=1.0  # 100% semantic (vector only)
)

# Semantic-focused (recommended default)
results = await hybrid.search(
    query="machine learning applications",
    semantic_weight=0.7  # 70% semantic, 30% keyword
)

# Balanced hybrid search
results = await hybrid.search(
    query="Python API documentation",
    semantic_weight=0.5  # 50% semantic, 50% keyword
)

# Keyword-focused (precision search)
results = await hybrid.search(
    query="GDPR Article 17 compliance",
    semantic_weight=0.3  # 30% semantic, 70% keyword
)

# Pure keyword search (exact terms only)
results = await hybrid.search(
    query="specific-product-code-XYZ",
    semantic_weight=0.0  # 0% semantic (text only)
)
```

### When to Use
- **Default choice for most applications** ✅
- General question answering
- Document retrieval systems
- When you want both precision and recall
- Production RAG systems

---

## Comparison & Use Cases

### Search Method Comparison

| Scenario | Vector Search | Text Search | Hybrid Search |
|----------|--------------|-------------|---------------|
| "What is AI?" | ⭐⭐⭐ Best | ⭐⭐ Good | ⭐⭐⭐ Best |
| "Find GDPR Article 17" | ⭐ Poor | ⭐⭐⭐ Best | ⭐⭐⭐ Best |
| "How to improve code quality?" | ⭐⭐⭐ Best | ⭐ Poor | ⭐⭐⭐ Best |
| "Python documentation" | ⭐⭐ Good | ⭐⭐⭐ Best | ⭐⭐⭐ Best |
| Exploratory research | ⭐⭐⭐ Best | ⭐ Poor | ⭐⭐ Good |
| Exact term search | ⭐ Poor | ⭐⭐⭐ Best | ⭐⭐ Good |

### Performance Characteristics

| Metric | Vector Search | Text Search | Hybrid Search |
|--------|--------------|-------------|---------------|
| Latency | ~100-200ms | ~10-50ms | ~100-250ms |
| Precision | Medium | High | High |
| Recall | High | Medium | Very High |
| Storage | High (vectors) | Low (index) | High |
| Setup Complexity | Medium | Low | Medium |

---

## Score Thresholds

### Overview

All search methods support optional score thresholds to filter out low-quality results automatically. Thresholds help improve precision by excluding irrelevant documents.

### Vector Search Thresholds

**Score Range:** 0-1 (for cosine similarity)

```python
# High relevance only
results = await hybrid.vector_search(
    query="artificial intelligence applications",
    min_score=0.75  # Only very similar documents
)

# Moderate relevance
results = await hybrid.vector_search(
    query="machine learning concepts",
    min_score=0.6  # Accept moderately similar documents
)

# Exploratory search
results = await hybrid.vector_search(
    query="technology trends",
    min_score=0.5  # Broader search
)
```

**Recommended Thresholds:**
- **0.8-1.0**: Extremely high relevance (very strict)
- **0.7-0.8**: High relevance (recommended for precision)
- **0.6-0.7**: Moderate relevance (balanced)
- **0.5-0.6**: Lower relevance (exploration)
- **< 0.5**: Likely not relevant

### Text Search Thresholds

**Score Range:** Varies by implementation
- Standard MongoDB text search: typically 0.5-3.0
- Atlas Search: typically 1.0-10.0+

```python
# Standard text search with threshold
results = client.text_search(
    collection_name="documents",
    index_name="text_idx",
    query_text="remote work policy",
    min_score=1.5,  # Filter weak matches
    use_atlas_search=False
)

# Atlas Search with threshold
results = client.text_search(
    collection_name="documents",
    index_name="atlas_text_idx",
    query_text="remote work policy",
    min_score=3.0,  # Higher threshold for Atlas
    use_atlas_search=True
)
```

**Recommended Thresholds:**

**Standard Text Search ($text):**
- **2.0+**: High relevance (multiple term matches)
- **1.0-2.0**: Moderate relevance (good matches)
- **0.5-1.0**: Lower relevance (weaker matches)

**Atlas Search:**
- **5.0+**: High relevance
- **2.0-5.0**: Moderate relevance
- **1.0-2.0**: Lower relevance

### Hybrid Search Thresholds

Hybrid search supports **three-tier filtering** for maximum control:

1. **Pre-RRF Filtering**: Filter individual search results before fusion
2. **RRF Fusion**: Combine remaining results using RRF algorithm
3. **Post-RRF Filtering**: Final quality gate on combined scores

```python
# Strict quality control
results = await hybrid.search(
    query="machine learning best practices",
    min_vector_score=0.7,   # High semantic similarity required
    min_text_score=1.5,     # Strong keyword match required
    min_rrf_score=0.015,    # High combined quality threshold
    limit=10
)

# Balanced approach (recommended)
results = await hybrid.search(
    query="data science techniques",
    min_vector_score=0.6,   # Moderate semantic similarity
    min_text_score=1.0,     # Reasonable keyword match
    min_rrf_score=0.01,     # Standard quality gate
    limit=10
)

# Lenient exploration
results = await hybrid.search(
    query="emerging technologies",
    min_vector_score=0.5,   # Accept broader matches
    min_text_score=None,    # No text filtering
    min_rrf_score=None,     # No final filtering
    limit=20
)

# Semantic-focused with text boost
results = await hybrid.search(
    query="innovative AI solutions",
    min_vector_score=0.65,  # Moderate semantic requirement
    min_text_score=None,    # Accept any text match
    min_rrf_score=0.008,    # Light final filter
    semantic_weight=0.8,    # Emphasize semantic (0.8 semantic, 0.2 keyword)
    limit=10
)
```

**Recommended RRF Thresholds:**
- **0.02+**: Very high quality (strict)
- **0.015-0.02**: High quality (recommended for precision)
- **0.01-0.015**: Standard quality (balanced)
- **0.005-0.01**: Lenient quality (more recall)
- **< 0.005**: Very lenient (exploration)

### Threshold Strategy by Use Case

#### Compliance / Legal Search
**Goal:** High precision, no false positives
```python
results = await hybrid.search(
    query="GDPR Article 17 right to erasure",
    min_vector_score=0.75,  # High semantic match
    min_text_score=2.0,     # Strong keyword presence
    min_rrf_score=0.02,     # Very high quality gate
    semantic_weight=0.5     # Balanced (50% semantic, 50% keyword)
)
```

#### General Q&A / RAG System
**Goal:** Balanced precision and recall
```python
results = await hybrid.search(
    query="What is the remote working policy?",
    min_vector_score=0.6,   # Moderate semantic
    min_text_score=1.0,     # Reasonable keywords
    min_rrf_score=0.01,     # Standard quality
    semantic_weight=0.7     # Semantic-focused (default)
)
```

#### Exploratory Research
**Goal:** Maximum recall, discover related content
```python
results = await hybrid.search(
    query="sustainability initiatives",
    min_vector_score=0.5,   # Broad semantic
    min_text_score=None,    # No text filter
    min_rrf_score=0.005,    # Light quality gate
    semantic_weight=0.8,    # Heavy semantic emphasis
    limit=20
)
```

#### Technical Documentation Search
**Goal:** Exact terms with context
```python
results = await hybrid.search(
    query="API authentication methods",
    min_vector_score=0.65,  # Moderate semantic
    min_text_score=1.5,     # Good keyword match
    min_rrf_score=0.012,    # Above average quality
    semantic_weight=0.6     # Slight semantic preference
)
```

### Dynamic Threshold Adjustment

Adjust thresholds based on result counts:

```python
async def adaptive_search(query: str, target_results: int = 10):
    """Adaptive search that adjusts thresholds based on results."""
    
    # Start with strict thresholds
    results = await hybrid.search(
        query=query,
        min_vector_score=0.75,
        min_text_score=2.0,
        min_rrf_score=0.015,
        limit=target_results * 2
    )
    
    # If too few results, relax thresholds
    if len(results) < target_results:
        results = await hybrid.search(
            query=query,
            min_vector_score=0.6,
            min_text_score=1.0,
            min_rrf_score=0.01,
            limit=target_results * 2
        )
    
    # If still too few, remove thresholds
    if len(results) < target_results:
        results = await hybrid.search(
            query=query,
            min_vector_score=None,
            min_text_score=None,
            min_rrf_score=None,
            limit=target_results
        )
    
    return results[:target_results]
```

### Monitoring and Tuning

Track threshold effectiveness:

```python
def analyze_threshold_impact(query: str):
    """Analyze how thresholds affect results."""
    
    # Search without thresholds
    all_results = await hybrid.search(query, limit=20)
    
    # Search with thresholds
    filtered_results = await hybrid.search(
        query,
        min_vector_score=0.7,
        min_text_score=1.5,
        min_rrf_score=0.01,
        limit=20
    )
    
    print(f"Without thresholds: {len(all_results)} results")
    print(f"With thresholds: {len(filtered_results)} results")
    print(f"Filtered out: {len(all_results) - len(filtered_results)} documents")
    
    # Show score distribution
    if all_results:
        vector_scores = [r.get('vector_score', 0) for r in all_results if r.get('vector_score')]
        text_scores = [r.get('text_score', 0) for r in all_results if r.get('text_score')]
        rrf_scores = [r.get('rrf_score', 0) for r in all_results]
        
        print(f"\nVector scores: min={min(vector_scores):.3f}, max={max(vector_scores):.3f}")
        print(f"Text scores: min={min(text_scores):.3f}, max={max(text_scores):.3f}")
        print(f"RRF scores: min={min(rrf_scores):.3f}, max={max(rrf_scores):.3f}")
```

## Best Practices

### 1. Choose the Right Search Method

```python
# Use vector search for:
results = await semantic_search.search("concepts related to neural networks")

# Use text search for:
results = client.text_search("specific-product-code-XYZ-123")

# Use hybrid search for (RECOMMENDED DEFAULT):
results = await hybrid_search.search("general question about anything")
```

### 2. Use Thresholds Appropriately

```python
# Start with moderate thresholds
results = await hybrid.search(
    query="your query",
    min_vector_score=0.6,
    min_text_score=1.0,
    min_rrf_score=0.01
)

# Adjust based on:
# - Result count (too few? lower thresholds)
# - Result quality (too many irrelevant? raise thresholds)
# - Use case requirements (precision vs recall)
```

### 3. Create Proper Indexes

```python
# For hybrid search, create BOTH indexes
hybrid.setup_indexes(
    collection_name="documents",
    vector_index_name="vector_idx",
    text_index_name="text_idx",
    text_fields=["content", "title"],  # Index multiple fields
    text_field_weights={"title": 10, "content": 1},  # Title more important
    dimensions=1536  # text-embedding-3-small
)
```

### 4. Optimize Text Queries

```python
# ❌ BAD: Too many stop words
query = "What is the best way to do machine learning?"

# ✅ GOOD: Focused keywords
query = "machine learning best practices"

# ✅ BETTER: Include specific terms
query = "machine learning model optimization techniques"
```

### 5. Monitor and Adjust

```python
# Log search results for analysis
results = await hybrid.search(query, limit=10)

for i, doc in enumerate(results, 1):
    logger.info(f"Rank {i}: RRF={doc['rrf_score']:.4f}, "
                f"Vector={doc['vector_rank']}, Text={doc['text_rank']}")
    
# Adjust semantic_weight based on which component performs better
if text_matches_are_better:
    semantic_weight = 0.5  # More balanced (decrease from 0.7)
elif vector_matches_are_better:
    semantic_weight = 0.8  # More semantic (increase from 0.7)
```

### 6. Handle Edge Cases

```python
# Empty results from text search (no keyword matches)
results = await hybrid.search(query="very specific unusual terminology")
# Hybrid search will still return vector results

# Documents with minimal text
# Ensure meaningful content in indexed fields
document = {
    "content": "Detailed explanation here...",  # Good
    # Not: "Document"  # Too short for meaningful search
}
```

### 7. Language Considerations

```python
# MongoDB text search supports multiple languages
# Set language at index level or document level
db.documents.create_index(
    [("content", "text")],
    default_language="english",  # or "spanish", "french", etc.
    language_override="language"  # field name for per-document language
)

# Document with language specification
document = {
    "content": "Contenido en español",
    "language": "spanish"
}
```

---

## Conclusion

**For most use cases, use Hybrid Search** - it provides the best balance of semantic understanding and keyword precision through the battle-tested RRF algorithm.

- **Vector Search**: When you need pure semantic similarity
- **Text Search**: When you need exact keyword matching  
- **Hybrid Search**: When you want the best overall results (recommended default)

The current implementation is production-ready and follows industry best practices used by major search platforms like Elasticsearch, Weaviate, and Vespa.

---

## References

- [MongoDB: How to Perform Hybrid Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/hybrid-search/) - Official guide on Reciprocal Rank Fusion and hybrid search use cases
- [MongoDB Atlas Vector Search Overview](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/) - Official documentation on vector search 

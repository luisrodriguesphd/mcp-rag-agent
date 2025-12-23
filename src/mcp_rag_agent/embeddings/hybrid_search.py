"""Hybrid search functionality combining vector and text search for MongoDB."""

from typing import Any, Optional

from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator

class HybridSearch:
    """Hybrid search engine combining vector similarity and full-text search."""
    
    def __init__(
        self,
        mongo_client: MongoDBClient,
        embedding_generator: EmbeddingGenerator,
        default_collection: str = "vectors",
        default_vector_index: str = "vector_index",
        default_text_index: str = "text_index",
        vector_field: str = "embedding",
        text_field: str = "content"
    ):
        """Initialize hybrid search.
        
        Args:
            mongo_client: MongoDB client instance.
            embedding_generator: Embedding generator instance.
            default_collection: Default collection name.
            default_vector_index: Default vector search index name.
            default_text_index: Default text search index name.
            vector_field: Field name for vector embeddings.
            text_field: Field name for text content.
        """
        self._mongo_client = mongo_client
        self._embedding_generator = embedding_generator
        self._default_collection = default_collection
        self._default_vector_index = default_vector_index
        self._default_text_index = default_text_index
        self._vector_field = vector_field
        self._text_field = text_field
    
    async def index_document(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> str:
        """Index a document for hybrid search.
        
        Args:
            content: Text content to index.
            metadata: Optional metadata to store with the document.
            collection_name: Collection to store the document.
            
        Returns:
            Inserted document ID.
        """
        collection = collection_name or self._default_collection
        
        # Generate embedding for the content
        embedding = await self._embedding_generator.generate(content)
        
        # Create document with embedding and text
        document = {
            self._text_field: content,
            self._vector_field: embedding,
            "metadata": metadata or {}
        }
        
        return self._mongo_client.insert_document(collection, document)
    
    async def index_documents(
        self,
        documents: list[dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> list[str]:
        """Index multiple documents for hybrid search.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'.
            collection_name: Collection to store the documents.
            
        Returns:
            List of inserted document IDs.
        """
        collection = collection_name or self._default_collection
        
        # Generate embeddings for all documents
        contents = [doc.get("content", "") for doc in documents]
        embeddings = await self._embedding_generator.generate_batch(contents)
        
        # Create documents with embeddings
        docs_with_embeddings = []
        for doc, embedding in zip(documents, embeddings):
            docs_with_embeddings.append({
                self._text_field: doc.get("content", ""),
                self._vector_field: embedding,
                "metadata": doc.get("metadata", {})
            })
        
        return self._mongo_client.insert_documents(collection, docs_with_embeddings)
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        collection_name: Optional[str] = None,
        vector_index_name: Optional[str] = None,
        text_index_name: Optional[str] = None,
        semantic_weight: float = 0.7,
        filter_query: Optional[dict[str, Any]] = None,
        rrf_k: int = 60,
        min_vector_score: Optional[float] = None,
        min_text_score: Optional[float] = None,
        min_rrf_score: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Perform hybrid search combining vector and text search.
        
        Args:
            query: Search query text.
            limit: Maximum number of results.
            collection_name: Collection to search.
            vector_index_name: Vector search index name.
            text_index_name: Text search index name.
            semantic_weight: Weight controlling semantic vs keyword search (0-1, default: 0.7).
                - 1.0 = Pure semantic search (vector only)
                - 0.7 = Semantic-focused (recommended default)
                - 0.5 = Balanced hybrid search
                - 0.3 = Keyword-focused
                - 0.0 = Pure keyword search (text only)
            filter_query: Optional filter to apply.
            rrf_k: RRF constant (default: 60).
            min_vector_score: Minimum vector similarity score threshold.
                Filters vector results before RRF. Example: 0.6 for moderate relevance.
            min_text_score: Minimum text relevance score threshold.
                Filters text results before RRF. Example: 1.0 for standard text search.
            min_rrf_score: Minimum RRF score threshold for final results.
                Filters combined results after RRF. Example: 0.01 for quality threshold.
            
        Returns:
            List of matching documents with RRF scores and rankings.
        """
        collection = collection_name or self._default_collection
        vector_index = vector_index_name or self._default_vector_index
        text_index = text_index_name or self._default_text_index
        
        # Generate embedding for the query
        query_vector = await self._embedding_generator.generate(query)
        
        # Perform hybrid search with thresholds
        results = self._mongo_client.hybrid_search(
            collection_name=collection,
            vector_index_name=vector_index,
            text_index_name=text_index,
            vector_field=self._vector_field,
            query_vector=query_vector,
            query_text=query,
            limit=limit,
            semantic_weight=semantic_weight,
            filter_query=filter_query,
            rrf_k=rrf_k,
            min_vector_score=min_vector_score,
            min_text_score=min_text_score,
            min_rrf_score=min_rrf_score
        )
        
        # Format results (remove embedding from response)
        formatted_results = []
        for result in results:
            result.pop(self._vector_field, None)
            result["_id"] = str(result.get("_id", ""))
            formatted_results.append(result)
        
        return formatted_results
    
    async def vector_search(
        self,
        query: str,
        limit: int = 10,
        collection_name: Optional[str] = None,
        index_name: Optional[str] = None,
        filter_query: Optional[dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Perform vector-only semantic search.
        
        Args:
            query: Search query text.
            limit: Maximum number of results.
            collection_name: Collection to search.
            index_name: Vector search index name.
            filter_query: Optional filter to apply.
            min_score: Minimum similarity score threshold (0-1 for cosine).
                Only documents with score >= min_score are returned.
            
        Returns:
            List of matching documents with scores.
        """
        collection = collection_name or self._default_collection
        index = index_name or self._default_vector_index
        
        # Generate embedding for the query
        query_vector = await self._embedding_generator.generate(query)
        
        # Perform vector search with threshold
        results = self._mongo_client.vector_search(
            collection_name=collection,
            index_name=index,
            vector_field=self._vector_field,
            query_vector=query_vector,
            limit=limit,
            filter_query=filter_query,
            min_score=min_score
        )
        
        # Format results
        formatted_results = []
        for result in results:
            result.pop(self._vector_field, None)
            result["_id"] = str(result.get("_id", ""))
            formatted_results.append(result)
        
        return formatted_results
    
    def text_search(
        self,
        query: str,
        limit: int = 10,
        collection_name: Optional[str] = None,
        index_name: Optional[str] = None,
        filter_query: Optional[dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Perform text-only keyword search.
        
        Args:
            query: Search query text.
            limit: Maximum number of results.
            collection_name: Collection to search.
            index_name: Text search index name.
            filter_query: Optional filter to apply.
            min_score: Minimum text relevance score threshold.
                Only documents with text_score >= min_score are returned.
            
        Returns:
            List of matching documents with text scores.
        """
        collection = collection_name or self._default_collection
        index = index_name or self._default_text_index
        
        # Perform text search with threshold (use standard text index by default)
        results = self._mongo_client.text_search(
            collection_name=collection,
            index_name=index,
            query_text=query,
            limit=limit,
            filter_query=filter_query,
            use_atlas_search=False,
            min_score=min_score
        )
        
        # Format results
        formatted_results = []
        for result in results:
            result.pop(self._vector_field, None)
            result["_id"] = str(result.get("_id", ""))
            formatted_results.append(result)
        
        return formatted_results
    
    def setup_indexes(
        self,
        collection_name: Optional[str] = None,
        vector_index_name: Optional[str] = None,
        text_index_name: Optional[str] = None,
        text_fields: Optional[list[str]] = None,
        text_field_weights: Optional[dict[str, int]] = None,
        dimensions: Optional[int] = None
    ) -> None:
        """Create both vector and text search indexes for a collection.
        
        Args:
            collection_name: Collection to create indexes on.
            vector_index_name: Name for the vector index.
            text_index_name: Name for the text index.
            text_fields: Fields to include in text index (defaults to [text_field]).
            text_field_weights: Optional weights for text fields.
            dimensions: Vector dimensions (defaults to embedding model dimensions).
        """
        collection = collection_name or self._default_collection
        vector_index = vector_index_name or self._default_vector_index
        text_index = text_index_name or self._default_text_index
        dims = dimensions or self._embedding_generator.dimensions
        fields = text_fields or [self._text_field]
        
        # Create vector search index
        self._mongo_client.create_vector_search_index(
            collection_name=collection,
            index_name=vector_index,
            vector_field=self._vector_field,
            dimensions=dims
        )
        
        # Create text search index
        self._mongo_client.create_text_search_index(
            collection_name=collection,
            index_name=text_index,
            text_fields=fields,
            weights=text_field_weights
        )

async def main():
    """Main function to demonstrate HybridSearch setup and usage."""
    import asyncio
    from mcp_rag_agent.core.config import config
    
    # Initialize MongoDB client
    print("Initializing MongoDB client...")
    mongo_client = MongoDBClient(uri=config.db_url, database_name=config.db_name)
    mongo_client.connect()
    
    # Initialize embedding generator
    print("Initializing embedding generator...")
    embedding_generator = EmbeddingGenerator(
        api_key=config.model_api_key,
        model=config.embedding_model,
        dimensions=config.embedding_dimension
    )
    
    # Initialize hybrid search
    print("Initializing hybrid search...")
    hybrid_search = HybridSearch(
        mongo_client=mongo_client,
        embedding_generator=embedding_generator,
        default_collection=config.db_vector_collection,
        default_vector_index="vector_index",
        default_text_index="text_index"
    )
    
    try:
        # Check if collection exists, if not create it
        print(f"\nChecking collection '{config.db_vector_collection}'...")
        
        if not mongo_client.collection_exists(config.db_vector_collection):
            print(f"Collection '{config.db_vector_collection}' does not exist. Creating collection...")
            mongo_client.create_collection(config.db_vector_collection)
            print(f"Collection '{config.db_vector_collection}' created successfully!")
        else:
            print(f"Collection '{config.db_vector_collection}' exists.")

        # Create vector and text search indexes
        print(f"Setting up hybrid search indexes...")
        try:
            hybrid_search.setup_indexes(
                collection_name=config.db_vector_collection,
                vector_index_name="vector_index",
                text_index_name="text_index",
                text_fields=["content"],
                dimensions=config.embedding_dimension
            )
            print("Hybrid search indexes created successfully!")
            print("Note: MongoDB Atlas indexes may take a few minutes to become active.")
        except Exception as e:
            print(f"Indexes may already exist or error occurred: {e}")
        
        # Index dummy documents
        print("\nIndexing dummy documents...")
        dummy_docs = [
            {
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "metadata": {"source": "dummy", "category": "AI"}
            },
            {
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"source": "dummy", "category": "Programming"}
            },
            {
                "content": "Cloud computing provides on-demand access to computing resources over the internet.",
                "metadata": {"source": "dummy", "category": "Cloud"}
            },
            {
                "content": "Artificial intelligence and machine learning are transforming software development.",
                "metadata": {"source": "dummy", "category": "AI"}
            }
        ]
        
        doc_ids = await hybrid_search.index_documents(
            dummy_docs,
            collection_name=config.db_vector_collection
        )
        print(f"Indexed {len(doc_ids)} documents with IDs: {doc_ids}")
        
        # Perform hybrid search
        print("\n" + "="*60)
        print("HYBRID SEARCH (Vector + Text with RRF)")
        print("="*60)
        query = "What is machine learning and AI?"
        print(f"Query: '{query}'")
        
        results = await hybrid_search.search(
            query=query,
            limit=3,
            collection_name=config.db_vector_collection,
            vector_index_name="vector_index",
            text_index_name="text_index",
            semantic_weight=0.7  # Semantic-focused (0.7 semantic, 0.3 keyword)
        )
        
        print(f"\nFound {len(results)} results:")
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. RRF Score: {result.get('rrf_score', 'N/A'):.4f}")
            print(f"   Vector Rank: {result.get('vector_rank', 'N/A')}, Text Rank: {result.get('text_rank', 'N/A')}")
            if result.get('vector_score'):
                print(f"   Vector Score: {result.get('vector_score', 'N/A'):.4f}")
            if result.get('text_score'):
                print(f"   Text Score: {result.get('text_score', 'N/A'):.4f}")
            print(f"   Content: {result.get('content', 'N/A')[:100]}...")
            print(f"   Metadata: {result.get('metadata', {})}")
        
        # Compare with vector-only search
        print("\n" + "="*60)
        print("VECTOR-ONLY SEARCH (for comparison)")
        print("="*60)
        vector_results = await hybrid_search.vector_search(
            query=query,
            limit=3,
            collection_name=config.db_vector_collection
        )
        
        print(f"\nFound {len(vector_results)} results:")
        for idx, result in enumerate(vector_results, 1):
            print(f"\n{idx}. Score: {result.get('score', 'N/A'):.4f}")
            print(f"   Content: {result.get('content', 'N/A')[:100]}...")
        
        # Compare with text-only search
        print("\n" + "="*60)
        print("TEXT-ONLY SEARCH (for comparison)")
        print("="*60)
        text_results = hybrid_search.text_search(
            query=query,
            limit=3,
            collection_name=config.db_vector_collection
        )
        
        print(f"\nFound {len(text_results)} results:")
        for idx, result in enumerate(text_results, 1):
            print(f"\n{idx}. Text Score: {result.get('text_score', 'N/A'):.4f}")
            print(f"   Content: {result.get('content', 'N/A')[:100]}...")
        
        # Demonstrate threshold filtering
        print("\n" + "="*60)
        print("HYBRID SEARCH WITH THRESHOLDS (Quality Filtering)")
        print("="*60)
        print(f"Query: '{query}'")
        print("Applying thresholds:")
        print("  - min_vector_score=0.6 (moderate semantic similarity)")
        print("  - min_text_score=1.0 (reasonable keyword match)")
        print("  - min_rrf_score=0.01 (standard quality gate)")
        
        # First get all results without thresholds for comparison
        all_results = await hybrid_search.search(
            query=query,
            limit=10,  # Get more results to show filtering effect
            collection_name=config.db_vector_collection,
            vector_index_name="vector_index",
            text_index_name="text_index",
            semantic_weight=0.7
        )
        
        # Then apply thresholds
        threshold_results = await hybrid_search.search(
            query=query,
            limit=10,  # Same limit for fair comparison
            collection_name=config.db_vector_collection,
            vector_index_name="vector_index",
            text_index_name="text_index",
            semantic_weight=0.7,
            min_vector_score=0.5,   # Filter vector results
            min_text_score=1.2,     # Filter text results
            min_rrf_score=0.01      # Filter final RRF results
        )
        
        print(f"\nAfter applying thresholds, found {len(threshold_results)} results:")
        for idx, result in enumerate(threshold_results, 1):
            print(f"\n{idx}. RRF Score: {result.get('rrf_score', 'N/A'):.4f}")
            print(f"   Vector Rank: {result.get('vector_rank', 'N/A')}, Text Rank: {result.get('text_rank', 'N/A')}")
            if result.get('vector_score'):
                print(f"   Vector Score: {result.get('vector_score', 'N/A'):.4f} (threshold: 0.6)")
            if result.get('text_score'):
                print(f"   Text Score: {result.get('text_score', 'N/A'):.4f} (threshold: 1.0)")
            print(f"   Content: {result.get('content', 'N/A')[:100]}...")
            
            # Highlight which thresholds were passed
            passed_thresholds = []
            if result.get('vector_score', 0) >= 0.6:
                passed_thresholds.append("vector✓")
            if result.get('text_score', 0) >= 1.0:
                passed_thresholds.append("text✓")
            if result.get('rrf_score', 0) >= 0.01:
                passed_thresholds.append("rrf✓")
            print(f"   Passed: {', '.join(passed_thresholds) if passed_thresholds else 'none'}")
        
        # Show comparison summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Hybrid search (limit=3, no thresholds):       {len(results)} results")
        print(f"Hybrid search (limit=10, no thresholds):      {len(all_results)} results")
        print(f"Hybrid search (limit=10, with thresholds):    {len(threshold_results)} results")
        print(f"Filtered out by thresholds:                   {len(all_results) - len(threshold_results)} documents")
        
        if len(threshold_results) < len(all_results):
            print("\n✓ Threshold filtering improved precision by removing low-quality matches!")
        elif len(threshold_results) == len(all_results):
            print("\n✓ All results passed the quality thresholds!")
        else:
            print("\n⚠ Note: Different limits used - increase limit to see filtering effect")
        
        # Delete the dummy documents
        print("\n" + "="*60)
        print("Cleaning up dummy documents...")
        deleted_count = mongo_client.delete_documents(
            collection_name=config.db_vector_collection,
            query={"metadata.source": "dummy"}
        )
        print(f"Deleted {deleted_count} dummy document(s)")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nDisconnecting from MongoDB...")
        mongo_client.disconnect()
        print("Done!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

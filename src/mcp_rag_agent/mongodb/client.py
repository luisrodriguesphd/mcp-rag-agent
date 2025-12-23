"""MongoDB client for database operations."""

from typing import Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


class MongoDBClient:
    """MongoDB client wrapper for semantic search operations."""
    
    def __init__(self, uri: str, database_name: str):
        """Initialize MongoDB client.
        
        Args:
            uri: MongoDB connection URI.
            database_name: Name of the database to use.
        """
        self._uri = uri
        self._database_name = database_name
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
    
    def connect(self) -> None:
        """Establish connection to MongoDB."""
        if self._client is None:
            self._client = MongoClient(self._uri)
            self._db = self._client[self._database_name]
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
    
    @property
    def db(self) -> Database:
        """Get the database instance."""
        if self._db is None:
            self.connect()
        return self._db
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a collection by name.
        
        Args:
            collection_name: Name of the collection.
            
        Returns:
            MongoDB collection instance.
        """
        return self.db[collection_name]
    
    def list_collections(self) -> list[str]:
        """List all collections in the database.
        
        Returns:
            List of collection names.
        """
        return self.db.list_collection_names()
    
    def create_collection(self, collection_name: str) -> Collection:
        """Create a new collection in the database.
        
        Args:
            collection_name: Name of the collection to create.
            
        Returns:
            MongoDB collection instance.
        """
        return self.db.create_collection(collection_name)
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in the database.
        
        Args:
            collection_name: Name of the collection to check.
            
        Returns:
            True if collection exists, False otherwise.
        """
        return collection_name in self.list_collections()
    
    def insert_document(
        self, 
        collection_name: str, 
        document: dict[str, Any]
    ) -> str:
        """Insert a document into a collection.
        
        Args:
            collection_name: Name of the collection.
            document: Document to insert.
            
        Returns:
            Inserted document ID as string.
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def insert_documents(
        self, 
        collection_name: str, 
        documents: list[dict[str, Any]]
    ) -> list[str]:
        """Insert multiple documents into a collection.
        
        Args:
            collection_name: Name of the collection.
            documents: List of documents to insert.
            
        Returns:
            List of inserted document IDs as strings.
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    def find_documents(
        self, 
        collection_name: str, 
        query: dict[str, Any], 
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find documents matching a query.
        
        Args:
            collection_name: Name of the collection.
            query: MongoDB query filter.
            limit: Maximum number of documents to return.
            
        Returns:
            List of matching documents.
        """
        collection = self.get_collection(collection_name)
        cursor = collection.find(query).limit(limit)
        return list(cursor)
    
    def delete_documents(
        self,
        collection_name: str,
        query: dict[str, Any]
    ) -> int:
        """Delete documents matching a query.
        
        Args:
            collection_name: Name of the collection.
            query: MongoDB query filter for documents to delete.
            
        Returns:
            Number of documents deleted.
        """
        collection = self.get_collection(collection_name)
        result = collection.delete_many(query)
        return result.deleted_count
    
    def create_vector_search_index(
        self,
        collection_name: str,
        index_name: str,
        vector_field: str,
        dimensions: int,
        similarity: str = "cosine"
    ) -> None:
        """Create a vector search index on a collection.
        
        Args:
            collection_name: Name of the collection.
            index_name: Name for the search index.
            vector_field: Field containing the vector embeddings.
            dimensions: Number of dimensions in the vectors.
            similarity: Similarity metric (cosine, euclidean, dotProduct).
        """
        collection = self.get_collection(collection_name)
        
        index_definition = {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": vector_field,
                        "numDimensions": dimensions,
                        "similarity": similarity
                    }
                ]
            }
        }
        
        collection.create_search_index(index_definition)
    
    def create_text_search_index(
        self,
        collection_name: str,
        index_name: str,
        text_fields: list[str],
        weights: Optional[dict[str, int]] = None
    ) -> None:
        """Create a full-text search index on a collection.
        
        This method creates a text search index that enables keyword-based searching
        across specified text fields. Text indexes support natural language queries
        with features like stemming, stop word removal, and relevance scoring.
        
        Args:
            collection_name: Name of the collection.
            index_name: Name for the search index.
            text_fields: List of field names to include in the text index.
                Example: ["content", "title", "description"]
            weights: Optional dictionary mapping field names to weight values (1-999).
                Higher weights give more importance to matches in that field.
                Example: {"title": 10, "content": 5, "description": 1}
                If not provided, all fields have equal weight (1).
        
        Example:
            >>> client.create_text_search_index(
            ...     collection_name="documents",
            ...     index_name="text_search_idx",
            ...     text_fields=["content", "title"],
            ...     weights={"title": 10, "content": 1}
            ... )
        
        Note:
            - Only one text index can exist per collection
            - Text indexes are automatically maintained as documents are added/updated
            - For MongoDB Atlas, consider using Atlas Search for more advanced features
        """
        collection = self.get_collection(collection_name)
        
        # Build index specification
        index_spec = {}
        for field in text_fields:
            index_spec[field] = "text"
        
        # Create index with optional weights
        index_options = {"name": index_name}
        if weights:
            index_options["weights"] = weights
        
        collection.create_index(
            list(index_spec.items()),
            **index_options
        )
    
    def vector_search(
        self,
        collection_name: str,
        index_name: str,
        vector_field: str,
        query_vector: list[float],
        limit: int = 10,
        num_candidates: int = 100,
        filter_query: Optional[dict[str, Any]] = None,
        min_score: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search.
        
        Args:
            collection_name: Name of the collection.
            index_name: Name of the vector search index.
            vector_field: Field containing the vector embeddings.
            query_vector: Query vector for similarity search.
            limit: Maximum number of results to return.
            num_candidates: Number of candidates to consider.
            filter_query: Optional filter to apply to results.
            min_score: Optional minimum similarity score threshold (0-1 for cosine).
                Only documents with score >= min_score are returned.
                Example: 0.7 for high relevance, 0.5 for moderate relevance.
                Default: None (no filtering).
            
        Returns:
            List of matching documents with similarity scores.
        """
        collection = self.get_collection(collection_name)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": vector_field,
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": limit
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        if filter_query:
            pipeline[0]["$vectorSearch"]["filter"] = filter_query
        
        results = list(collection.aggregate(pipeline))
        
        # Apply score threshold filtering if specified
        if min_score is not None:
            results = [doc for doc in results if doc.get("score", 0) >= min_score]
        
        return results
    
    def text_search(
        self,
        collection_name: str,
        index_name: str,
        query_text: str,
        limit: int = 10,
        filter_query: Optional[dict[str, Any]] = None,
        use_atlas_search: bool = False,
        min_score: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Perform full-text search using MongoDB text index or Atlas Search.
        
        This method performs keyword-based search across text-indexed fields,
        supporting natural language queries with features like stemming, stop
        word removal, and relevance scoring.
        
        Args:
            collection_name: Name of the collection to search.
            index_name: Name of the text search index.
            query_text: Text query string for keyword search.
                Example: "remote work policy"
            limit: Maximum number of results to return (default: 10).
            filter_query: Optional MongoDB query filter to apply to results.
                Example: {"department": "HR", "status": "active"}
            use_atlas_search: If True, uses Atlas Search ($search operator).
                If False, uses standard text index ($text operator). Default: False.
            min_score: Optional minimum text relevance score threshold.
                Only documents with text_score >= min_score are returned.
                Note: Score ranges vary between standard ($text) and Atlas Search.
                Standard text search: typically 0.5-3.0 for relevant matches.
                Atlas Search: typically 1.0-10.0+ for relevant matches.
                Default: None (no filtering).
        
        Returns:
            List of documents sorted by text relevance score (highest first).
            Each document includes:
                - All original document fields
                - "text_score": Text relevance score
        
        Example:
            >>> # Create text search index first (one-time setup)
            >>> client.create_text_search_index(
            ...     "documents", "text_idx", ["content", "title"]
            ... )
            >>> 
            >>> # Perform text search (standard MongoDB)
            >>> results = client.text_search(
            ...     collection_name="documents",
            ...     index_name="text_idx",
            ...     query_text="remote work policy",
            ...     limit=5,
            ...     use_atlas_search=False
            ... )
            >>> 
            >>> for doc in results:
            ...     print(f"Score: {doc['text_score']:.2f}")
            ...     print(f"Content: {doc['content'][:100]}...")
        
        Note:
            - Standard text search ($text) works with self-hosted MongoDB
            - Atlas Search ($search) requires MongoDB Atlas and Search index
            - Both support stemming, stop word removal, and relevance scoring
            - Standard text search scores are lower magnitude than Atlas Search
        
        See Also:
            - create_text_search_index(): Create the required search index
            - vector_search(): Semantic similarity search
            - hybrid_search(): Combines text and vector search
        """
        collection = self.get_collection(collection_name)
        
        if use_atlas_search:
            # Use MongoDB Atlas Search ($search operator)
            pipeline = [
                {
                    "$search": {
                        "index": index_name,
                        "text": {
                            "query": query_text,
                            "path": {"wildcard": "*"}
                        }
                    }
                },
                {
                    "$limit": limit
                },
                {
                    "$addFields": {
                        "text_score": {"$meta": "searchScore"}
                    }
                }
            ]
            
            if filter_query:
                pipeline.insert(1, {"$match": filter_query})
            
            results = list(collection.aggregate(pipeline))
            
            # Apply score threshold filtering if specified
            if min_score is not None:
                results = [doc for doc in results if doc.get("text_score", 0) >= min_score]
            
            return results
        else:
            # Use standard MongoDB text search ($text operator)
            query = {"$text": {"$search": query_text}}
            if filter_query:
                query.update(filter_query)
            
            # Use aggregation to add text score
            pipeline = [
                {"$match": query},
                {
                    "$addFields": {
                        "text_score": {"$meta": "textScore"}
                    }
                },
                {"$sort": {"text_score": -1}},
                {"$limit": limit}
            ]
            
            results = list(collection.aggregate(pipeline))
            
            # Apply score threshold filtering if specified
            if min_score is not None:
                results = [doc for doc in results if doc.get("text_score", 0) >= min_score]
            
            return results
    
    def hybrid_search(
        self,
        collection_name: str,
        vector_index_name: str,
        text_index_name: str,
        vector_field: str,
        query_vector: list[float],
        query_text: str,
        limit: int = 10,
        num_candidates: int = 100,
        semantic_weight: float = 0.7,
        filter_query: Optional[dict[str, Any]] = None,
        rrf_k: int = 60,
        min_vector_score: Optional[float] = None,
        min_text_score: Optional[float] = None,
        min_rrf_score: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Perform hybrid search combining vector similarity and full-text search.
        
        This method implements Reciprocal Rank Fusion (RRF) to combine results from
        both vector similarity search (semantic understanding) and full-text search
        (keyword matching). RRF is an industry-standard approach that combines rankings
        rather than raw scores, making it robust to score scale differences.
        
        The RRF formula for each document is:
            RRF_score = Σ(weight_i / (k + rank_i))
        
        where:
            - weight_i is the weight for search type i (vector or text)
            - k is a constant (default 60) that reduces impact of high ranks
            - rank_i is the rank position (1, 2, 3, ...) from search type i
        
        Args:
            collection_name: Name of the collection to search.
            vector_index_name: Name of the vector search index.
            text_index_name: Name of the text search index.
            vector_field: Field containing the vector embeddings.
            query_vector: Query vector for semantic similarity search.
            query_text: Text query string for keyword search.
            limit: Maximum number of final results to return (default: 10).
            num_candidates: Number of candidates for vector search (default: 100).
                Higher values improve recall but reduce performance.
            semantic_weight: Weight controlling semantic vs keyword search (0-1, default: 0.7).
                - 1.0 = Pure semantic search (vector only)
                - 0.7 = Semantic-focused (recommended default)
                - 0.5 = Balanced hybrid search
                - 0.3 = Keyword-focused
                - 0.0 = Pure keyword search (text only)
                The text weight is automatically calculated as (1 - semantic_weight).
            filter_query: Optional MongoDB query filter to apply to both searches.
                Example: {"department": "HR", "status": "active"}
            rrf_k: RRF constant (default: 60). Lower values give more weight to
                top-ranked results. Typical values: 1-100.
            min_vector_score: Optional minimum vector similarity score threshold.
                Filters vector results BEFORE RRF fusion. Example: 0.6 for moderate relevance.
                Default: None (no filtering).
            min_text_score: Optional minimum text relevance score threshold.
                Filters text results BEFORE RRF fusion. Example: 1.0 for standard text search.
                Default: None (no filtering).
            min_rrf_score: Optional minimum RRF score threshold for final results.
                Filters combined results AFTER RRF fusion. Example: 0.01 for quality threshold.
                Default: None (no filtering).
        
        Returns:
            List of documents sorted by combined RRF score (highest first).
            Each document includes:
                - All original document fields
                - "rrf_score": Combined score from RRF algorithm
                - "vector_rank": Rank from vector search (if found, else None)
                - "text_rank": Rank from text search (if found, else None)
                - "vector_score": Original vector similarity score (if found)
                - "text_score": Original text relevance score (if found)
        
        Example:
            >>> # Create indexes first (one-time setup)
            >>> client.create_vector_search_index(
            ...     "documents", "vec_idx", "embedding", 1536
            ... )
            >>> client.create_text_search_index(
            ...     "documents", "text_idx", ["content", "title"]
            ... )
            >>> 
            >>> # Generate query embedding
            >>> query_vec = embedder.generate_embedding("remote work policy")
            >>> 
            >>> # Perform hybrid search (semantic-focused)
            >>> results = client.hybrid_search(
            ...     collection_name="documents",
            ...     vector_index_name="vec_idx",
            ...     text_index_name="text_idx",
            ...     vector_field="embedding",
            ...     query_vector=query_vec,
            ...     query_text="remote work policy",
            ...     limit=5,
            ...     semantic_weight=0.7  # Semantic-focused (0.7 semantic, 0.3 keyword)
            ... )
            >>> 
            >>> for doc in results:
            ...     print(f"RRF Score: {doc['rrf_score']:.4f}")
            ...     print(f"Vector Rank: {doc['vector_rank']}, Text Rank: {doc['text_rank']}")
            ...     print(f"Content: {doc['content'][:100]}...")
        
        Note:
            - Both vector and text indexes must exist before calling this method
            - Documents appearing in both result sets are deduplicated
            - RRF is more robust than simple score normalization
            - Adjust weights based on your use case (semantic vs keyword importance)
            - For best results, ensure query_text and query_vector represent the same query
        
        References:
            - RRF Paper: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
            - Used by: Elasticsearch, Weaviate, Vespa, and other search engines
        """
        # Calculate vector and text weights from semantic_weight parameter
        vector_weight = semantic_weight
        text_weight = 1.0 - semantic_weight
        
        # 1. Perform vector search using vector_search method with threshold
        vector_results = self.vector_search(
            collection_name=collection_name,
            index_name=vector_index_name,
            vector_field=vector_field,
            query_vector=query_vector,
            limit=limit * 2,  # Get more candidates for fusion
            num_candidates=num_candidates,
            filter_query=filter_query,
            min_score=min_vector_score  # Apply vector score threshold
        )
        
        # 2. Perform text search using text_search method with threshold
        text_results = self.text_search(
            collection_name=collection_name,
            index_name=text_index_name,
            query_text=query_text,
            limit=limit * 2,  # Get more candidates for fusion
            filter_query=filter_query,
            min_score=min_text_score  # Apply text score threshold
        )
        
        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, dict[str, Any]] = {}
        
        # Process vector search results
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = str(doc["_id"])
            rrf_score = vector_weight / (rrf_k + rank)
            
            rrf_scores[doc_id] = {
                "document": doc,
                "rrf_score": rrf_score,
                "vector_rank": rank,
                "vector_score": doc.get("score"),  # vector_search returns "score"
                "text_rank": None,
                "text_score": None
            }
        
        # Process text search results
        for rank, doc in enumerate(text_results, start=1):
            doc_id = str(doc["_id"])
            rrf_score = text_weight / (rrf_k + rank)
            
            if doc_id in rrf_scores:
                # Document found in both searches - combine scores
                rrf_scores[doc_id]["rrf_score"] += rrf_score
                rrf_scores[doc_id]["text_rank"] = rank
                rrf_scores[doc_id]["text_score"] = doc.get("text_score")
            else:
                # Document only in text search
                rrf_scores[doc_id] = {
                    "document": doc,
                    "rrf_score": rrf_score,
                    "vector_rank": None,
                    "vector_score": None,
                    "text_rank": rank,
                    "text_score": doc.get("text_score")
                }
        
        # 4. Sort by RRF score and prepare final results
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # 5. Build final result documents with optional RRF score filtering
        final_results = []
        for item in sorted_results[:limit]:
            # Apply RRF score threshold if specified
            if min_rrf_score is not None and item["rrf_score"] < min_rrf_score:
                continue
            
            doc = item["document"]
            doc["rrf_score"] = item["rrf_score"]
            doc["vector_rank"] = item["vector_rank"]
            doc["text_rank"] = item["text_rank"]
            
            # Include original scores if available
            if item["vector_score"] is not None:
                doc["vector_score"] = item["vector_score"]
            if item["text_score"] is not None:
                doc["text_score"] = item["text_score"]
            
            final_results.append(doc)
        
        return final_results


def main():
    """Main function to demonstrate list_collections method."""
    from mcp_rag_agent.core.config import config
    
    # Get MongoDB configuration from config
    uri = config.db_url
    database_name = config.db_name
    
    # Create client instance
    client = MongoDBClient(uri=uri, database_name=database_name)
    
    try:
        # Connect and list collections
        print(f"Connecting to MongoDB database: {database_name}")
        client.connect()
        
        collections = client.list_collections()
        
        print(f"\nFound {len(collections)} collection(s):")
        for collection in collections:
            print(f"  - {collection}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure cleanup
        client.disconnect()
        print("\nDisconnected from MongoDB")


if __name__ == "__main__":
    main()

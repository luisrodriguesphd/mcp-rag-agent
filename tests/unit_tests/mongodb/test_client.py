"""Unit tests for MongoDB client."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.results import InsertOneResult, InsertManyResult, DeleteResult
from bson import ObjectId

from mcp_rag_agent.mongodb.client import MongoDBClient


class TestMongoDBClient:
    """Test suite for MongoDBClient class."""

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mocked MongoClient."""
        with patch('mcp_rag_agent.mongodb.client.MongoClient') as mock:
            yield mock

    @pytest.fixture
    def client(self, mock_mongo_client):
        """Create MongoDBClient instance with mocked dependencies."""
        return MongoDBClient(uri="mongodb://localhost:27017", database_name="test_db")

    def test_init(self, client):
        """Test client initialization."""
        assert client._uri == "mongodb://localhost:27017"
        assert client._database_name == "test_db"
        assert client._client is None
        assert client._db is None

    def test_connect(self, client, mock_mongo_client):
        """Test connecting to MongoDB."""
        mock_db = MagicMock(spec=Database)
        mock_mongo_client.return_value.__getitem__.return_value = mock_db

        client.connect()

        mock_mongo_client.assert_called_once_with("mongodb://localhost:27017")
        assert client._client is not None
        assert client._db is not None

    def test_connect_idempotent(self, client, mock_mongo_client):
        """Test that connect is idempotent."""
        mock_db = MagicMock(spec=Database)
        mock_mongo_client.return_value.__getitem__.return_value = mock_db

        client.connect()
        client.connect()

        mock_mongo_client.assert_called_once()

    def test_disconnect(self, client, mock_mongo_client):
        """Test disconnecting from MongoDB."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_db = MagicMock(spec=Database)
        mock_client_instance.__getitem__.return_value = mock_db

        client.connect()
        client.disconnect()

        mock_client_instance.close.assert_called_once()
        assert client._client is None
        assert client._db is None

    def test_disconnect_when_not_connected(self, client):
        """Test disconnect when not connected does nothing."""
        client.disconnect()
        assert client._client is None

    def test_db_property_auto_connects(self, client, mock_mongo_client):
        """Test that db property auto-connects if not connected."""
        mock_db = MagicMock(spec=Database)
        mock_mongo_client.return_value.__getitem__.return_value = mock_db

        db = client.db

        mock_mongo_client.assert_called_once()
        assert db is mock_db

    def test_db_property_returns_existing(self, client, mock_mongo_client):
        """Test that db property returns existing connection."""
        mock_db = MagicMock(spec=Database)
        mock_mongo_client.return_value.__getitem__.return_value = mock_db

        client.connect()
        db1 = client.db
        db2 = client.db

        assert db1 is db2
        mock_mongo_client.assert_called_once()

    def test_get_collection(self, client, mock_mongo_client):
        """Test getting a collection."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection

        collection = client.get_collection("test_collection")

        mock_db.__getitem__.assert_called_once_with("test_collection")
        assert collection is mock_collection

    def test_list_collections(self, client, mock_mongo_client):
        """Test listing collections."""
        mock_db = MagicMock(spec=Database)
        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.list_collection_names.return_value = ["col1", "col2", "col3"]

        collections = client.list_collections()

        assert collections == ["col1", "col2", "col3"]
        mock_db.list_collection_names.assert_called_once()

    def test_create_collection(self, client, mock_mongo_client):
        """Test creating a collection."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.create_collection.return_value = mock_collection

        collection = client.create_collection("new_collection")

        mock_db.create_collection.assert_called_once_with("new_collection")
        assert collection is mock_collection

    def test_collection_exists_true(self, client, mock_mongo_client):
        """Test checking if collection exists (true case)."""
        mock_db = MagicMock(spec=Database)
        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.list_collection_names.return_value = ["col1", "col2"]

        exists = client.collection_exists("col1")

        assert exists is True

    def test_collection_exists_false(self, client, mock_mongo_client):
        """Test checking if collection exists (false case)."""
        mock_db = MagicMock(spec=Database)
        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.list_collection_names.return_value = ["col1", "col2"]

        exists = client.collection_exists("col3")

        assert exists is False

    def test_insert_document(self, client, mock_mongo_client):
        """Test inserting a single document."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_result = Mock(spec=InsertOneResult)
        mock_result.inserted_id = ObjectId("507f1f77bcf86cd799439011")

        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.insert_one.return_value = mock_result

        document = {"name": "test", "value": 123}
        doc_id = client.insert_document("test_collection", document)

        mock_collection.insert_one.assert_called_once_with(document)
        assert doc_id == "507f1f77bcf86cd799439011"

    def test_insert_documents(self, client, mock_mongo_client):
        """Test inserting multiple documents."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_result = Mock(spec=InsertManyResult)
        mock_result.inserted_ids = [
            ObjectId("507f1f77bcf86cd799439011"),
            ObjectId("507f1f77bcf86cd799439012")
        ]

        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.insert_many.return_value = mock_result

        documents = [{"name": "test1"}, {"name": "test2"}]
        doc_ids = client.insert_documents("test_collection", documents)

        mock_collection.insert_many.assert_called_once_with(documents)
        assert len(doc_ids) == 2
        assert doc_ids == ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"]

    def test_find_documents(self, client, mock_mongo_client):
        """Test finding documents."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = iter([{"name": "test1"}, {"name": "test2"}])

        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.find.return_value = mock_cursor

        query = {"status": "active"}
        documents = client.find_documents("test_collection", query, limit=5)

        mock_collection.find.assert_called_once_with(query)
        mock_cursor.limit.assert_called_once_with(5)
        assert len(documents) == 2

    def test_delete_documents(self, client, mock_mongo_client):
        """Test deleting documents."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_result = Mock(spec=DeleteResult)
        mock_result.deleted_count = 3

        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.delete_many.return_value = mock_result

        query = {"status": "inactive"}
        count = client.delete_documents("test_collection", query)

        mock_collection.delete_many.assert_called_once_with(query)
        assert count == 3

    def test_create_vector_search_index(self, client, mock_mongo_client):
        """Test creating a vector search index."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)

        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection

        client.create_vector_search_index(
            collection_name="test_collection",
            index_name="vector_index",
            vector_field="embeddings",
            dimensions=768,
            similarity="cosine"
        )

        expected_index = {
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embeddings",
                        "numDimensions": 768,
                        "similarity": "cosine"
                    }
                ]
            }
        }

        mock_collection.create_search_index.assert_called_once_with(expected_index)

    def test_vector_search(self, client, mock_mongo_client):
        """Test vector similarity search."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)
        mock_results = [
            {"_id": "1", "text": "doc1", "score": 0.95},
            {"_id": "2", "text": "doc2", "score": 0.89}
        ]

        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.aggregate.return_value = iter(mock_results)

        query_vector = [0.1] * 768
        results = client.vector_search(
            collection_name="test_collection",
            index_name="vector_index",
            vector_field="embeddings",
            query_vector=query_vector,
            limit=5,
            num_candidates=50
        )

        assert len(results) == 2
        assert results[0]["score"] == 0.95

        # Verify aggregate pipeline
        call_args = mock_collection.aggregate.call_args[0][0]
        assert call_args[0]["$vectorSearch"]["index"] == "vector_index"
        assert call_args[0]["$vectorSearch"]["path"] == "embeddings"
        assert call_args[0]["$vectorSearch"]["limit"] == 5
        assert call_args[0]["$vectorSearch"]["numCandidates"] == 50

    def test_vector_search_with_filter(self, client, mock_mongo_client):
        """Test vector search with filter query."""
        mock_db = MagicMock(spec=Database)
        mock_collection = MagicMock(spec=Collection)

        mock_mongo_client.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.aggregate.return_value = iter([])

        query_vector = [0.1] * 768
        filter_query = {"category": "policy"}

        client.vector_search(
            collection_name="test_collection",
            index_name="vector_index",
            vector_field="embeddings",
            query_vector=query_vector,
            limit=5,
            filter_query=filter_query
        )

        # Verify filter is included in pipeline
        call_args = mock_collection.aggregate.call_args[0][0]
        assert call_args[0]["$vectorSearch"]["filter"] == filter_query

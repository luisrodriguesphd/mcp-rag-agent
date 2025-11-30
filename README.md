# MCP RAG Agent

A production-ready Retrieval-Augmented Generation (RAG) agent built with the Model Context Protocol (MCP), LangChain, MongoDB Atlas Vector Search, and OpenAI. This project demonstrates how to build a grounded, policy-aware chatbot that answers questions strictly based on retrieved document context.

## Overview

The MCP RAG Agent is a sophisticated question-answering system that:
- Uses semantic search to find relevant documents from a policy corpus
- Employs a LangGraph ReAct agent to reason about and retrieve information
- Integrates via the Model Context Protocol (MCP) for modular, reusable components
- Ensures grounded responses using the COSTAR prompting framework
- Stores and retrieves documents using MongoDB Atlas Vector Search
- Provides comprehensive evaluation tools using RAGAS metrics

## Key Features

- **MCP Integration**: Standardized protocol for tool exposure and agent communication
- **Semantic Search**: Vector-based document retrieval using OpenAI embeddings
- **Grounded Responses**: Strict context-based answering with no hallucinations
- **ReAct Pattern**: Reasoning and acting cycles for intelligent tool usage
- **MongoDB Atlas**: Scalable vector storage with efficient similarity search
- **Automated Evaluation**: RAGAS-based metrics for answer quality assessment
- **COSTAR Prompting**: Structured prompt design for consistent, high-quality outputs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 LangChain ReAct Agent                       │
│  • Analyzes query                                           │
│  • Decides which tools to use                               │
│  • Formulates grounded response                             │
└────────────────────────┬────────────────────────────────────┘
                         │ Uses MCP Tools
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (stdio)                       │
│  Tools:                                                     │
│  • search_documents(query, top_k) → results                 │
│  Prompts:                                                   │
│  • grounded_qa_prompt → ensures factual responses           │
└────────────────────────┬────────────────────────────────────┘
                         │ Calls
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  Semantic Search Module                     │
│  • Generates query embeddings                               │
│  • Performs vector similarity search                        │
│  • Returns top-k relevant documents                         │
└────────────────────────┬────────────────────────────────────┘
                         │ Queries
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              MongoDB Atlas Vector Store                     │
│  • Stores document embeddings (1536 dimensions)             │
│  • Vector search index (cosine similarity)                  │
│  • Efficient approximate nearest neighbor search            │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mcp-rag-agent/
├── data/
│   ├── ingested_documents/         # Source documents (policies)
│   │   └── policies/
│   │       ├── 1 - Remote Working.txt
│   │       ├── 2 - Expenses.txt
│   │       ├── 3 - Annual Leave.txt
│   │       ├── 4 - IT Security.txt
│   │       └── 5 - Sustainability.txt
│   └── evaluation_documents/       # Test cases for evaluation
│       └── expected_behaviour.xlsx
├── evaluation/                     # Automated testing and metrics
│   ├── main.py                     # Main evaluation orchestration script
│   ├── answer_generator.py         # Generates answers using the agent
│   ├── metrics_evaluator.py        # Evaluates answers using RAGAS metrics
│   ├── metrics.py                  # RAGAS metrics wrapper and definitions
│   ├── results/                    # Evaluation output (CSV files)
│   └── README.md                   # Evaluation module documentation
├── src/mcp_rag_agent/
│   ├── agent/                      # LangChain agent implementation
│   │   ├── create_agent.py         # Agent creation and configuration
│   │   ├── prompts/                # COSTAR-based system prompts
│   │   │   ├── __init__.py         # Prompts module exports
│   │   │   └── system_prompt.py    # System prompt definitions
│   │   ├── utils/                  # Agent utility functions
│   │   │   ├── mcp_rag_agent_creator.py  # MCP-enabled agent factory
│   │   │   └── rag_agent_creator.py      # Base RAG agent factory
│   │   └── README.md               # Agent module documentation
│   ├── embeddings/                 # Document processing and indexing
│   │   ├── embedding_generator.py  # OpenAI embeddings generation
│   │   ├── index_documents.py      # Document indexing pipeline
│   │   ├── semantic_search.py      # Vector similarity search
│   │   └── README.md               # Embeddings module documentation
│   ├── mcp_server/                 # MCP server implementation
│   │   ├── server.py               # FastMCP server with tools
│   │   ├── tools.py                # MCP tool implementations
│   │   └── README.md               # MCP server documentation
│   ├── mongodb/                    # Database client
│   │   ├── client.py               # MongoDB wrapper with vector search
│   │   └── README.md               # MongoDB module documentation
│   └── core/                       # Configuration and utilities
│       ├── config.py               # Environment-based configuration
│       └── log_setup.py            # Logging configuration
├── tests/                          # Tests
│   └── unit_tests                  # Unit tests
├── .env.example                    # Example environment configuration
├── .gitignore                      # Git ignore patterns
├── requirements.txt                # Production dependencies
├── requirements_dev.txt            # Development dependencies
├── setup.py                        # Package installation configuration
├── start.cmd                       # Windows startup script
└── README.md                       # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- MongoDB Atlas account (for vector search)
- OpenAI API key

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd mcp-rag-agent
```

2. **Run the `start` file**:
```bash
# Windows:
start.cmd

# Linux/macOS:
chmod +x start.sh
./start.sh
```
This script will automatically:
- Install and upgrade pip
- Create and activate a virtual environment
- Install all development dependencies
- Install the package in editable mode

3. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Setup Workflow

1. **Index documents**:
```bash
python -m mcp_rag_agent.embeddings.index_documents
```
This will:
- Read documents from `data/ingested_documents/`
- Generate embeddings using OpenAI
- Store vectors in MongoDB Atlas
- Create vector search index

2. **Test the MCP server** (optional - requires Node.js):
```bash
mcp dev src/mcp_rag_agent/mcp_server/server.py
```
This opens a UI to test the `search_documents` tool and other resources.

3. **Run the agent**:
```bash
python -m mcp_rag_agent.agent.create_agent
```
This runs a demo query showing the agent in action.

4. **Evaluate performance** (optional):
```bash
python evaluation/main.py
```
Runs automated evaluation using RAGAS metrics.

## Usage Examples

### Basic Agent Query

```python
import asyncio
from mcp_rag_agent.agent.create_agent import create_mcp_rag_agent
from mcp_rag_agent.agent.prompts import system_prompt
from mcp_rag_agent.core.config import config

async def main():
    # Create agent
    agent = await create_mcp_rag_agent(
        system_prompt=system_prompt,
        config=config
    )
    
    # Query the agent
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": "What is the remote working policy?"
        }]
    })
    
    # Get the answer
    answer = result["messages"][-1].content
    print(answer)

asyncio.run(main())
```

### Direct Semantic Search

```python
import asyncio
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator
from mcp_rag_agent.embeddings.semantic_search import SemanticSearch
from mcp_rag_agent.core.config import config

async def main():
    # Setup
    mongo_client = MongoDBClient(config.db_url, config.db_name)
    mongo_client.connect()
    
    embedder = EmbeddingGenerator(
        api_key=config.model_api_key,
        model=config.embedding_model
    )
    
    search = SemanticSearch(mongo_client, embedder)
    
    # Search
    results = await search.search(
        query="annual leave entitlement",
        limit=3
    )
    
    for doc in results:
        print(f"File: {doc['file_name']}")
        print(f"Score: {doc['score']:.3f}")
        print(f"Content: {doc['content'][:200]}...\n")
    
    mongo_client.disconnect()

asyncio.run(main())
```

### Indexing New Documents

```python
import asyncio
from mcp_rag_agent.embeddings.index_documents import index_documents
from mcp_rag_agent.core.config import config

async def main():
    await index_documents(
        directory_path="data/ingested_documents",
        config=config
    )

asyncio.run(main())
```

## Module Documentation

Each module has detailed documentation:

- **[Agent](src/mcp_rag_agent/agent/README.md)**: LangGraph ReAct agent with MCP integration
- **[MCP Server](src/mcp_rag_agent/mcp_server/README.md)**: FastMCP server providing RAG tools
- **[MongoDB](src/mcp_rag_agent/mongodb/README.md)**: Database client with vector search
- **[Embeddings](src/mcp_rag_agent/embeddings/README.md)**: Document indexing and semantic search
- **[Evaluation](evaluation/README.md)**: Automated testing with RAGAS metrics

## Configuration

Configuration is managed through two layers:
1. Environment Variables (`.env`): Most settings are configured via environment variables, although only the external dependencies are included in the `.env.sample` file.
2. Code Configuration (`src/mcp_rag_agent/core/config.py`): Some advanced settings are configured directly in the `Config` class, such as text generation parameters (temperature,...)

**Note:** To modify these settings, edit `src/mcp_rag_agent/core/config.py` directly. The `Config` class loads environment variables and provides default values for all configuration parameters.

## Key Technologies

- **[LangChain](https://python.langchain.com/)**: Agent framework and orchestration
- **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/)**: Standardized tool integration
- **[FastMCP](https://github.com/jlowin/fastmcp)**: MCP server implementation
- **[MongoDB Atlas](https://www.mongodb.com/atlas/database)**: Vector storage and search
- **[OpenAI](https://openai.com/)**: LLM and embedding models
- **[RAGAS](https://docs.ragas.io/)**: RAG evaluation framework

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- Follow Python best practices and PEP 8
- Use type hints for all functions
- Add docstrings to public APIs
- Keep modules focused and cohesive

### Adding New Features

1. **New MCP Tool**:
   - Add `@mcp.tool()` decorated function in `server.py`
   - Document in MCP server README
   - Test with `mcp dev`

2. **New Document Type**:
   - Update `index_documents.py` to handle new format
   - Ensure metadata is preserved
   - Re-index documents

3. **New Metric**:
   - Add to `evaluation/metrics.py`
   - Update evaluator to compute and save metric
   - Document in evaluation README

## Evaluation

The project includes comprehensive evaluation tools using RAGAS:

```bash
python evaluation/evaluator.py
```

**Metrics computed**:
- Answer Relevancy
- Answer Similarity
- Answer Correctness

Results are saved to `evaluation/results/` with timestamps.

## Troubleshooting

### Common Issues

**MongoDB connection fails**:
- Verify MongoDB Atlas cluster is running
- Check IP whitelist in Atlas
- Validate connection URI in `.env`

**MCP server won't start**:
- Ensure MongoDB is connected
- Check OpenAI API key is valid
- Verify all dependencies are installed

**No search results**:
- Run `index_documents.py` to populate database
- Check vector index exists in MongoDB Atlas
- Verify embedding dimensions match

**Agent doesn't call tools**:
- Check MCP server is accessible
- Review system prompt encourages tool usage
- Increase model temperature if needed

**Evaluation errors**:
- Ensure `expected_behaviour.xlsx` exists
- Check OpenAI API quota
- Verify evaluation model is accessible

## Performance Considerations

- **Indexing**: ~1-2 seconds per document (depends on document size)
- **Query**: ~2-5 seconds per query (embedding + search + generation)
- **Vector Search**: Sub-second for collections up to 100K documents
- **Batch Operations**: Use `insert_documents()` for bulk indexing

## Best Practices

1. **Prompt Engineering**: Use COSTAR framework for all prompts
2. **Error Handling**: Always handle connection failures gracefully
3. **Logging**: Use structured logging for debugging
4. **Testing**: Run evaluation after significant changes
5. **Vector Index**: Create during setup, not runtime
6. **Connection Pooling**: Reuse MongoDB client instances
7. **API Rate Limits**: Implement exponential backoff for OpenAI calls

## Security

- Never commit `.env` file to version control
- Rotate API keys regularly
- Use MongoDB Atlas IP whitelisting
- Implement rate limiting for production deployments
- Sanitize user inputs before processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## License

MIT

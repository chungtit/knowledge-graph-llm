# Knowledge Graph Integration with Open Source LLMs (Updatings)

This project provides a Python framework for integrating Knowledge Graphs with open-source Large Language Models (LLMs). It combines the structured data capabilities of graph databases with the natural language understanding of LLMs to create a powerful question-answering system.

## Features

- Semantic search over knowledge graphs
- Integration with popular open-source LLMs (Mistral, LLaMA, Falcon, etc.)
- Neo4j graph database integration
- Explainable reasoning process
- Optimized for performance and memory efficiency
- Configurable and extensible architecture

## Installation

```bash
# Clone the repository
git clone https://github.com/chungtit/knowledge-graph-llm.git

# Install dependencies
pip install -r structured-data/requirements-dev.txt
```

### Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
neo4j>=5.0.0
networkx>=3.0
numpy>=1.24.0
```

## Quick Start

```python
from knowledge_graph_opensource import OpenSourceKGLLM

# Initialize the system
kg_llm = OpenSourceKGLLM(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",
    model_name="mistralai/Mistral-7B-Instruct-v0.2"
)

# Query the system
query = "What products did John Doe purchase?"
result = kg_llm.query_llm(query)

# Print the answer
print(result["answer"])

# Get reasoning explanation
explanation = kg_llm.explain_reasoning(
    query, 
    result["answer"], 
    result["subgraph"]
)
print(explanation)

# Clean up
kg_llm.close()
```

## Configuration Options

### LLM Models

The system supports various open-source LLMs:

```python
# Available model options
models = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "falcon": "tiiuae/falcon-7b-instruct",
    "mpt": "mosaicml/mpt-7b-instruct"
}
```

### Hardware Requirements

- **GPU**: Recommended for optimal performance
- **RAM**: Minimum 16GB, 32GB+ recommended
- **Storage**: 50GB+ for model weights

## Advanced Usage

### Custom Knowledge Graph Queries

```python
# Define custom Cypher query
cypher_query = """
MATCH (p:Product)<-[:PURCHASED]-(c:Customer)
WHERE c.name = 'John Doe'
RETURN p.name as product, p.category as category
"""

# Execute query
results = kg_llm.query_knowledge_graph(cypher_query)
```

### Subgraph Extraction

```python
# Get relevant subgraph for a query
subgraph = kg_llm.get_relevant_subgraph(
    query="What electronics did John buy?",
    max_nodes=10,
    similarity_threshold=0.5
)
```

### Memory Optimization

For large graphs or limited memory environments:

```python
# Initialize with memory optimization
kg_llm = OpenSourceKGLLM(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda",
    max_length=1024,
    load_in_8bit=True  # Enable 8-bit quantization
)
```

## Performance Tips

1. **Batch Processing**
   - Use batch processing for large graphs
   - Implement pagination for node retrieval

2. **Model Optimization**
   - Enable quantization for memory efficiency
   - Use appropriate model size for your hardware

3. **Query Optimization**
   - Limit subgraph size for faster processing
   - Use appropriate similarity thresholds

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -r structured-data/requirements-dev.txt
```

## Acknowledgments

- Neo4j team for the graph database
- Hugging Face for the transformers library
- Mistral AI, Meta, and other organizations for open-source LLMs

<!-- ## Support

For questions and support, please:
1. Check the [Issues](https://github.com/yourusername/kg-llm-integration/issues) page
2. Open a new issue if needed
3. Join our [Discord community](https://discord.gg/yourinvite) -->
# Unstructured Data

This folder contains resources and scripts related to handling unstructured data for the Knowledge Graph LLM project.

## Contents

- `demo-data/`: Directory containing raw unstructured demo data files.
- `notebooks/`: Jupyter notebooks for educational purposes.
- `README.md`: This file.

## Getting Started

1. **Clone the repository:**
    ```sh
    git clone https://github.com/chungtit/knowledge-graph-llm.git
    cd knowledge-graph-llm/unstructured-data
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
## Notebooks

The `notebooks/` directory contains Jupyter notebooks designed for educational purposes. These notebooks shows basic techniques and workflows for handling unstructured data and building knowledge graphs.

1. **Hardcoded Knowledge Graph with NetworkX:**
    The `notebooks/kg_hardcode.ipynb` notebook provides a step-by-step guide to building a simple knowledge graph from scratch using NetworkX. The edges and nodes of the graph are hardcoded to help users understand the basic concepts and structure of a knowledge graph.

2. **Interactive Knowledge Graph with Pyvis:**
    The `notebooks/interactive_graph.ipynb` notebook shows how to create an interactive knowledge graph using Pyvis. This notebook guides users through the process of visualizing the knowledge graph in an interactive manner, allowing for better exploration and understanding of the graph's structure and relationships.

3. **Dynamic Knowledge Graph with SpaCy and NetworkX:**
    The `notebooks/dynamic.ipynb` notebook shows how to build a dynamic knowledge graph by extracting information from unstructured text using SpaCy and constructing the graph with NetworkX. It covers the process of entity recognition, relationship extraction, and graph construction.
import networkx as nx
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import openai

class KnowledgeGraphLLM:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_key: str):
        """Initialize the KG-LLM integration system."""
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize OpenAI client
        openai.api_key = openai_key
        
    def query_knowledge_graph(self, query: str) -> List[Dict]:
        """Query the knowledge graph using Cypher."""
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    
    def get_relevant_subgraph(self, query: str, max_nodes: int = 10) -> nx.Graph:
        """Extract relevant subgraph based on query semantics."""
        # Convert query to embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Get nodes and their text representations
        cypher_query = """
        MATCH (n)
        RETURN n.name as name, labels(n) as labels, id(n) as id
        LIMIT 1000
        """
        nodes = self.query_knowledge_graph(cypher_query)
        
        # Calculate relevance scores
        node_scores = []
        for node in nodes:
            node_text = f"{' '.join(node['labels'])} {node['name']}"
            node_embedding = self.embedding_model.encode(node_text)
            similarity = np.dot(query_embedding, node_embedding)
            node_scores.append((node['id'], similarity))
        
        # Get top relevant nodes
        top_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_ids = [str(node[0]) for node in top_nodes]
        
        # Extract subgraph containing these nodes
        subgraph_query = f"""
        MATCH (n1)
        WHERE id(n1) IN {top_node_ids}
        MATCH (n1)-[r]-(n2)
        WHERE id(n2) IN {top_node_ids}
        RETURN n1, r, n2
        """
        relationships = self.query_knowledge_graph(subgraph_query)
        
        # Convert to NetworkX graph
        G = nx.Graph()
        for rel in relationships:
            G.add_edge(rel['n1']['name'], rel['n2']['name'], 
                      relationship=type(rel['r']).__name__)
        
        return G
    
    def generate_graph_context(self, subgraph: nx.Graph) -> str:
        """Convert subgraph to textual context for LLM."""
        context = []
        
        # Add nodes information
        context.append("Entities in the knowledge graph:")
        for node in subgraph.nodes():
            context.append(f"- {node}")
        
        # Add relationships information
        context.append("\nRelationships between entities:")
        for edge in subgraph.edges(data=True):
            context.append(f"- {edge[0]} {edge[2]['relationship']} {edge[1]}")
        
        return "\n".join(context)
    
    def query_llm(self, user_query: str) -> str:
        """Process user query using KG and LLM."""
        # Get relevant subgraph
        subgraph = self.get_relevant_subgraph(user_query)
        
        # Generate context from subgraph
        graph_context = self.generate_graph_context(subgraph)
        
        # Construct prompt for LLM
        prompt = f"""Based on the following knowledge graph information:

{graph_context}

Please answer this question: {user_query}

Use only the information provided in the knowledge graph. If you need additional information that's not in the graph, please mention it.
"""
        
        # Query LLM
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on knowledge graph information."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def explain_reasoning(self, query: str, answer: str, subgraph: nx.Graph) -> str:
        """Explain the reasoning process using the knowledge graph."""
        explanation_prompt = f"""I found this answer: "{answer}"
        
Based on this knowledge graph:
{self.generate_graph_context(subgraph)}

Explain step by step how the knowledge graph information was used to arrive at this answer.
Include any assumptions made and mention any missing information that would have been helpful.
"""
        
        explanation = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains reasoning processes."},
                {"role": "user", "content": explanation_prompt}
            ]
        )
        
        return explanation.choices[0].message.content

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
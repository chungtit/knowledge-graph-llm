import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
from typing import List, Dict, Any
import logging

class OpenSourceKGLLM:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048
    ):
        """Initialize the Knowledge Graph - Open Source LLM integration system."""
        self.device = device
        self.max_length = max_length
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            # Connect to Neo4j
            self.driver = GraphDatabase.driver(
                neo4j_uri, 
                auth=(neo4j_user, neo4j_password)
            )
            self.logger.info("Successfully connected to Neo4j")
            
            # Initialize the embedding model
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=device
            )
            self.logger.info("Successfully loaded embedding model")
            
            # Initialize the LLM and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            # Create pipeline for text generation
            self.llm = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device=device,
                max_length=max_length
            )
            
            self.logger.info(f"Successfully loaded LLM model {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise
    
    def query_knowledge_graph(self, query: str) -> List[Dict]:
        """Execute Cypher query on Neo4j database."""
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [dict(record) for record in result]
        except Exception as e:
            self.logger.error(f"Error querying knowledge graph: {str(e)}")
            return []

    def get_relevant_subgraph(
        self,
        query: str,
        max_nodes: int = 10,
        similarity_threshold: float = 0.5
    ) -> nx.Graph:
        """Extract relevant subgraph based on semantic similarity to query."""
        try:
            # Convert query to embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Get nodes and their text representations
            cypher_query = """
            MATCH (n)
            RETURN n.name as name, labels(n) as labels, id(n) as id, 
                   properties(n) as props
            LIMIT 1000
            """
            nodes = self.query_knowledge_graph(cypher_query)
            
            # Calculate relevance scores using batch processing
            node_texts = [
                f"{' '.join(node['labels'])} {node['name']} " + 
                f"{' '.join([f'{k}:{v}' for k,v in node['props'].items()])}"
                for node in nodes
            ]
            node_embeddings = self.embedding_model.encode(node_texts)
            similarities = np.dot(node_embeddings, query_embedding)
            
            # Filter nodes by similarity threshold
            relevant_indices = np.where(similarities > similarity_threshold)[0]
            relevant_nodes = [nodes[i]['id'] for i in relevant_indices]
            
            # Sort by similarity and take top nodes
            top_nodes = sorted(
                zip(relevant_nodes, similarities[relevant_indices]),
                key=lambda x: x[1],
                reverse=True
            )[:max_nodes]
            
            top_node_ids = [str(node[0]) for node in top_nodes]
            
            # Extract subgraph
            subgraph_query = f"""
            MATCH (n1)
            WHERE id(n1) IN {top_node_ids}
            MATCH (n1)-[r]-(n2)
            WHERE id(n2) IN {top_node_ids}
            RETURN n1, r, n2, type(r) as relationship_type
            """
            relationships = self.query_knowledge_graph(subgraph_query)
            
            # Build NetworkX graph
            G = nx.Graph()
            for rel in relationships:
                G.add_edge(
                    rel['n1']['name'],
                    rel['n2']['name'],
                    relationship=rel['relationship_type']
                )
            
            return G
            
        except Exception as e:
            self.logger.error(f"Error getting relevant subgraph: {str(e)}")
            return nx.Graph()

    def generate_graph_context(self, subgraph: nx.Graph) -> str:
        """Convert subgraph to structured text context."""
        context_parts = []
        
        # Add nodes
        nodes = list(subgraph.nodes())
        if nodes:
            context_parts.append("Entities:")
            for node in nodes:
                context_parts.append(f"- {node}")
        
        # Add relationships
        edges = list(subgraph.edges(data=True))
        if edges:
            context_parts.append("\nRelationships:")
            for edge in edges:
                context_parts.append(
                    f"- {edge[0]} {edge[2]['relationship']} {edge[1]}"
                )
        
        return "\n".join(context_parts)

    def format_prompt(self, query: str, context: str) -> str:
        """Format prompt for the LLM using instruction tuning format."""
        return f"""<s>[INST] Given this knowledge graph information:

{context}

Answer this question: {query}

Use only the information provided in the knowledge graph. If you need additional information that's not available, please mention it. [/INST]"""

    def query_llm(self, user_query: str) -> Dict[str, Any]:
        """Process user query using KG and LLM."""
        try:
            # Get relevant subgraph
            subgraph = self.get_relevant_subgraph(user_query)
            
            # Generate context
            context = self.generate_graph_context(subgraph)
            
            # Prepare prompt
            prompt = self.format_prompt(user_query, context)
            
            # Generate response
            response = self.llm(
                prompt,
                max_length=self.max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )[0]['generated_text']
            
            # Extract the actual response (after the instruction)
            response = response.split("[/INST]")[-1].strip()
            
            return {
                "answer": response,
                "subgraph": subgraph,
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Error in query_llm: {str(e)}")
            return {
                "answer": "Error processing query",
                "subgraph": nx.Graph(),
                "context": ""
            }

    def explain_reasoning(
        self,
        query: str,
        answer: str,
        subgraph: nx.Graph
    ) -> str:
        """Explain the reasoning process using the knowledge graph."""
        try:
            explanation_prompt = f"""<s>[INST] I found this answer: "{answer}"
            
Based on this knowledge graph:
{self.generate_graph_context(subgraph)}

Explain step by step how the knowledge graph information was used to arrive at this answer.
Include any assumptions made and mention any missing information that would have been helpful. [/INST]"""
            
            explanation = self.llm(
                explanation_prompt,
                max_length=self.max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )[0]['generated_text']
            
            return explanation.split("[/INST]")[-1].strip()
            
        except Exception as e:
            self.logger.error(f"Error in explain_reasoning: {str(e)}")
            return "Error generating explanation"

    def close(self):
        """Clean up resources."""
        self.driver.close()
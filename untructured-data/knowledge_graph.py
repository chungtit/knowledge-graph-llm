import spacy
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import re

class KnowledgeGraphExtractor:
    def __init__(self, model="en_core_web_lg"):
        """
        Initialize the knowledge graph extractor with a comprehensive NLP pipeline
        
        Args:
            model (str): SpaCy language model to use. 
            Recommend en_core_web_lg for more comprehensive entity recognition
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Model {model} not found. Please download it using:")
            print(f"python -m spacy download {model}")
            raise

    def preprocess_text(self, text):
        """
        Preprocess the text to clean and prepare for NLP processing
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove problematic characters if needed
        text = text.replace('\n', ' ')
        
        return text

    def extract_entities_and_relations(self, text):
        """
        Extract named entities, their types, and potential relationships
        
        Args:
            text (str): Preprocessed text to analyze
        
        Returns:
            tuple: Entities and their relationships
        """
        # Preprocess the text first
        processed_text = self.preprocess_text(text)
        
        # Process with spaCy
        doc = self.nlp(processed_text)
        
        # Collect entities
        entities = []
        entity_types = {}
        
        for ent in doc.ents:
            if ent.text not in entities:
                entities.append(ent.text)
                entity_types[ent.text] = ent.label_
        
        # Extract potential relationships using dependency parsing
        relationships = defaultdict(list)
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ['nsubj', 'dobj', 'pobj', 'attr']:
                    # Look for subject-verb-object relationships
                    subject = None
                    verb = None
                    obj = None
                    
                    for child in token.children:
                        if child.dep_ == 'nsubj':
                            subject = child.text
                        elif child.dep_ in ['dobj', 'pobj']:
                            obj = child.text
                    
                    # If a meaningful relationship is found
                    if subject and token.pos_ == 'VERB' and obj:
                        relationships[subject].append({
                            'verb': token.text,
                            'object': obj
                        })
        
        return entities, entity_types, relationships

    def build_knowledge_graph(self, text):
        """
        Build a comprehensive knowledge graph from the text
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            networkx.DiGraph: Knowledge graph
        """
        # Extract entities and relationships
        entities, entity_types, relationships = self.extract_entities_and_relations(text)
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with their types
        for entity in entities:
            G.add_node(entity, type=entity_types.get(entity, 'UNKNOWN'))
        
        # Add edges based on relationships
        for subject, rels in relationships.items():
            for rel in rels:
                G.add_edge(subject, rel['object'], 
                          relation=rel['verb'])
        
        return G

    def visualize_graph(self, G, output_file='knowledge_graph.png'):
        """
        Visualize the knowledge graph
        
        Args:
            G (networkx.DiGraph): Input graph
            output_file (str): Path to save the graph visualization
        """
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(G, seed=42, k=0.5)  # k controls the distance between nodes
        
        # Color nodes by type
        node_colors = []
        color_map = {
            'PERSON': 'lightblue',
            'ORG': 'lightgreen',
            'GPE': 'salmon',
            'UNKNOWN': 'lightgray'
        }
        
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'UNKNOWN')
            node_colors.append(color_map.get(node_type, 'lightgray'))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                node_size=1000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                arrows=True, alpha=0.5)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
        
        # Edge labels (relationships)
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                     font_size=6)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        
        # Save or show the graph
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def graph_to_dict(self, G):
        """
        Convert graph to a dictionary representation
        
        Args:
            G (networkx.DiGraph): Input graph
        
        Returns:
            dict: Graph representation
        """
        graph_dict = {
            'nodes': [],
            'edges': []
        }
        
        for node, data in G.nodes(data=True):
            graph_dict['nodes'].append({
                'name': node,
                'type': data.get('type', 'UNKNOWN')
            })
        
        for source, target, data in G.edges(data=True):
            graph_dict['edges'].append({
                'source': source,
                'target': target,
                'relation': data.get('relation', 'connected')
            })
        
        return graph_dict

# Example usage
def main():
    # Example long text (replace with your actual long text)
    with open('long_text_file_path.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Initialize extractor
    extractor = KnowledgeGraphExtractor()
    
    # Build knowledge graph
    graph = extractor.build_knowledge_graph(text)
    
    # Visualize graph
    extractor.visualize_graph(graph)
    
    # Convert to dictionary for further processing
    graph_dict = extractor.graph_to_dict(graph)
    
    # Print some stats
    print(f"Total Nodes: {len(graph.nodes)}")
    print(f"Total Edges: {len(graph.edges)}")

if __name__ == "__main__":
    main()
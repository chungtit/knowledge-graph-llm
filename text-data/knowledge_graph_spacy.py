import re
import spacy
import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraphExtractorSpacy:
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
            print(f"Downloading {model} model...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)

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

    def resolve_coreferences(self, text):
        """
        Resolve coreferences in the text
        
        Args:
            text (str): Input text
        
        Returns:
            str: Text with resolved coreferences
        """
        # Process the text
        doc = self.nlp(text)
        
        # Create a list to store resolved tokens
        resolved_tokens = []
        
        # Track the last known named entity
        last_named_entity = None
        
        for token in doc:
            # If token is a pronoun (he, she, it, etc.)
            if token.pos_ == "PRON":
                # If we have a last known named entity, replace the pronoun
                if last_named_entity:
                    resolved_tokens.append(last_named_entity)
                else:
                    resolved_tokens.append(token.text)
            else:
                # If token is a named entity, update last_named_entity
                if token.ent_type_:
                    last_named_entity = token.text
                resolved_tokens.append(token.text)
        
        # Reconstruct the text
        return " ".join(resolved_tokens)

    def extract_entities(self, text):
        """
        Extract named entities and their types from the text
        
        Args:
            text (str): Preprocessed text to analyze
        
        Returns:
            tuple: Entities and their types
        """
        # First, resolve coreferences
        resolved_text = self.resolve_coreferences(text)
        
        # Process with spaCy
        doc = self.nlp(resolved_text)
        
        # Collect entities
        entities = []
        entity_types = {}
        
        for ent in doc.ents:
            if ent.text not in entities:
                entities.append(ent.text)
                entity_types[ent.text] = ent.label_
        
        return entities, entity_types

    def extract_svo_relationships(self, text):
        """
        Extract Subject-Verb-Object relationships from a given text using dependency parsing.
        
        Args:
            text (str): Input text to parse
        
        Returns:
            list: A list of dictionaries containing SVO relationships
        """
        # First, resolve coreferences
        resolved_text = self.resolve_coreferences(text)
        
        # Process with spaCy
        doc = self.nlp(resolved_text)
        
        # List to store SVO relationships
        svo_relationships = []
        
        # Iterate through sentences in the document
        for sent in doc.sents:
            # Dictionaries to store subject, verb, and object
            current_svo = {
                "subject": None,
                "verb": None,
                "object": None,
                "subject_type": None,
                "object_type": None
            }
            
            # Iterate through tokens in the sentence
            for token in sent:
                # Find the verb (ROOT of the sentence)
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    current_svo["verb"] = token.lemma_
                    
                    # Look for subject (nsubj or nsubjpass)
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            current_svo["subject"] = child.lemma_
                    
                    # Look for direct object (dobj)
                    for child in token.children:
                        if child.dep_ == "dobj":
                            current_svo["object"] = child.lemma_
            
            # Attempt to get types for subject and object
            if current_svo["subject"] or current_svo["object"]:
                # Reprocess to get entity types
                entity_sent_doc = self.nlp(sent.text)
                for ent in entity_sent_doc.ents:
                    if current_svo["subject"] and ent.text.lower() == current_svo["subject"].lower():
                        current_svo["subject_type"] = ent.label_
                    if current_svo["object"] and ent.text.lower() == current_svo["object"].lower():
                        current_svo["object_type"] = ent.label_
            
            # Only add if we found a meaningful relationship
            if current_svo["subject"] and current_svo["verb"] and current_svo["object"]:
                svo_relationships.append(current_svo)
        
        return svo_relationships

    def merge_similar_entities(self, entities, entity_types):
        """
        Merge similar entities, prioritizing full names or more specific entities
        
        Args:
            entities (list): List of entities
            entity_types (dict): Dictionary of entity types
        
        Returns:
            tuple: Merged entities and their types
        """
        # Create a mapping to merge entities
        entity_mapping = {}
        merged_entities = []
        merged_types = {}
        
        for entity in entities:
            # Prefer longer, more specific entities
            matched = False
            for existing in merged_entities:
                if entity.lower() in existing.lower() or existing.lower() in entity.lower():
                    # Keep the longer, more specific entity
                    if len(entity) > len(existing):
                        entity_mapping[existing] = entity
                        merged_entities.remove(existing)
                        merged_entities.append(entity)
                        merged_types[entity] = entity_types.get(entity, entity_types.get(existing, 'UNKNOWN'))
                    matched = True
                    break
            
            if not matched:
                merged_entities.append(entity)
                merged_types[entity] = entity_types.get(entity, 'UNKNOWN')
        
        return merged_entities, merged_types

    def build_knowledge_graph(self, text):
        """
        Build a comprehensive knowledge graph from the text
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            networkx.DiGraph: Knowledge graph
        """
        # Extract entities
        entities, entity_types = self.extract_entities(text)
        
        # Merge similar entities
        entities, entity_types = self.merge_similar_entities(entities, entity_types)
        
        # Extract SVO relationships
        svo_relationships = self.extract_svo_relationships(text)
        
        # Create graph
        G = nx.DiGraph()
        
        # Identify the main entity (first entity)
        main_entity = entities[0] if entities else None
        main_entity_type = entity_types.get(main_entity, 'UNKNOWN') if main_entity else 'UNKNOWN'
        
        # Add nodes with their types
        for entity in entities:
            G.add_node(entity, type=entity_types.get(entity, 'UNKNOWN'))
        
        # Add edges based on SVO relationships
        for rel in svo_relationships:
            if rel['subject'] and rel['object']:
                # If the main entity is involved, prioritize its connections
                if rel['subject'] == main_entity:
                    G.add_edge(rel['subject'], rel['object'], 
                              relation=rel['verb'],
                              subject_type=rel.get('subject_type', 'UNKNOWN'),
                              object_type=rel.get('object_type', 'UNKNOWN'),
                              weight=2)  # Higher weight for main entity connections
                elif rel['object'] == main_entity:
                    G.add_edge(rel['subject'], rel['object'], 
                              relation=rel['verb'],
                              subject_type=rel.get('subject_type', 'UNKNOWN'),
                              object_type=rel.get('object_type', 'UNKNOWN'),
                              weight=2)  # Higher weight for main entity connections
                else:
                    G.add_edge(rel['subject'], rel['object'], 
                              relation=rel['verb'],
                              subject_type=rel.get('subject_type', 'UNKNOWN'),
                              object_type=rel.get('object_type', 'UNKNOWN'),
                              weight=1)
        
        # Set graph-level attributes for the main entity
        G.graph['main_entity'] = main_entity
        G.graph['main_entity_type'] = main_entity_type
        
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
        
        # Color nodes by type, with special highlighting for main entity
        node_colors = []
        color_map = {
            'PERSON': 'lightblue',
            'ORG': 'lightgreen',
            'GPE': 'salmon',
            'UNKNOWN': 'lightgray'
        }
        
        main_entity = G.graph.get('main_entity')
        main_entity_type = G.graph.get('main_entity_type', 'UNKNOWN')
        
        for node in G.nodes():
            if node == main_entity:
                node_colors.append('gold')  # Highlight main entity
            else:
                node_type = G.nodes[node].get('type', 'UNKNOWN')
                node_colors.append(color_map.get(node_type, 'lightgray'))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                node_size=1000, alpha=0.8)
        
        # Draw edges with varying thickness based on weight
        edge_weights = [G[u][v].get('weight', 1) for (u, v) in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                arrows=True, alpha=0.5,
                                width=[w * 1.5 for w in edge_weights])
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
        
        # Edge labels (relationships)
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                     font_size=6)
        
        plt.title(f"Knowledge Graph (Main Entity: {main_entity} - {main_entity_type})")
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
            'main_entity': G.graph.get('main_entity', 'UNKNOWN'),
            'main_entity_type': G.graph.get('main_entity_type', 'UNKNOWN'),
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
                'relation': data.get('relation', 'connected'),
                'subject_type': data.get('subject_type', 'UNKNOWN'),
                'object_type': data.get('object_type', 'UNKNOWN'),
                'weight': data.get('weight', 1)
            })
        
        return graph_dict
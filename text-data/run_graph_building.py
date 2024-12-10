import argparse
from knowledge_graph_spacy import KnowledgeGraphExtractorSpacy

def parse_arguments():
        """
        Parse command-line arguments for the Knowledge Graph Extractor.
        
        Returns:
            argparse.Namespace: Parsed command-line arguments
        """
        parser = argparse.ArgumentParser(
            description='Extract and visualize knowledge graphs from text files',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            '-i', '--input_txt', 
            type=str, 
            help='Path to the input text file to process'
        )
        
        parser.add_argument(
            '-o', '--output_png', 
            type=str, 
            default='knowledge_graph_spacy.png',
            help='Path to save the output graph visualization'
        )
        
        parser.add_argument(
            '-m', '--spacy_model', 
            type=str, 
            default='en_core_web_lg',
            help='SpaCy language model to use for NLP processing'
        )
        
        return parser.parse_args()

# Example usage
def main():
    # ArgParser
    args = parse_arguments()
    # Example text (replace with your actual text)
    with open(args.input_txt, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Initialize extractor
    extractor = KnowledgeGraphExtractorSpacy(model=args.spacy_model)
    
    # Extract and print SVO relationships
    print("SVO Relationships:")
    svo_relationships = extractor.extract_svo_relationships(text)
    for rel in svo_relationships:
        print(f"Subject: {rel['subject']} ({rel.get('subject_type', 'N/A')}) "
              f"- Verb: {rel['verb']} - "
              f"Object: {rel['object']} ({rel.get('object_type', 'N/A')})")
    
    # Build knowledge graph
    graph = extractor.build_knowledge_graph(text)
    
    # Visualize graph
    extractor.visualize_graph(graph, output_file=args.output_png)
    
    # Convert to dictionary for further processing
    # graph_dict = extractor.graph_to_dict(graph)
    
    # Print some stats and main entity info
    print(f"\nMain Entity: {graph.graph.get('main_entity', 'N/A')}")
    print(f"Main Entity Type: {graph.graph.get('main_entity_type', 'N/A')}")
    print(f"Total Nodes: {len(graph.nodes)}")
    print(f"Total Edges: {len(graph.edges)}")
    # print(graph_dict)

if __name__ == "__main__":
    main()
    # python text-data/run_graph_building.py -i ./text-data/demo-data/long_text.txt -o ./text-data/images/knowledge_graph_spacy.png -m "en_core_web_lg"
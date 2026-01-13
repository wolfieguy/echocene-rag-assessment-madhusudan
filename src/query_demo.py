import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add project root to path
from src.rag import query_rag
from src.metrics_logger import log_metrics, visualize_metrics



# Demo queries from the task description
DEMO_QUERIES = [
    "For a medium-sized German construction company, detail the main CSRD reporting thresholds, scopes, and timelines.",
    "Explain EU Taxonomy alignment criteria for sustainable building renovations, including incentives for reducing embodied carbon.",
    "Outline key GEG requirements and compliance risks for heat pump systems in new commercial buildings in Germany."
    #"Explain CSRD reporting thresholds"#, scopes, and timelines for a medium-sized German construction company."
]




if __name__ == "__main__":
    
    for q in DEMO_QUERIES: # loop through demo queries
        print(f"\nRunning: {q}")
        result = query_rag(q)
        log_metrics(q, result)
       
        
        
        print("Retrieved Chunks:") 
        for content, metadata, score in result["retrieved_chunks"]: # loop through retrieved chunks 
            source = metadata.get('source', 'unknown source').split('/')[-1] if metadata.get('source') else 'unknown' # get source filename and metadata
            page = metadata.get('page', '?')
            print(f"Score: {score:.2f} | Source: {source}, Page: {page}")
            print(f"{content[:200]}...\n")
        
        
        
       
        print(f"Metrics: Latency: {result['latency_sec']:.2f}s, Chunks: {result['num_chunks']}, Avg Score: {result['avg_score']:.2f}") # print metrics
        print(f"Answer: {result['answer']}")
   
    
   
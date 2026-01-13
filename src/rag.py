from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import time


load_dotenv()



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # same embedding model as ingest.py


vectorstore = Chroma(
    persist_directory="/Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/chroma_db",
    embedding_function=embeddings
) # Load existing Chroma vector store



llm = ChatGroq(model="llama-3.1-8b-instant")  # LLM for RAG – Groq Llama 3.1 Instant (8B) for balance of cost/performance

# Defensive prompt – strict to avoid hallucinations in compliance data 
system_prompt = """
You are an expert in EU regulatory sustainability documents (CSRD, GEG, EU Taxonomy).
Answer ONLY using the provided context. Be precise, factual, and cite sources.
Use format: [source: filename, page: X]

If the context does not contain enough information, respond exactly:
"Insufficient information in the provided sources."

Do NOT hallucinate, guess, or use external knowledge.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion: {input}")
])


# === Hybrid Retrieval Setup ===
# Inspired by Lewis et al. RAG paper (2020) – combining keyword (BM25) + semantic for better recall on exact regulatory terms


docs_in_index = vectorstore.get()["documents"] # load all docs from vector store/Chroma
if not docs_in_index:
    raise ValueError("Index is empty – re-run ingest.py to populate documents first.") 

# Keyword-based (BM25) – good for exact matches like "CSRD thresholds"
my_keyword_retriever = BM25Retriever.from_texts(docs_in_index)
my_keyword_retriever.k = 4  # Top 4 keyword results – kept low to balance


my_semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Top 4 semantic results – captures related concepts



my_hybrid_retriever = EnsembleRetriever(
    retrievers=[my_keyword_retriever, my_semantic_retriever],
    weights=[0.3, 0.7]  
) # ensemble retriever with 30(precision)/70(semantic recall) weighting



rag_chain = create_retrieval_chain(my_hybrid_retriever, prompt | llm) # RAG chain with hybrid retriever using prompt and LLM


def query_rag(input_query):
    """
    Run query through hybrid RAG pipeline and return metrics.
    - Uses hybrid for retrieval, falls back to semantic for reliable scores/metadata
    - Human note: Hybrid sometimes loses metadata from BM25 – using semantic fallback
    """
    start_time = time.time()

    
    response = rag_chain.invoke({"input": input_query}) # hybrid retrieval via RAG chain
    retrieved = response.get("context", [])  # Docs from hybrid

    
    semantic_docs_and_scores = vectorstore.similarity_search_with_score(input_query, k=len(retrieved) or 8) # get reliable scores from semantic search
    semantic_docs = [doc for doc, _ in semantic_docs_and_scores] # extract docs
    semantic_scores = [score for _, score in semantic_docs_and_scores] # extract scores

    
    chunks_with_scores = [] # initialize list for chunks with scores
    for i, doc in enumerate(retrieved):
        if i < len(semantic_docs):
            meta = semantic_docs[i].metadata # get metadata from semantic doc
            sc = semantic_scores[i] # get score from semantic search
        else:
            meta = doc.metadata or {}  # Empty dict if None
            sc = 0.0
        chunks_with_scores.append((doc.page_content, meta, sc))

    latency = time.time() - start_time

    answer = response["answer"]

    

    if "[source:" not in answer:
        answer += "\n(Note: No explicit source citations detected – check retrieved chunks.)"


    
    print(f"Hybrid retrieved {len(retrieved)} chunks for query '{input_query}'") # debug feedback for testing


    return {
        "answer": answer,
        "retrieved_chunks": chunks_with_scores,
        "latency_sec": round(latency, 3),
        "num_chunks": len(retrieved),
        "avg_score": round(sum(sc for _, _, sc in chunks_with_scores) / len(chunks_with_scores), 4) if chunks_with_scores else 0.0
    }
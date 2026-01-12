from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time


load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # same embeddings as ingest.py

# Load existing Chroma vector store
vectorstore = Chroma(
    persist_directory="Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/chroma_db",
    embedding_function=embeddings
)




llm = ChatGroq(model="llama-3.1-8b-instant")   # Using Groq Llama 3.1 8B Instant model

# Strong defensive prompt for compliance 
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




retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # sample top 8 chunks


rag_chain = create_retrieval_chain(retriever, prompt | llm) # Initialising the RAG chain with prompt and LLM

def query_rag(input_query):
    
    start_time = time.time()
    docs_and_scores = vectorstore.similarity_search_with_score(input_query, k=8) # Get top 8 with scores 

    if not docs_and_scores:
        latency = time.time() - start_time
        return {
            "answer": "No relevant information found in the indexed documents.",
            "retrieved_chunks": [],
            "latency_sec": latency,
            "num_chunks": 0,
            "avg_score": 0.0
        }

    
    retrieved_docs = [doc for doc, _ in docs_and_scores] # retrieves Document objects
    scores = [score for _, score in docs_and_scores]  # float values





    print(f"Retrieved {len(docs_and_scores)} chunks for query: \"{input_query}\"") # print retrieval info
    for i, (doc, score) in enumerate(zip(retrieved_docs[:3], scores[:3])):
        source = doc.metadata.get("source", "unknown").split("/")[-1]  # get filename only
        page = doc.metadata.get("page", "?") # get page number
        print(f"Top {i+1}: score = {score:.4f} | {source} (page {page})") # print the top retrieved chunks and their scores and sources
        print(doc.page_content[:180].replace("\n", " ") + "...\n") # print snippet of content

    
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) # build context string from retrieved docs

    # Run the chain
    response = rag_chain.invoke({
        "input": input_query,
        "context": context_text
    })

    latency = time.time() - start_time

    answer = response["answer"]

    
    if "[source:" not in answer and "Insufficient information" not in answer:
        answer += "\n\n(Note: No explicit source citations detected in response.)" # for hallucination check

    
    chunks_with_scores = [
        (doc.page_content, doc.metadata, score)
        for doc, score in docs_and_scores
    ]# chunks with metadata and and their respective scores

    return {
        "answer": answer,
        "retrieved_chunks": chunks_with_scores,
        "latency_sec": round(latency, 3),
        "num_chunks": len(docs_and_scores),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0
    }
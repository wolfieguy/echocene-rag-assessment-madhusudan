import os
from langchain_community.document_loaders import PyPDFLoader  
from langchain_community.document_loaders import PyPDFDirectoryLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document


load_dotenv()


DATA_DIR = "/Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/data" # Local path for PDF documents
CHROMA_DIR = "/Users/madhusudangorabal/Desktop/Echoene-GitHub/echocene-rag-assessment-madhusudan/chroma_db"  # Local path for Chroma vector store



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # HuggingFaceEmbeddings model using MiniLM(L6-v2)

def ingest_documents(my_chunk_size=1200, overlap_buffer=300):

    docs = []
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')] # List all PDFs in data dir
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIR, pdf_file)
        try:
            loader = PyPDFLoader(file_path)  # Load each PDF
            sub_docs = loader.load() # Load returns list of Document objects
            docs.extend(sub_docs) # Aggregate all pages
            print(f"Successfully loaded {len(sub_docs)} pages from {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")  # This will show which file fails
    print(f"Total loaded {len(docs)} pages from all PDFs.")

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=my_chunk_size, chunk_overlap=overlap_buffer) # Using Recursive splitter for better context handling
    chunks = splitter.split_documents(docs) # Split all documents into chunks
    
    


    for chunk in chunks:
        chunk.metadata["source"] = chunk.metadata.get("source", "unknown.pdf")  # Fallback if source missing
        chunk.metadata["page"] = chunk.metadata.get("page", 0)  # 0 if page not detected
    
    print(f"Created {len(chunks)} chunks with size {my_chunk_size} and overlap {overlap_buffer}.")

    

    # Removing duplicate chunks based on content hash
    
    unique_chunks = [] # to hold deduplicated chunks
    seen = set() 
    for chunk in chunks:
        text_hash = hash(chunk.page_content.strip()) # simple hash of content
        if text_hash not in seen:
            seen.add(text_hash)
            unique_chunks.append(chunk) # Appends unique chunk
    chunks = unique_chunks
    print(f"Deduplicated to {len(chunks)} unique chunks.")


    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    ) # creates a vector store from chunks
    #vectorstore.persist()  # Save to disk
    print(f"Vector store persisted at {CHROMA_DIR}.")  # log feedback



    
    index_size_mb = 0
    for root, _, files in os.walk(CHROMA_DIR):
        for file in files:
            index_size_mb += os.path.getsize(os.path.join(root, file)) / (1024 * 1024) # index size in MB
    print(f"Index size: {index_size_mb:.2f} MB")  # logs for README insights

if __name__ == "__main__":
    print("Ingesting with my_chunk_size=1500 and overlap_buffer=400")
    ingest_documents(my_chunk_size=1500, overlap_buffer=400)
    